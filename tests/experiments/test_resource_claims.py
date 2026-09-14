"""CPU-only unit tests for atomic cross-host resource claims.

Tests prove:
1. Competing acquisition (exactly one winner)
2. Distinct device independence
3. Non-owner release rejection
4. Busy-device rejection after recheck (no killing/preemption)
5. Failed-launch cleanup (no leaked claims)
6. Retained claims while child process runs
7. CPU allocation oversubscription refusal under host cap
8. Launch helper / wrapper execution and environment isolation
9. Stale claim liveness verification (local dead cleanup vs remote refusal)
10. Clean integration with experiments.jobs.monitor
"""

from __future__ import annotations

import sqlite3  # noqa: F401 -- host ABI ordering
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pytest

from experiments.jobs import monitor
from experiments.jobs.claims import (
    CPUOversubscriptionError,
    DeviceBusyError,
    DeviceClaimConflictError,
    DeviceUnavailableError,
    InvalidTokenError,
    acquire_cpu_claim,
    acquire_device_claim,
    atomic_dir_lock,
    build_child_env,
    claim_resources,
    get_available_cores,
    get_canonical_host,
    get_cgroup_cpu_limit,
    get_claims_status,
    is_claim_active,
    launch,
    read_json_safe,
    release_cpu_claim,
    release_device_claim,
    run_supervised,
    terminate_and_reap_process_group,
    write_json_atomic,
)


def mock_gpu(
    uuid: str,
    index: int = 0,
    free_mb: float = 40000.0,
    total_mb: float = 48000.0,
    pids: list[int] | None = None,
) -> dict[str, Any]:
    """Helper to generate mock GPU probe dictionary."""
    return {
        "index": index,
        "uuid": uuid,
        "name": f"Mock-GPU-{index}",
        "memory_free_mb": free_mb,
        "memory_total_mb": total_mb,
        "active_pids": pids or [],
    }


MOCK_TEST_DEVICES = [
    mock_gpu("GPU-compete-uuid-001"),
    mock_gpu("GPU-device-alpha", 0),
    mock_gpu("GPU-device-beta", 1),
    mock_gpu("GPU-owner-test"),
    mock_gpu("GPU-busy-test"),
    mock_gpu("GPU-fail-launch"),
    mock_gpu("GPU-retained-test"),
    mock_gpu("GPU-isolated-uuid-42"),
    mock_gpu("GPU-dead-pid-test"),
    mock_gpu("GPU-monitor-target"),
    mock_gpu("GPU-ctx-test-99"),
    mock_gpu("GPU-interrupted-child"),
    mock_gpu("GPU-busy-pre-popen"),
    mock_gpu("GPU-mandatory-cpu"),
]


@pytest.fixture(autouse=True)
def mock_environment_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide mock GPU probe by default in CPU-only test environment."""
    from experiments.jobs import claims

    def query_with_fallback(probe_fn: Any = None) -> list[dict[str, Any]]:
        if probe_fn is not None:
            return probe_fn()
        return list(MOCK_TEST_DEVICES)

    monkeypatch.setattr(claims, "query_gpus", query_with_fallback)


def test_competing_acquisition_one_winner(tmp_path: Path) -> None:
    """Atomic acquire must give exactly one owner when competing processes request one device."""
    target_uuid = "GPU-compete-uuid-001"
    claims_dir = tmp_path / "claims"

    def worker(_: int) -> bool:
        try:
            claim = acquire_device_claim(
                device_uuid=target_uuid,
                claims_dir=claims_dir,
                job_id="compete-job",
            )
            return claim is not None
        except DeviceClaimConflictError:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(8)))

    assert sum(results) == 1, f"Expected exactly 1 winner, got {sum(results)}"
    status = get_claims_status(claims_dir)
    assert len(status["devices"]) == 1
    assert status["devices"][0]["device_uuid"] == target_uuid


def test_distinct_device_independence(tmp_path: Path) -> None:
    """Distinct devices can be claimed concurrently without interference."""
    claims_dir = tmp_path / "claims"
    uuid1 = "GPU-device-alpha"
    uuid2 = "GPU-device-beta"

    claim1 = acquire_device_claim(
        device_uuid=uuid1,
        claims_dir=claims_dir,
        job_id="job-alpha",
    )
    claim2 = acquire_device_claim(
        device_uuid=uuid2,
        claims_dir=claims_dir,
        job_id="job-beta",
    )

    assert claim1.device_uuid == uuid1
    assert claim2.device_uuid == uuid2
    assert claim1.token != claim2.token

    status = get_claims_status(claims_dir)
    uuids = {d["device_uuid"] for d in status["devices"]}
    assert uuids == {uuid1, uuid2}

    # Release device 1; device 2 must remain claimed
    release_device_claim(claim1.host, claim1.device_uuid, claim1.token, claims_dir)
    status_after = get_claims_status(claims_dir)
    remaining_uuids = {d["device_uuid"] for d in status_after["devices"]}
    assert remaining_uuids == {uuid2}

    # Release device 2
    release_device_claim(claim2.host, claim2.device_uuid, claim2.token, claims_dir)
    assert len(get_claims_status(claims_dir)["devices"]) == 0


def test_non_owner_release_rejection(tmp_path: Path) -> None:
    """Release rejected when token does not match the owning token."""
    claims_dir = tmp_path / "claims"
    target_uuid = "GPU-owner-test"

    claim = acquire_device_claim(
        device_uuid=target_uuid,
        claims_dir=claims_dir,
        job_id="owner-job",
    )

    # Attempt release with bogus token
    with pytest.raises(InvalidTokenError, match="token mismatch"):
        release_device_claim(claim.host, claim.device_uuid, "fraudulent-token-123", claims_dir)

    # Claim must remain active on disk
    status = get_claims_status(claims_dir)
    assert len(status["devices"]) == 1
    assert status["devices"][0]["token"] == claim.token

    # Legitimate owner release succeeds
    release_device_claim(claim.host, claim.device_uuid, claim.token, claims_dir)
    assert len(get_claims_status(claims_dir)["devices"]) == 0


def test_busy_device_rejection_after_recheck(tmp_path: Path) -> None:
    """Device free before selection but busy on recheck must abort and release claim."""
    claims_dir = tmp_path / "claims"
    target_uuid = "GPU-busy-test"

    call_count = [0]

    def mock_recheck(_uuid: str, _min_mem: float) -> bool:
        call_count[0] += 1
        # First call might be selection, recheck under claim returns False (colleague process)
        return False

    with pytest.raises(DeviceBusyError, match="became busy during pre-launch recheck"):
        acquire_device_claim(
            device_uuid=target_uuid,
            claims_dir=claims_dir,
            job_id="busy-job",
            recheck_fn=mock_recheck,
        )

    assert call_count[0] >= 1
    # The claim must be cleaned up / released, NOT leaked
    status = get_claims_status(claims_dir)
    assert len(status["devices"]) == 0


def test_failed_launch_cleanup(tmp_path: Path) -> None:
    """If child process fails to launch, both device and CPU claims are immediately released."""
    claims_dir = tmp_path / "claims"
    target_uuid = "GPU-fail-launch"

    with pytest.raises(FileNotFoundError):
        launch(
            ["/nonexistent/binary/that/cannot/exist/on/any/path_12345"],
            gpu_uuid=target_uuid,
            cpu_threads=4,
            claims_dir=claims_dir,
            available_cores=16,
        )

    # Verify both device claim and CPU allocation are empty
    status = get_claims_status(claims_dir)
    assert len(status["devices"]) == 0
    for host_info in status["cpu_hosts"].values():
        assert len(host_info.get("allocations", {})) == 0


def test_retained_claims_while_child_runs(tmp_path: Path) -> None:
    """Claims must be retained while the child process runs and released upon termination."""
    claims_dir = tmp_path / "claims"
    target_uuid = "GPU-retained-test"

    proc, session = launch(
        [sys.executable, "-c", "import time; time.sleep(0.4)"],
        gpu_uuid=target_uuid,
        cpu_threads=2,
        claims_dir=claims_dir,
        available_cores=8,
    )

    # While child is running, competing acquire must fail
    assert proc.poll() is None
    with pytest.raises(DeviceClaimConflictError):
        acquire_device_claim(
            device_uuid=target_uuid,
            claims_dir=claims_dir,
            job_id="competing-during-run",
        )

    # Wait for child to exit
    exit_code = proc.wait()
    assert exit_code == 0

    # Explicit release of session claims
    from experiments.jobs.claims import release_session_claims

    release_session_claims(session)

    # After release, device is available again
    new_claim = acquire_device_claim(
        device_uuid=target_uuid,
        claims_dir=claims_dir,
        job_id="subsequent-job",
    )
    assert new_claim.device_uuid == target_uuid
    release_device_claim(new_claim.host, new_claim.device_uuid, new_claim.token, claims_dir)


def test_cpu_allocation_oversubscription_refusal(tmp_path: Path) -> None:
    """CPU threads per host are capped at floor(0.90 * available_cores); oversubscription is refused."""
    claims_dir = tmp_path / "claims"
    # Available cores = 10 -> cap = floor(0.90 * 10) = 9
    claim1 = acquire_cpu_claim(
        threads=6,
        claims_dir=claims_dir,
        available_cores=10,
        job_id="cpu-job-1",
    )
    assert claim1.threads == 6

    # 6 + 4 = 10 > cap (9) -> refused!
    with pytest.raises(CPUOversubscriptionError, match="exceeds cap 9"):
        acquire_cpu_claim(
            threads=4,
            claims_dir=claims_dir,
            available_cores=10,
            job_id="cpu-job-2",
        )

    # 6 + 3 = 9 == cap (9) -> fits exactly!
    claim2 = acquire_cpu_claim(
        threads=3,
        claims_dir=claims_dir,
        available_cores=10,
        job_id="cpu-job-3",
    )
    assert claim2.threads == 3

    # Now at full capacity (9/9) -> even 1 thread must be refused
    with pytest.raises(CPUOversubscriptionError):
        acquire_cpu_claim(
            threads=1,
            claims_dir=claims_dir,
            available_cores=10,
            job_id="cpu-job-4",
        )

    # Release claim 1
    release_cpu_claim(claim1.host, claim1.token, claims_dir)

    # Now 3 allocated, 6 available -> 4 threads can be allocated
    claim4 = acquire_cpu_claim(
        threads=4,
        claims_dir=claims_dir,
        available_cores=10,
        job_id="cpu-job-4",
    )
    assert claim4.threads == 4

    release_cpu_claim(claim2.host, claim2.token, claims_dir)
    release_cpu_claim(claim4.host, claim4.token, claims_dir)


def test_launch_helper_execution_and_env_isolation(tmp_path: Path) -> None:
    """Launch helper sets CUDA_VISIBLE_DEVICES, thread caps, and cleans up on completion."""
    claims_dir = tmp_path / "claims"
    target_uuid = "GPU-isolated-uuid-42"
    output_file = tmp_path / "env_output.json"

    script = (
        "import os, json, sys\n"
        "data = {\n"
        "    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
        "    'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),\n"
        "    'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),\n"
        "    'PS_CLAIMED_GPU_UUID': os.environ.get('PS_CLAIMED_GPU_UUID'),\n"
        "}\n"
        f"open(r'{output_file}', 'w').write(json.dumps(data))\n"
    )

    exit_code = run_supervised(
        [sys.executable, "-c", script],
        gpu_uuid=target_uuid,
        cpu_threads=3,
        claims_dir=claims_dir,
        available_cores=8,
    )

    assert exit_code == 0
    assert output_file.exists()
    import json

    data = json.loads(output_file.read_text())
    assert data["CUDA_VISIBLE_DEVICES"] == target_uuid
    assert data["PS_CLAIMED_GPU_UUID"] == target_uuid
    assert data["OMP_NUM_THREADS"] == "3"
    assert data["MKL_NUM_THREADS"] == "3"

    # Verify claims were cleanly released
    status = get_claims_status(claims_dir)
    assert len(status["devices"]) == 0
    for host_info in status["cpu_hosts"].values():
        assert len(host_info.get("allocations", {})) == 0


def test_stale_claim_remote_never_steals_on_ttl(tmp_path: Path) -> None:
    """Remote claims cannot be stolen merely because of elapsed time."""
    claims_dir = tmp_path / "claims"
    remote_host = "remote-node-99.domain"
    target_uuid = "GPU-remote-held"
    devices_dir = claims_dir / "devices"
    lock_dir = devices_dir / f"{remote_host}_{target_uuid}"
    lock_dir.mkdir(parents=True, exist_ok=True)

    # Write a claim that is 10 days old with a non-existent PID
    old_timestamp = time.time() - 86400 * 10
    from experiments.jobs.claims import write_json_atomic

    write_json_atomic(
        lock_dir / "claim.json",
        {
            "token": "remote-token-old",
            "host": remote_host,
            "device_uuid": target_uuid,
            "device_ordinal": 0,
            "child_ordinals": {target_uuid: 0},
            "pid": 99999999,
            "created_at": old_timestamp,
            "job_id": "remote-old-job",
        },
    )

    # A local acquisition attempt must NOT steal the claim
    # Note: acquire_device_claim uses get_canonical_host() for its own key,
    # so we simulate an acquisition targeting the exact remote lock directly:
    from experiments.jobs.claims import is_claim_active, read_json_safe

    existing_claim = read_json_safe(lock_dir / "claim.json")
    # Liveness check on remote claim without terminal job status returns None (unverifiable)
    assert is_claim_active(existing_claim) is None


def test_stale_claim_local_verified_dead_cleanup(tmp_path: Path) -> None:
    """A claim on the local host with a confirmed dead PID is safely reclaimed."""
    claims_dir = tmp_path / "claims"
    from experiments.jobs.claims import get_canonical_host, write_json_atomic

    local_host = get_canonical_host()
    target_uuid = "GPU-dead-pid-test"
    devices_dir = claims_dir / "devices"
    lock_dir = devices_dir / f"{local_host}_{target_uuid}"
    lock_dir.mkdir(parents=True, exist_ok=True)

    # Find a definitely non-existent PID (e.g. 4194300 on Linux)
    dead_pid = 4194300
    write_json_atomic(
        lock_dir / "claim.json",
        {
            "token": "stale-dead-token",
            "host": local_host,
            "device_uuid": target_uuid,
            "device_ordinal": 0,
            "child_ordinals": {target_uuid: 0},
            "pid": dead_pid,
            "created_at": time.time() - 3600,
            "job_id": "crashed-job",
        },
    )

    # Acquisition on the same host must notice the dead PID, clean it up, and succeed
    claim = acquire_device_claim(
        device_uuid=target_uuid,
        claims_dir=claims_dir,
        job_id="rescuing-job",
    )
    assert claim.device_uuid == target_uuid
    assert claim.token != "stale-dead-token"

    release_device_claim(claim.host, claim.device_uuid, claim.token, claims_dir)


def test_monitor_supervise_integration(tmp_path: Path) -> None:
    """Detached monitor.supervise acquires claims, isolates child, and releases upon exit."""
    job_dir = tmp_path / "job-001"
    job_dir.mkdir(parents=True)
    (job_dir / "events").mkdir()
    claims_dir = tmp_path / "claims"
    target_uuid = "GPU-monitor-target"
    output_file = job_dir / "child_env.json"

    script = (
        "import os, json\n"
        "data = {'CUDA': os.environ.get('CUDA_VISIBLE_DEVICES'), 'OMP': os.environ.get('OMP_NUM_THREADS')}\n"
        f"open(r'{output_file}', 'w').write(json.dumps(data))\n"
    )

    monitor.write_json(
        job_dir / "policy.json",
        {
            "report_at": None,
            "repeat_seconds": None,
            "quiet_until": 0,
            "urgent_during_quiet": False,
            "stall_seconds": 100,
        },
    )
    monitor.write_json(
        job_dir / "request.json",
        {
            "command": [sys.executable, "-c", script],
            "cwd": str(job_dir),
            "budget_seconds": 30,
            "thread": None,
            "codex": "codex",
            "claims": {
                "gpu": target_uuid,
                "cpu_threads": 2,
                "claims_dir": str(claims_dir),
            },
        },
    )

    assert monitor.supervise(job_dir) == 0

    # Verify status.json shows complete
    status = monitor.read_json(job_dir / "status.json")
    assert status["status"] == "complete"

    # Verify child saw isolated CUDA_VISIBLE_DEVICES and thread cap
    import json

    env_data = json.loads(output_file.read_text())
    assert env_data["CUDA"] == target_uuid
    assert env_data["OMP"] == "2"

    # Verify claims were released after supervisor completion
    claims_status = get_claims_status(claims_dir)
    assert len(claims_status["devices"]) == 0
    for host_info in claims_status["cpu_hosts"].values():
        assert len(host_info.get("allocations", {})) == 0


def test_claim_resources_context_manager(tmp_path: Path) -> None:
    """claim_resources context manager acquires claims, builds child env, and auto-releases."""
    claims_dir = tmp_path / "claims"
    target_uuid = "GPU-ctx-test-99"

    with claim_resources(
        gpu_uuid=target_uuid,
        cpu_threads=3,
        claims_dir=claims_dir,
        available_cores=8,
    ) as session:
        assert session.device_claim is not None
        assert session.device_claim.device_uuid == target_uuid
        assert session.cpu_claim is not None
        assert session.cpu_claim.threads == 3

        env = build_child_env(session, {"PATH": "/bin"})
        assert env["CUDA_VISIBLE_DEVICES"] == target_uuid
        assert env["OMP_NUM_THREADS"] == "3"
        assert env["MKL_NUM_THREADS"] == "3"

        # Active while inside context
        status = get_claims_status(claims_dir)
        assert len(status["devices"]) == 1

    # Cleaned up upon exit
    status_after = get_claims_status(claims_dir)
    assert len(status_after["devices"]) == 0
    for host_info in status_after["cpu_hosts"].values():
        assert len(host_info.get("allocations", {})) == 0


def test_mutex_no_ttl_takeover_and_token_safe_release(tmp_path: Path) -> None:
    """Blocker 1: Eliminate TTL-only takeover.

    A held mutex aged past 30s cannot be stolen by another process while the owner is alive.
    Verified liveness recovery only succeeds for confirmed dead local PIDs.
    Non-owner release is rejected (token-safe release).
    """
    claims_dir = tmp_path / "claims"
    lock_dir = claims_dir / "mutex_ttl_test.lock"

    with atomic_dir_lock(lock_dir, timeout=1.0):
        owner_file = lock_dir / "owner.json"
        assert owner_file.exists()
        owner_data = read_json_safe(owner_file)
        assert owner_data["pid"] == os.getpid()

        # Age the lock metadata to 45s (older than 30s)
        aged_data = dict(owner_data)
        aged_data["created_at"] = time.time() - 45.0
        write_json_atomic(owner_file, aged_data)

        # Another process attempt must timeout: NO TTL-only takeover while owner is alive!
        with pytest.raises(TimeoutError, match="Failed to acquire atomic directory lock"):
            with atomic_dir_lock(lock_dir, timeout=0.15, poll_interval=0.02):
                pass

        # Remote owner: even if aged 100 days, remote lock must NEVER be stolen
        remote_data = dict(aged_data)
        remote_data["host"] = "remote-cluster-node-99.domain"
        remote_data["created_at"] = time.time() - 86400 * 100
        write_json_atomic(owner_file, remote_data)

        with pytest.raises(TimeoutError):
            with atomic_dir_lock(lock_dir, timeout=0.15, poll_interval=0.02):
                pass

        # Restore original token and local host for token-safe exit
        write_json_atomic(owner_file, owner_data)

    # After exit, lock is cleanly released
    assert not lock_dir.exists()

    # Recovery with confirmed dead local PID:
    lock_dir.mkdir(parents=True, exist_ok=True)
    dead_pid = 4194300
    write_json_atomic(
        owner_file,
        {
            "token": "stale-dead-token",
            "pid": dead_pid,
            "host": get_canonical_host(),
            "created_at": time.time() - 100.0,
        },
    )
    # Acquisition must recover the abandoned lock from dead PID
    with atomic_dir_lock(lock_dir, timeout=0.5):
        new_meta = read_json_safe(owner_file)
        assert new_meta["token"] != "stale-dead-token"
        assert new_meta["pid"] == os.getpid()

    assert not lock_dir.exists()


def test_run_supervised_child_interruption_reaped_and_claims_handling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blocker 2: Interrupted run_supervised terminates and reaps process group before release.

    If child death cannot be verified, claims are retained as unresolved.
    """
    claims_dir = tmp_path / "claims"
    target_uuid = "GPU-interrupted-child"

    # Case A: Interruption reaps process group and releases claims
    child_marker = tmp_path / "child_running.txt"
    script = (
        "import time, sys\n"
        f"open(r'{child_marker}', 'w').write('started')\n"
        "time.sleep(30)\n"
    )

    child_pid_holder: list[int] = []
    call_count = [0]
    orig_wait = subprocess.Popen.wait

    def mock_wait_interrupted(self: subprocess.Popen[Any], *args: Any, **kwargs: Any) -> Any:
        call_count[0] += 1
        if call_count[0] == 1:
            child_pid_holder.append(self.pid)
            for _ in range(50):
                if child_marker.exists():
                    break
                time.sleep(0.05)
            raise RuntimeError("Simulated user interrupt")
        return orig_wait(self, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "wait", mock_wait_interrupted)

    with pytest.raises(RuntimeError, match="Simulated user interrupt"):
        run_supervised(
            [sys.executable, "-c", script],
            gpu_uuid=target_uuid,
            cpu_threads=2,
            claims_dir=claims_dir,
            available_cores=8,
            probe_fn=lambda: [mock_gpu(target_uuid)],
        )

    # Verify child process was reaped and is dead
    assert len(child_pid_holder) == 1
    reaped_pid = child_pid_holder[0]
    time.sleep(0.2)
    try:
        os.kill(reaped_pid, 0)
        is_alive = True
    except ProcessLookupError:
        is_alive = False
    assert not is_alive, f"Child PID {reaped_pid} survived interruption!"

    # Verify claims were cleanly released
    status = get_claims_status(claims_dir)
    assert len(status["devices"]) == 0

    # Case B: Unresolved retention when child death cannot be verified
    from experiments.jobs import claims as claims_mod

    orig_reap = claims_mod.terminate_and_reap_process_group

    def mock_reap_failure(proc: Any, *args: Any, **kwargs: Any) -> bool:
        return False  # Simulated failure to verify child death

    monkeypatch.setattr(claims_mod, "terminate_and_reap_process_group", mock_reap_failure)

    def mock_wait_raise(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Simulated interrupt failure")

    monkeypatch.setattr(subprocess.Popen, "wait", mock_wait_raise)

    with pytest.raises(RuntimeError, match="Simulated interrupt failure"):
        run_supervised(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            gpu_uuid=target_uuid,
            cpu_threads=2,
            claims_dir=claims_dir,
            available_cores=8,
            probe_fn=lambda: [mock_gpu(target_uuid)],
        )

    # When child death cannot be verified, claims must be RETAINED (not released)
    retained_status = get_claims_status(claims_dir)
    assert len(retained_status["devices"]) == 1, "Claims must be retained when child death cannot be verified!"
    # Clean up retained claim manually for following tests
    retained_token = retained_status["devices"][0]["token"]
    release_device_claim(get_canonical_host(), target_uuid, retained_token, claims_dir)

    # Restore patched functions before Case C
    monkeypatch.setattr(claims_mod, "terminate_and_reap_process_group", orig_reap)
    monkeypatch.setattr(subprocess.Popen, "wait", orig_wait)

    # Case C: Direct terminate_and_reap_process_group verification
    direct_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True)
    assert terminate_and_reap_process_group(direct_proc, sigterm_timeout=1.0) is True
    assert direct_proc.poll() is not None


def test_is_claim_active_priority_over_status_file(tmp_path: Path) -> None:
    """Blocker 3: is_claim_active must check process liveness before trusting status text.

    A live local PID with interrupted or failed status text remains active.
    Remote claims with terminal status return None (conservatively unverifiable).
    """
    job_dir = tmp_path / "job-status-test"
    job_dir.mkdir(parents=True)

    # Start a live background process
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        # Write terminal status text: "interrupted"
        status_file = job_dir / "status.json"
        write_json_atomic(
            status_file,
            {
                "status": "interrupted",
                "pid": proc.pid,
                "started": time.time(),
            },
        )

        claim_data = {
            "token": "live-proc-token",
            "host": get_canonical_host(),
            "pid": proc.pid,
            "job_dir": str(job_dir),
            "created_at": time.time(),
        }

        # Even though status is "interrupted", the local PID is STILL ALIVE.
        # Claim must be reported as ACTIVE!
        assert is_claim_active(claim_data) is True

        # Write status "failed" while process still alive
        write_json_atomic(
            status_file,
            {
                "status": "failed",
                "pid": proc.pid,
                "started": time.time(),
            },
        )
        assert is_claim_active(claim_data) is True

    finally:
        # Terminate process and verify it is dead
        proc.terminate()
        proc.wait(timeout=5)

    # Now that the process is dead, is_claim_active returns False
    assert is_claim_active(claim_data) is False

    # Remote host claim: cannot verify remotely, returns None
    remote_claim = dict(claim_data)
    remote_claim["host"] = "other-host.domain"
    assert is_claim_active(remote_claim) is None


def test_occupancy_query_fail_closed_and_pre_popen_recheck(tmp_path: Path) -> None:
    """Blocker 4: Missing UUID or failed query must fail closed / defer (raise DeviceUnavailableError).

    Pre-Popen recheck under claim must abort if device becomes busy before launch.
    """
    claims_dir = tmp_path / "claims"

    # 1. Missing UUID with empty query probe: MUST fail closed
    with pytest.raises(DeviceUnavailableError, match="not found or probe failed"):
        acquire_device_claim(
            device_uuid="GPU-NOT-PRESENT",
            claims_dir=claims_dir,
            probe_fn=lambda: [],
        )

    # 2. Missing UUID with non-matching probe: MUST fail closed
    with pytest.raises(DeviceUnavailableError, match="not found or probe failed"):
        acquire_device_claim(
            device_uuid="GPU-NOT-PRESENT",
            claims_dir=claims_dir,
            probe_fn=lambda: [mock_gpu("GPU-OTHER-123")],
        )

    # 3. auto_select with empty probe: MUST fail closed
    with pytest.raises(DeviceUnavailableError, match="No GPU devices found or query failed"):
        acquire_device_claim(
            auto_select=True,
            claims_dir=claims_dir,
            probe_fn=lambda: [],
        )

    # 4. Pre-Popen recheck: device is free during acquisition, but busy immediately before Popen
    target_uuid = "GPU-busy-pre-popen"
    call_count = [0]

    def mock_recheck_pre_popen(uuid: str, min_mem: float) -> bool:
        call_count[0] += 1
        # 1st call is during acquire_device_claim (returns True so acquire succeeds)
        # 2nd call is immediately before Popen in launch (returns False)
        return call_count[0] == 1

    with pytest.raises(DeviceBusyError, match="became busy during pre-Popen recheck"):
        launch(
            [sys.executable, "-c", "import time; time.sleep(0.1)"],
            gpu_uuid=target_uuid,
            cpu_threads=2,
            claims_dir=claims_dir,
            available_cores=8,
            probe_fn=lambda: [mock_gpu(target_uuid)],
            recheck_fn=mock_recheck_pre_popen,
        )

    assert call_count[0] >= 1
    # Both GPU and CPU claims must be cleaned up
    status = get_claims_status(claims_dir)
    assert len(status["devices"]) == 0
    for host_info in status["cpu_hosts"].values():
        assert len(host_info.get("allocations", {})) == 0


def test_cpu_cgroup_quota_load_and_mandatory_allowance(tmp_path: Path) -> None:
    """Blocker 5: Cgroup quota and system load are accounted for; explicit CPU allowance is mandatory."""
    # 1. Test cgroup v2 parsing
    cg2_dir = tmp_path / "cg2"
    cg2_dir.mkdir(parents=True)
    (cg2_dir / "cpu.max").write_text("200000 100000\n", encoding="utf-8")
    assert get_cgroup_cpu_limit(cg2_dir) == 2.0

    # 2. Test cgroup v1 parsing
    cg1_dir = tmp_path / "cg1"
    cg1_dir.mkdir(parents=True)
    (cg1_dir / "cpu.cfs_quota_us").write_text("400000\n", encoding="utf-8")
    (cg1_dir / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")
    assert get_cgroup_cpu_limit(cg1_dir) == 4.0

    # 3. Test get_available_cores with cgroup limit
    cores = get_available_cores(respect_cgroup=True, respect_load=False, cgroup_base=cg2_dir)
    assert cores <= 2

    # 4. Mandatory explicit per-job CPU allowance
    claims_dir = tmp_path / "claims"
    target_uuid = "GPU-mandatory-cpu"

    # launch without cpu_threads must be rejected
    with pytest.raises(ValueError, match="explicit positive cpu_threads allowance"):
        launch(
            [sys.executable, "-c", "pass"],
            gpu_uuid=target_uuid,
            cpu_threads=None,
            claims_dir=claims_dir,
            probe_fn=lambda: [mock_gpu(target_uuid)],
        )

    # run_supervised with cpu_threads=0 must be rejected
    with pytest.raises(ValueError, match="explicit positive cpu_threads allowance"):
        run_supervised(
            [sys.executable, "-c", "pass"],
            gpu_uuid=target_uuid,
            cpu_threads=0,
            claims_dir=claims_dir,
            probe_fn=lambda: [mock_gpu(target_uuid)],
        )

    # claim_resources with cpu_threads=None must be rejected
    with pytest.raises(ValueError, match="explicit positive cpu_threads allowance"):
        with claim_resources(
            gpu_uuid=target_uuid,
            cpu_threads=None,
            claims_dir=claims_dir,
            probe_fn=lambda: [mock_gpu(target_uuid)],
        ):
            pass

    # 5. Environment variables verified
    with claim_resources(
        gpu_uuid=target_uuid,
        cpu_threads=4,
        claims_dir=claims_dir,
        available_cores=16,
        probe_fn=lambda: [mock_gpu(target_uuid)],
    ) as session:
        env = build_child_env(session)
        assert env["OMP_NUM_THREADS"] == "4"
        assert env["MKL_NUM_THREADS"] == "4"
        assert env["TORCH_NUM_THREADS"] == "4"
        assert env["RAY_NUM_CPUS"] == "4"
        assert env["POLARS_MAX_THREADS"] == "4"
        assert env["PS_CPU_ALLOWANCE"] == "4"
        assert env["PS_NUM_WORKERS"] == "4"

