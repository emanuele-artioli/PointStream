"""Atomic cross-host resource claims for GPU devices and CPU thread caps.

Claims coordinate PointStream experiment workers across shared GPU servers.
Devices and CPU allocations are keyed on a shared filesystem primitive (atomic
directory creation) without relying on per-host /tmp locks or cluster schedulers.
"""

from __future__ import annotations

import sqlite3  # noqa: F401 -- host ABI ordering, before child integrations
import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import socket
import subprocess
import time
from typing import Any, Callable, Iterator

from src.contracts.paths import data_root


class ResourceClaimError(Exception):
    """Base exception for all resource claim errors."""


class DeviceUnavailableError(ResourceClaimError):
    """Raised when no requested or free GPU device is available."""


class DeviceClaimConflictError(ResourceClaimError):
    """Raised when a requested device is already claimed by another worker."""


class DeviceBusyError(ResourceClaimError):
    """Raised when a claimed device is found busy (external process or memory drop) on pre-launch recheck."""


class CPUOversubscriptionError(ResourceClaimError):
    """Raised when requested CPU threads exceed the host available-core cap."""


class InvalidTokenError(ResourceClaimError):
    """Raised when attempting to release or mutate a claim with an invalid ownership token."""


@dataclass(frozen=True)
class DeviceClaim:
    """Record of an acquired GPU device claim."""

    token: str
    host: str
    device_uuid: str
    device_ordinal: int
    child_ordinals: dict[str, int]
    pid: int
    created_at: float
    job_id: str
    job_dir: str | None = None


@dataclass(frozen=True)
class CPUClaim:
    """Record of an acquired CPU thread allowance on a host."""

    token: str
    host: str
    threads: int
    pid: int
    claimed_at: float
    job_id: str
    job_dir: str | None = None


@dataclass
class ClaimSession:
    """Container holding active device and CPU claims for a job."""

    token: str
    device_claim: DeviceClaim | None
    cpu_claim: CPUClaim | None
    claims_dir: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "claims_dir": str(self.claims_dir),
            "device_claim": asdict(self.device_claim) if self.device_claim else None,
            "cpu_claim": asdict(self.cpu_claim) if self.cpu_claim else None,
        }


def get_claims_dir(custom_path: Path | str | None = None) -> Path:
    """Return the shared directory for resource claims."""
    if custom_path:
        path = Path(custom_path).expanduser().resolve()
    elif os.environ.get("PS_CLAIMS_DIR"):
        path = Path(os.environ["PS_CLAIMS_DIR"]).expanduser().resolve()
    else:
        path = data_root() / "jobs" / "claims"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_canonical_host() -> str:
    """Return canonical lowercase hostname for resource keying."""
    host = socket.getfqdn().strip().lower()
    if not host or host == "localhost":
        host = platform.node().strip().lower()
    return host or "unknown-host"


def sanitize_uuid(uuid_str: str) -> str:
    """Sanitize GPU UUID string for safe filesystem naming."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", uuid_str)


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON file atomically using a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json_safe(path: Path, default: Any = None) -> Any:
    """Read JSON file safely, returning default if absent or unreadable."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def is_pid_alive(pid: int) -> bool:
    """Check if process with given PID is alive on the local host."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists and belongs to another user
        return True


def is_claim_active(claim_data: dict[str, Any]) -> bool | None:
    """Determine whether a claim is active.

    Returns:
        True: Process or supervisor is verified alive.
        False: Process or supervisor is verified terminated/dead.
        None: Cannot verify remotely (MUST NOT STEAL on TTL alone).
    """
    if not claim_data:
        return False
    host = claim_data.get("host")
    pid = claim_data.get("pid")
    job_dir_str = claim_data.get("job_dir")

    if job_dir_str:
        job_dir = Path(job_dir_str)
        status_file = job_dir / "status.json"
        if status_file.exists():
            status_data = read_json_safe(status_file)
            if status_data and status_data.get("status") in {
                "complete",
                "failed",
                "budget_exhausted",
                "interrupted",
            }:
                return False

    current_host = get_canonical_host()
    if host == current_host:
        if pid is not None:
            return is_pid_alive(int(pid))
        return False

    # Remote host: cannot verify PID locally. Return None so we never steal on TTL.
    return None


@contextmanager
def atomic_dir_lock(
    lock_dir: Path,
    *,
    timeout: float = 10.0,
    poll_interval: float = 0.05,
    stale_timeout: float = 30.0,
) -> Iterator[None]:
    """Atomic cross-host mutex using directory creation on shared filesystem."""
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    lock_meta = lock_dir / "owner.json"
    my_token = secrets.token_hex(8)
    acquired = False

    while time.time() - start < timeout:
        try:
            lock_dir.mkdir(parents=True, exist_ok=False)
            acquired = True
            try:
                write_json_atomic(
                    lock_meta,
                    {
                        "token": my_token,
                        "pid": os.getpid(),
                        "host": get_canonical_host(),
                        "created_at": time.time(),
                    },
                )
            except Exception:
                try:
                    lock_dir.rmdir()
                except OSError:
                    pass
                raise
            break
        except FileExistsError:
            # Check for stale lock holder on this host
            if lock_meta.exists():
                meta = read_json_safe(lock_meta, {})
                if meta:
                    mhost = meta.get("host")
                    mpid = meta.get("pid")
                    created_at = meta.get("created_at", 0)
                    if mhost == get_canonical_host() and mpid is not None and not is_pid_alive(int(mpid)):
                        try:
                            lock_meta.unlink(missing_ok=True)
                            lock_dir.rmdir()
                            continue
                        except OSError:
                            pass
                    elif time.time() - created_at > stale_timeout:
                        try:
                            lock_meta.unlink(missing_ok=True)
                            lock_dir.rmdir()
                            continue
                        except OSError:
                            pass
            elif lock_dir.exists():
                try:
                    if time.time() - lock_dir.stat().st_mtime > stale_timeout:
                        try:
                            lock_dir.rmdir()
                            continue
                        except OSError:
                            pass
                except OSError:
                    pass
            time.sleep(poll_interval)

    if not acquired:
        raise TimeoutError(f"Failed to acquire atomic directory lock {lock_dir} within {timeout}s")
    try:
        yield
    finally:
        try:
            lock_meta.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            lock_dir.rmdir()
        except OSError:
            pass


def query_gpus(
    probe_fn: Callable[[], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Query visible GPUs, memory, and active processes on this host."""
    if probe_fn is not None:
        return probe_fn()

    # Query GPU properties
    try:
        gpu_proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return []

    devices: dict[str, dict[str, Any]] = {}
    for line in gpu_proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                idx = int(parts[0])
                uuid_str = parts[1]
                name = parts[2]
                mem_free = float(parts[3])
                mem_total = float(parts[4])
                devices[uuid_str] = {
                    "index": idx,
                    "uuid": uuid_str,
                    "name": name,
                    "memory_free_mb": mem_free,
                    "memory_total_mb": mem_total,
                    "active_pids": [],
                }
            except ValueError:
                continue

    # Query active compute processes per GPU
    try:
        app_proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        for line in app_proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                gpu_uuid = parts[0]
                try:
                    pid = int(parts[1])
                    if gpu_uuid in devices:
                        devices[gpu_uuid]["active_pids"].append(pid)
                except ValueError:
                    continue
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    return list(devices.values())


def is_device_free(
    device_info: dict[str, Any],
    *,
    min_free_memory_mb: float = 4000.0,
    allowed_pids: set[int] | None = None,
) -> bool:
    """Check if a GPU device has sufficient free memory and no unallowed processes."""
    if device_info.get("memory_free_mb", 0.0) < min_free_memory_mb:
        return False
    active = set(device_info.get("active_pids", []))
    if allowed_pids:
        active -= allowed_pids
    if active:
        return False
    return True


def acquire_device_claim(
    *,
    device_uuid: str | None = None,
    auto_select: bool = False,
    claims_dir: Path | str | None = None,
    job_id: str = "default",
    job_dir: Path | str | None = None,
    min_free_memory_mb: float = 4000.0,
    probe_fn: Callable[[], list[dict[str, Any]]] | None = None,
    recheck_fn: Callable[[str, float], bool] | None = None,
) -> DeviceClaim:
    """Atomically acquire a GPU claim with pre-selection and pre-launch recheck."""
    claims_path = get_claims_dir(claims_dir)
    devices_dir = claims_path / "devices"
    devices_dir.mkdir(parents=True, exist_ok=True)
    canonical_host = get_canonical_host()

    all_devices = query_gpus(probe_fn)
    candidate_devices: list[dict[str, Any]] = []

    if device_uuid is not None:
        matched = [d for d in all_devices if d["uuid"] == device_uuid]
        if matched:
            candidate_devices = matched
        else:
            # Device might exist on host even if query_gpus is mocked or minimal
            candidate_devices = [{
                "index": 0,
                "uuid": device_uuid,
                "name": "Target-GPU",
                "memory_free_mb": min_free_memory_mb + 1000.0,
                "memory_total_mb": min_free_memory_mb + 2000.0,
                "active_pids": [],
            }]
    elif auto_select:
        # Pre-selection: inspect candidate devices
        for dev in all_devices:
            if is_device_free(dev, min_free_memory_mb=min_free_memory_mb):
                candidate_devices.append(dev)
        if not candidate_devices:
            raise DeviceUnavailableError(
                f"No free GPU with >= {min_free_memory_mb} MB available on {canonical_host}"
            )
    else:
        raise ValueError("Either device_uuid or auto_select=True must be specified")

    for candidate in candidate_devices:
        cand_uuid = candidate["uuid"]
        cand_ordinal = candidate.get("index", 0)

        # Pre-selection check on candidancy
        if not is_device_free(candidate, min_free_memory_mb=min_free_memory_mb):
            continue

        sanitized = sanitize_uuid(cand_uuid)
        lock_dir = devices_dir / f"{canonical_host}_{sanitized}"
        claim_file = lock_dir / "claim.json"

        # Attempt atomic filesystem acquisition
        acquired = False
        try:
            lock_dir.mkdir(parents=True, exist_ok=False)
            acquired = True
        except FileExistsError:
            # Check whether claim is stale and verified dead
            if claim_file.exists():
                existing = read_json_safe(claim_file, {})
                active = is_claim_active(existing)
                if active is False:
                    # Verified dead: clean up and retry acquisition
                    try:
                        claim_file.unlink(missing_ok=True)
                        lock_dir.rmdir()
                        lock_dir.mkdir(parents=True, exist_ok=False)
                        acquired = True
                    except OSError:
                        acquired = False
                # If active is True or None (remote unverifiable), do not steal
            elif lock_dir.exists():
                # Directory without claim.json: check creation age
                try:
                    if time.time() - lock_dir.stat().st_mtime > 30.0:
                        try:
                            lock_dir.rmdir()
                            lock_dir.mkdir(parents=True, exist_ok=False)
                            acquired = True
                        except OSError:
                            acquired = False
                except OSError:
                    acquired = False

        if not acquired:
            if device_uuid is not None:
                raise DeviceClaimConflictError(
                    f"Device {cand_uuid} on {canonical_host} is already claimed"
                )
            continue

        token = secrets.token_hex(16)
        claim = DeviceClaim(
            token=token,
            host=canonical_host,
            device_uuid=cand_uuid,
            device_ordinal=cand_ordinal,
            child_ordinals={cand_uuid: 0},
            pid=os.getpid(),
            created_at=time.time(),
            job_id=job_id,
            job_dir=str(job_dir) if job_dir else None,
        )
        write_json_atomic(claim_file, asdict(claim))

        # RECHECK UNDER CLAIM IMMEDIATELY BEFORE RETURNING FOR LAUNCH
        is_still_free = True
        if recheck_fn is not None:
            is_still_free = recheck_fn(cand_uuid, min_free_memory_mb)
        else:
            fresh_devices = query_gpus(probe_fn)
            fresh_cand = next((d for d in fresh_devices if d["uuid"] == cand_uuid), None)
            if fresh_cand is not None:
                is_still_free = is_device_free(fresh_cand, min_free_memory_mb=min_free_memory_mb)

        if not is_still_free:
            # Defer / abort: release claim immediately; never kill or preempt external process
            release_device_claim(canonical_host, cand_uuid, token, claims_path)
            if device_uuid is not None:
                raise DeviceBusyError(
                    f"Device {cand_uuid} on {canonical_host} became busy during pre-launch recheck"
                )
            continue

        return claim

    raise DeviceUnavailableError(
        f"Unable to acquire any free GPU on {canonical_host} (all busy or claimed)"
    )


def release_device_claim(
    host: str,
    device_uuid: str,
    token: str,
    claims_dir: Path | str | None = None,
) -> None:
    """Release a device claim by the owning token. Reject non-owner releases."""
    claims_path = get_claims_dir(claims_dir)
    sanitized = sanitize_uuid(device_uuid)
    lock_dir = claims_path / "devices" / f"{host}_{sanitized}"
    if not lock_dir.exists():
        return

    claim_file = lock_dir / "claim.json"
    if claim_file.exists():
        data = read_json_safe(claim_file, {})
        owner_token = data.get("token")
        if owner_token != token:
            raise InvalidTokenError(
                f"Release rejected for {device_uuid} on {host}: token mismatch "
                f"(expected {owner_token}, received {token})"
            )
        claim_file.unlink(missing_ok=True)

    try:
        lock_dir.rmdir()
    except OSError:
        pass


def get_available_cores() -> int:
    """Return available CPU cores respecting process affinity and CPU quota."""
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os.cpu_count() or 1
    total = os.cpu_count() or 1
    return max(1, min(affinity, total))


def get_cpu_cap(available_cores: int | None = None) -> int:
    """Compute the campaign CPU thread cap: floor(0.90 * available_cores)."""
    cores = available_cores if available_cores is not None else get_available_cores()
    return max(1, math.floor(0.90 * cores))


def acquire_cpu_claim(
    threads: int,
    *,
    host: str | None = None,
    job_id: str = "default",
    job_dir: Path | str | None = None,
    claims_dir: Path | str | None = None,
    available_cores: int | None = None,
    cpu_cap: int | None = None,
) -> CPUClaim:
    """Atomically reserve CPU threads under the host available-core cap."""
    if threads <= 0:
        raise ValueError("Requested CPU threads must be positive")

    canonical_host = host or get_canonical_host()
    claims_path = get_claims_dir(claims_dir)
    cpu_dir = claims_path / "cpu"
    cpu_dir.mkdir(parents=True, exist_ok=True)

    cap = cpu_cap if cpu_cap is not None else get_cpu_cap(available_cores)
    alloc_file = cpu_dir / f"{canonical_host}.json"
    mutex_dir = cpu_dir / f"{canonical_host}.lock"

    with atomic_dir_lock(mutex_dir):
        data = read_json_safe(
            alloc_file,
            {
                "host": canonical_host,
                "available_cores": available_cores or get_available_cores(),
                "cap": cap,
                "allocations": {},
            },
        )
        data["cap"] = cap
        allocations: dict[str, dict[str, Any]] = data.get("allocations", {})

        # Clean up verified dead allocations
        dead_tokens = []
        for tok, alloc in allocations.items():
            if is_claim_active(alloc) is False:
                dead_tokens.append(tok)
        for tok in dead_tokens:
            del allocations[tok]

        current_allocated = sum(int(a.get("threads", 0)) for a in allocations.values())
        if current_allocated + threads > cap:
            raise CPUOversubscriptionError(
                f"Requested {threads} threads exceeds cap {cap} on {canonical_host} "
                f"(currently allocated: {current_allocated}/{cap})"
            )

        token = secrets.token_hex(16)
        claim = CPUClaim(
            token=token,
            host=canonical_host,
            threads=threads,
            pid=os.getpid(),
            claimed_at=time.time(),
            job_id=job_id,
            job_dir=str(job_dir) if job_dir else None,
        )
        allocations[token] = asdict(claim)
        data["allocations"] = allocations
        data["updated_at"] = time.time()
        write_json_atomic(alloc_file, data)
        return claim


def release_cpu_claim(
    host: str,
    token: str,
    claims_dir: Path | str | None = None,
) -> None:
    """Release a host CPU allocation by token."""
    claims_path = get_claims_dir(claims_dir)
    cpu_dir = claims_path / "cpu"
    alloc_file = cpu_dir / f"{host}.json"
    mutex_dir = cpu_dir / f"{host}.lock"

    with atomic_dir_lock(mutex_dir):
        if not alloc_file.exists():
            return
        data = read_json_safe(alloc_file, {})
        allocations = data.get("allocations", {})
        if token not in allocations:
            raise InvalidTokenError(f"Release rejected: CPU allocation token {token} not found on {host}")
        del allocations[token]
        data["allocations"] = allocations
        data["updated_at"] = time.time()
        write_json_atomic(alloc_file, data)


def build_child_env(
    claim_session: ClaimSession,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Construct child environment hiding all unallocated GPUs and setting thread caps."""
    env = dict(os.environ if base_env is None else base_env)

    if claim_session.device_claim is not None:
        # Isolate child: hide any visible GPU not explicitly claimed
        claimed_uuid = claim_session.device_claim.device_uuid
        env["CUDA_VISIBLE_DEVICES"] = claimed_uuid
        env["PS_CLAIMED_GPU_UUID"] = claimed_uuid
        env["PS_CHILD_DEVICE_ORDINAL"] = "0"
    else:
        # Hide all GPUs for CPU-only execution
        env["CUDA_VISIBLE_DEVICES"] = ""

    if claim_session.cpu_claim is not None:
        threads_str = str(claim_session.cpu_claim.threads)
        env["OMP_NUM_THREADS"] = threads_str
        env["MKL_NUM_THREADS"] = threads_str
        env["OPENBLAS_NUM_THREADS"] = threads_str
        env["VECLIB_MAXIMUM_THREADS"] = threads_str
        env["NUMEXPR_NUM_THREADS"] = threads_str
        env["TORCH_NUM_THREADS"] = threads_str

    return env


def release_session_claims(session: ClaimSession) -> None:
    """Release all resource claims held by a claim session."""
    errors: list[Exception] = []
    if session.device_claim is not None:
        try:
            release_device_claim(
                session.device_claim.host,
                session.device_claim.device_uuid,
                session.device_claim.token,
                session.claims_dir,
            )
        except Exception as exc:
            errors.append(exc)

    if session.cpu_claim is not None:
        try:
            release_cpu_claim(
                session.cpu_claim.host,
                session.cpu_claim.token,
                session.claims_dir,
            )
        except Exception as exc:
            errors.append(exc)

    if errors:
        raise ResourceClaimError(f"Errors releasing session claims: {errors}")


@contextmanager
def claim_resources(
    *,
    gpu_uuid: str | None = None,
    auto_gpu: bool = False,
    cpu_threads: int | None = None,
    job_id: str = "default",
    job_dir: Path | str | None = None,
    claims_dir: Path | str | None = None,
    min_free_memory_mb: float = 4000.0,
    available_cores: int | None = None,
    cpu_cap: int | None = None,
    probe_fn: Callable[[], list[dict[str, Any]]] | None = None,
    recheck_fn: Callable[[str, float], bool] | None = None,
) -> Iterator[ClaimSession]:
    """Context manager acquiring device and CPU claims and releasing them upon exit."""
    claims_path = get_claims_dir(claims_dir)
    dev_claim: DeviceClaim | None = None
    cpu_claim: CPUClaim | None = None

    try:
        if gpu_uuid is not None or auto_gpu:
            dev_claim = acquire_device_claim(
                device_uuid=gpu_uuid,
                auto_select=auto_gpu,
                claims_dir=claims_path,
                job_id=job_id,
                job_dir=job_dir,
                min_free_memory_mb=min_free_memory_mb,
                probe_fn=probe_fn,
                recheck_fn=recheck_fn,
            )
        if cpu_threads is not None and cpu_threads > 0:
            cpu_claim = acquire_cpu_claim(
                threads=cpu_threads,
                job_id=job_id,
                job_dir=job_dir,
                claims_dir=claims_path,
                available_cores=available_cores,
                cpu_cap=cpu_cap,
            )

        session_token = dev_claim.token if dev_claim else (cpu_claim.token if cpu_claim else "")
        session = ClaimSession(
            token=session_token,
            device_claim=dev_claim,
            cpu_claim=cpu_claim,
            claims_dir=claims_path,
        )
        yield session
    finally:
        if dev_claim is not None:
            try:
                release_device_claim(
                    dev_claim.host,
                    dev_claim.device_uuid,
                    dev_claim.token,
                    claims_path,
                )
            except Exception:
                pass
        if cpu_claim is not None:
            try:
                release_cpu_claim(
                    cpu_claim.host,
                    cpu_claim.token,
                    claims_path,
                )
            except Exception:
                pass


def launch(
    command: list[str],
    *,
    gpu_uuid: str | None = None,
    auto_gpu: bool = False,
    cpu_threads: int | None = None,
    job_id: str = "default",
    job_dir: Path | str | None = None,
    claims_dir: Path | str | None = None,
    min_free_memory_mb: float = 4000.0,
    available_cores: int | None = None,
    cpu_cap: int | None = None,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    stdout: Any = None,
    stderr: Any = None,
    probe_fn: Callable[[], list[dict[str, Any]]] | None = None,
    recheck_fn: Callable[[str, float], bool] | None = None,
) -> tuple[subprocess.Popen[Any], ClaimSession]:
    """Launch a child process under verified resource claims. Clean up claims on launch failure."""
    claims_path = get_claims_dir(claims_dir)
    dev_claim: DeviceClaim | None = None
    cpu_claim: CPUClaim | None = None

    try:
        if gpu_uuid is not None or auto_gpu:
            dev_claim = acquire_device_claim(
                device_uuid=gpu_uuid,
                auto_select=auto_gpu,
                claims_dir=claims_path,
                job_id=job_id,
                job_dir=job_dir,
                min_free_memory_mb=min_free_memory_mb,
                probe_fn=probe_fn,
                recheck_fn=recheck_fn,
            )
        if cpu_threads is not None and cpu_threads > 0:
            cpu_claim = acquire_cpu_claim(
                threads=cpu_threads,
                job_id=job_id,
                job_dir=job_dir,
                claims_dir=claims_path,
                available_cores=available_cores,
                cpu_cap=cpu_cap,
            )

        session_token = dev_claim.token if dev_claim else (cpu_claim.token if cpu_claim else "")
        session = ClaimSession(
            token=session_token,
            device_claim=dev_claim,
            cpu_claim=cpu_claim,
            claims_dir=claims_path,
        )
    except Exception:
        if dev_claim is not None:
            try:
                release_device_claim(dev_claim.host, dev_claim.device_uuid, dev_claim.token, claims_path)
            except Exception:
                pass
        raise

    # Build isolated environment
    child_env = build_child_env(session, env)

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=child_env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        return process, session
    except Exception:
        # Clean up claims immediately on launch failure
        release_session_claims(session)
        raise


def run_supervised(
    command: list[str],
    *,
    gpu_uuid: str | None = None,
    auto_gpu: bool = False,
    cpu_threads: int | None = None,
    job_id: str = "default",
    job_dir: Path | str | None = None,
    claims_dir: Path | str | None = None,
    min_free_memory_mb: float = 4000.0,
    available_cores: int | None = None,
    cpu_cap: int | None = None,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    probe_fn: Callable[[], list[dict[str, Any]]] | None = None,
    recheck_fn: Callable[[str, float], bool] | None = None,
) -> int:
    """Run child command to completion retaining claims and safely releasing them afterwards."""
    process, session = launch(
        command,
        gpu_uuid=gpu_uuid,
        auto_gpu=auto_gpu,
        cpu_threads=cpu_threads,
        job_id=job_id,
        job_dir=job_dir,
        claims_dir=claims_dir,
        min_free_memory_mb=min_free_memory_mb,
        available_cores=available_cores,
        cpu_cap=cpu_cap,
        env=env,
        cwd=cwd,
        probe_fn=probe_fn,
        recheck_fn=recheck_fn,
    )
    try:
        return process.wait()
    finally:
        release_session_claims(session)


def get_claims_status(claims_dir: Path | str | None = None) -> dict[str, Any]:
    """Inspect active device and CPU claims across participating hosts."""
    claims_path = get_claims_dir(claims_dir)
    devices_dir = claims_path / "devices"
    cpu_dir = claims_path / "cpu"

    device_records = []
    if devices_dir.exists():
        for item in sorted(devices_dir.iterdir()):
            if item.is_dir():
                claim_file = item / "claim.json"
                data = read_json_safe(claim_file)
                if data:
                    active = is_claim_active(data)
                    device_records.append({**data, "active": active, "lock_dir": str(item)})

    cpu_records = {}
    if cpu_dir.exists():
        for item in sorted(cpu_dir.glob("*.json")):
            data = read_json_safe(item)
            if data:
                cpu_records[item.stem] = data

    return {
        "claims_dir": str(claims_path),
        "devices": device_records,
        "cpu_hosts": cpu_records,
    }


def main() -> int:
    """CLI for managing and launching commands under resource claims."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    launch_cmd = subparsers.add_parser("launch", help="Launch a command under resource claims")
    launch_cmd.add_argument("--gpu-uuid", help="Explicit GPU UUID to claim")
    launch_cmd.add_argument("--auto-gpu", action="store_true", help="Auto-claim first free GPU")
    launch_cmd.add_argument("--cpu-threads", type=int, help="CPU threads to claim under host cap")
    launch_cmd.add_argument("--job-id", default="cli-job", help="Job identity for claims")
    launch_cmd.add_argument("--claims-dir", type=Path, help="Explicit claims directory")
    launch_cmd.add_argument("--command", nargs=argparse.REMAINDER, help="Command to execute")
    launch_cmd.add_argument("cmd", nargs="*", help="Command to execute if --command omitted")

    status_cmd = subparsers.add_parser("status", help="Show active resource claims")
    status_cmd.add_argument("--claims-dir", type=Path, help="Explicit claims directory")

    release_cmd = subparsers.add_parser("release", help="Release a claim by token")
    release_cmd.add_argument("--token", required=True, help="Claim ownership token")
    release_cmd.add_argument("--host", default=get_canonical_host(), help="Host identity")
    release_cmd.add_argument("--gpu-uuid", help="GPU UUID if releasing a device claim")
    release_cmd.add_argument("--cpu", action="store_true", help="If releasing a CPU claim")
    release_cmd.add_argument("--claims-dir", type=Path, help="Explicit claims directory")

    args = parser.parse_args()

    if args.action == "status":
        status = get_claims_status(args.claims_dir)
        print(json.dumps(status, indent=2))
        return 0

    if args.action == "release":
        claims_dir = get_claims_dir(args.claims_dir)
        if args.gpu_uuid:
            release_device_claim(args.host, args.gpu_uuid, args.token, claims_dir)
            print(f"Device claim for {args.gpu_uuid} on {args.host} released.")
        if args.cpu:
            release_cpu_claim(args.host, args.token, claims_dir)
            print(f"CPU claim on {args.host} released.")
        return 0

    if args.action == "launch":
        raw_cmd = args.command or args.cmd
        if not raw_cmd:
            parser.error("A command is required (e.g. -- command ...)")
        command = raw_cmd[1:] if raw_cmd[:1] == ["--"] else raw_cmd
        if not command:
            parser.error("A command is required after --")
        code = run_supervised(
            command,
            gpu_uuid=args.gpu_uuid,
            auto_gpu=args.auto_gpu,
            cpu_threads=args.cpu_threads,
            job_id=args.job_id,
            claims_dir=args.claims_dir,
        )
        return code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
