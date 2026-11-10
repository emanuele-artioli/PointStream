"""Doctor keeps workload probes while using the local standard-library helper."""

import pytest

from experiments.jobs import fleet, inbox


@pytest.mark.parametrize("mode", ["success", "invisible", "timeout", "no_winner", "two_winners"])
def test_doctor_stdlib_helper_and_fail_closed(monkeypatch, mode):
    hosts = [f"gpu{i}" for i in range(1, 7)]
    probes = [
        {"alias": host, "available": True, "data_root_available": True,
         "python_available": True, "data_root": "/shared/data",
         "python": f"/nfs/{host}/pointstream/bin/python"}
        for host in hosts
    ]
    calls = []
    monkeypatch.setattr(fleet, "inspect_fleet", lambda names: probes if names == tuple(hosts) else [])

    def remote(host, python, script):
        calls.append((host, python, script))
        if "visible=" in script:
            if mode == "timeout" and host == "gpu6":
                raise TimeoutError("atomic check timed out")
            return {"visible": not (mode == "invisible" and host == "gpu6"),
                    "won": host == "gpu1" if mode != "no_winner" else False} | (
                        {"won": True} if mode == "two_winners" and host == "gpu2" else {}
                    )
        return {}

    monkeypatch.setattr(fleet, "_remote_json", remote)
    result = inbox.doctor(hosts)
    assert result["passed"] is (mode == "success")
    assert result["probes"] == probes
    assert [p["python"] for p in result["probes"]] == [p["python"] for p in probes]
    assert len(result["atomic_checks"]) == 6
    assert len(calls) == 8
    assert all(python == "/usr/bin/python3" for _, python, _ in calls)
    assert "report.json" in calls[-1][2]
    assert probes[0]["python"] in calls[-1][2]
    if mode == "timeout":
        assert result["atomic_checks"]["gpu6"]["visible"] is False


def test_doctor_rejects_failed_workload_probe_before_atomic_checks(monkeypatch):
    probes = [{"alias": "gpu1", "available": True, "data_root_available": True,
               "python_available": False, "data_root": "/shared/data", "python": "/nfs/python"}]
    monkeypatch.setattr(fleet, "inspect_fleet", lambda _: probes)
    monkeypatch.setattr(fleet, "_remote_json", lambda *_: pytest.fail("atomic check ran after failed probe"))
    assert inbox.doctor(["gpu1"])["passed"] is False


@pytest.mark.parametrize("failure", ["init", "report"])
def test_doctor_does_not_ignore_helper_failure(monkeypatch, failure):
    probe = {"alias": "gpu1", "available": True, "data_root_available": True,
             "python_available": True, "data_root": "/shared/data", "python": "/nfs/python"}
    monkeypatch.setattr(fleet, "inspect_fleet", lambda _: [probe])

    def remote(_host, _python, script):
        if (failure == "init" and "p.mkdir(parents=True)" in script) or (
            failure == "report" and "report.json" in script
        ):
            raise TimeoutError("helper failed")
        return {"visible": True, "won": True} if "visible=" in script else {}

    monkeypatch.setattr(fleet, "_remote_json", remote)
    with pytest.raises(TimeoutError, match="helper failed"):
        inbox.doctor(["gpu1"])
