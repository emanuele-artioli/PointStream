"""Claude Code hooks: ledger staleness, after-compaction context, large-read guard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.agent_hooks import guard_large_read as guard
from tools.agent_hooks import guard_protected as protected
from tools.agent_hooks import ledger, session_start, subagent_guard

ROOT = Path(__file__).resolve().parents[1]

PLAN = """# Plan

### Dispatch ledger

| item | card | branch | suggested model | status |
|---|---|---|---|---|
| H5p 4–5 | F stream | `claude/h5p-f-stream` | Opus | queued 2026-10-22 |
| K6b hands 2c | bank | `claude/k6b-hand-bank` | Opus | done 2026-10-23 |
| J | rejudge | `claude/j-experience-rejudge` | Sonnet | queued 2026-10-22 |

## Done

| a | b | `claude/not-a-ledger` | c | queued |
"""

SUBJECTS = [
    "Merge pull request #221 from emanuele-artioli/claude/k6b-hand-bank",
    "Merge remote-tracking branch 'origin/main' into claude/h5p-f-stream",
    "Merge pull request #220 from emanuele-artioli/claude/h5p-f-stream",
    "Merge pull request #219 from emanuele-artioli/claude/h5p-f-stream-old",
]


def test_ledger_rows_only_from_the_ledger_table() -> None:
    rows = ledger.ledger_rows(PLAN)
    assert [r.branch for r in rows] == ["claude/h5p-f-stream", "claude/k6b-hand-bank", "claude/j-experience-rejudge"]
    assert rows[0].status == "queued 2026-10-22"


def test_merged_prs_matches_whole_branch_names() -> None:
    merged = ledger.merged_prs(SUBJECTS)
    assert merged["claude/h5p-f-stream"] == [220]
    assert merged["claude/h5p-f-stream-old"] == [219]
    assert "claude/j-experience-rejudge" not in merged


def test_ledger_rows_read_the_branch_from_the_current_column_format() -> None:
    plan = """### Dispatch ledger

| item | card | branch | model / effort | status |
|---|---|---|---|---|
| J | rejudge | `claude/j-experience-rejudge` @ `6a1b09a`, PR #218 | Sonnet | reported 2026-10-22: fail |
| K6m | matte | — | Sonnet | card dismissed 2026-10-24 |
| T1 | split | `codex/t1-tennis-split` | Codex high | card ready 2026-10-24 ([card](docs/cards/t1.md)) |
"""
    rows = ledger.ledger_rows(plan)
    assert [(r.item, r.branch) for r in rows] == [
        ("J", "claude/j-experience-rejudge"),
        ("T1", "codex/t1-tennis-split"),
    ]


def test_the_ledger_on_main_parses() -> None:
    rows = ledger.ledger_rows((ROOT / "PLAN.md").read_text(encoding="utf-8"))
    assert rows and all("/" in r.branch and " " not in r.branch for r in rows)


def test_card_ready_rows_with_merged_prs_are_stale_running_ones_are_not() -> None:
    rows = [
        ledger.Row("A", "claude/h5p-f-stream", "card ready 2026-10-24"),
        ledger.Row("B", "claude/h5p-f-stream", "running 2026-10-24"),
        ledger.Row("C", "claude/h5p-f-stream", "reported 2026-10-24: pass"),
    ]
    stale = ledger.stale_rows(rows, ledger.merged_prs(SUBJECTS))
    assert len(stale) == 1 and stale[0].startswith("- A ")


def test_stale_rows_flags_queued_rows_with_merged_prs_only() -> None:
    stale = ledger.stale_rows(ledger.ledger_rows(PLAN), ledger.merged_prs(SUBJECTS))
    assert len(stale) == 1
    assert "claude/h5p-f-stream" in stale[0] and "#220" in stale[0]


def _repo(tmp_path: Path) -> Path:
    def run(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (tmp_path / "PLAN.md").write_text(PLAN, encoding="utf-8")
    run("add", "PLAN.md")
    run("commit", "-qm", "plan")
    return tmp_path


def test_session_start_is_silent_on_startup_without_ledger_problems(tmp_path: Path) -> None:
    root = _repo(tmp_path)  # no origin/main: no merges known, nothing stale
    assert session_start.build("startup", root, fetch=False) == ""


def test_session_start_after_compaction_reports_git_state(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "new.txt").write_text("x", encoding="utf-8")
    text = session_start.build("compact", root, fetch=False)
    assert "Branch: main" in text
    assert "Uncommitted files (1)" in text and "new.txt" in text
    assert "plan" in text  # last commit
    assert "ps-fleet status JOB_ID" in text
    assert len(text.splitlines()) <= 30


def test_session_start_runs_as_a_hook() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "tools.agent_hooks.session_start"],
        cwd=ROOT, input=json.dumps({"source": "clear"}), capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0


def _long(tmp_path: Path, name: str, lines: int) -> Path:
    path = tmp_path / name
    path.write_text("".join(f"line {i} with some padding text\n" for i in range(lines)), encoding="utf-8")
    return path


def test_read_of_long_file_without_limit_is_blocked(tmp_path: Path) -> None:
    path = _long(tmp_path, "big.md", 1500)
    message = guard.check({"tool_name": "Read", "tool_input": {"file_path": str(path)}})
    assert message is not None and "1500 lines" in message and "offset" in message
    assert guard.check({"tool_name": "Read", "tool_input": {"file_path": str(path), "limit": 200}}) is None


def test_short_files_pass(tmp_path: Path) -> None:
    path = _long(tmp_path, "small.md", 900)
    assert guard.check({"tool_name": "Read", "tool_input": {"file_path": str(path)}}) is None
    assert guard.check({"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "missing.md")}}) is None


def test_logs_have_a_lower_limit_and_tail_advice(tmp_path: Path) -> None:
    path = _long(tmp_path, "job.log", 400)
    message = guard.check({"tool_name": "Bash", "tool_input": {"command": f"cat {path}"}, "cwd": str(tmp_path)})
    assert message is not None and "tail -n 40" in message


def test_bash_printers_only_when_output_reaches_the_session(tmp_path: Path) -> None:
    _long(tmp_path, "job.log", 400)

    def blocked(command: str) -> bool:
        return guard.check({"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(tmp_path)}) is not None

    assert blocked("cat job.log")
    assert blocked("cd x && cat -n job.log")
    assert not blocked("cat job.log | grep Error")
    assert not blocked("cat job.log > copy.log")
    assert not blocked("tail -n 40 job.log")
    assert not blocked("cat <<EOF > notes.md\nhi\nEOF")


def test_guard_runs_as_a_hook_and_exits_2_when_blocking(tmp_path: Path) -> None:
    path = _long(tmp_path, "big.md", 1500)
    payload = json.dumps({"tool_name": "Read", "tool_input": {"file_path": str(path)}})
    done = subprocess.run(
        [sys.executable, "-m", "tools.agent_hooks.guard_large_read"],
        cwd=ROOT, input=payload, capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 2 and "1500 lines" in done.stderr



def test_settings_wire_both_hooks() -> None:
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    commands = json.dumps(settings["hooks"])
    assert "tools.agent_hooks.session_start" in commands
    assert "tools.agent_hooks.guard_large_read" in commands


# Codex: same hook scripts, Codex's payloads (tool_name "Bash" / "apply_patch").


def test_codex_bash_payload_hits_the_large_read_guard(tmp_path: Path) -> None:
    _long(tmp_path, "job.log", 400)
    payload = {"tool_name": "Bash", "tool_input": {"command": "cat job.log"}, "cwd": str(tmp_path), "turn_id": "t"}
    message = guard.check(payload)
    assert message is not None and "grep -nE" in message


def test_force_push_and_hard_reset_are_blocked() -> None:
    def blocked(command: str) -> bool:
        return protected.check({"tool_name": "Bash", "tool_input": {"command": command}}) is not None

    assert blocked("git push --force origin main")
    assert blocked("git push -f origin x")
    assert blocked("git -C repo push origin +main")
    assert blocked("cd x && git reset --hard origin/main")
    assert not blocked("git push --force-with-lease -u origin claude/x")
    assert not blocked("git push -u origin claude/x")
    assert not blocked("git reset --soft HEAD~1")
    assert not blocked("git commit -m 'reset'")


def test_patches_into_models_or_datasets_are_blocked(tmp_path: Path) -> None:
    def blocked(patch: str) -> bool:
        payload = {"tool_name": "apply_patch", "tool_input": {"command": patch}, "cwd": str(tmp_path)}
        return protected.check(payload) is not None

    assert blocked("*** Begin Patch\n*** Update File: Datasets/manifests/x.json\n@@\n-a\n+b\n*** End Patch\n")
    assert blocked("*** Begin Patch\n*** Add File: /home/itec/emanuele/Models/YOLO/w.pt\n+x\n*** End Patch\n")
    assert blocked("*** Begin Patch\n*** Update File: src/a.py\n*** Move to: Models/a.py\n*** End Patch\n")
    assert not blocked("*** Begin Patch\n*** Update File: src/segmentation/storage.py\n@@\n-a\n+b\n*** End Patch\n")
    link = tmp_path / "data"
    link.symlink_to("/home/itec/emanuele/Datasets", target_is_directory=True)
    assert blocked("*** Begin Patch\n*** Add File: data/new.txt\n+x\n*** End Patch\n")


def test_codex_hooks_file_runs_the_same_scripts() -> None:
    config = json.loads((ROOT / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    hooks = config["hooks"]
    assert hooks["SessionStart"][0]["matcher"] == "startup|resume|compact"
    commands = json.dumps(hooks)
    for module in ("session_start", "guard_large_read", "guard_protected"):
        assert f"tools.agent_hooks.{module}" in commands
    assert "git rev-parse --show-toplevel" in commands


# Subagent hooks wired from .claude/agents/.

AGENTS = ROOT / ".claude" / "agents"
DOWNHILL = {
    "opus-designer": ["sonnet-builder", "haiku-runner"],
    "sonnet-builder": ["haiku-runner"],
}


def _agent_call(kind: str | None = None, model: str | None = None) -> dict:
    tool_input: dict = {"prompt": "p", "description": "d"}
    if kind:
        tool_input["subagent_type"] = kind
    if model:
        tool_input["model"] = model
    return {"tool_name": "Agent", "tool_input": tool_input}


def test_spawn_guard_allows_only_listed_agents_at_their_own_model() -> None:
    allowed = ["haiku-runner"]
    assert subagent_guard.spawn_block(_agent_call("haiku-runner"), allowed) is None
    assert subagent_guard.spawn_block(_agent_call("opus-designer"), allowed) is not None
    assert subagent_guard.spawn_block(_agent_call(), allowed) is not None
    assert subagent_guard.spawn_block(_agent_call("fork"), allowed) is not None
    assert subagent_guard.spawn_block(_agent_call("haiku-runner", "opus"), allowed) is not None
    assert subagent_guard.spawn_block({"tool_name": "Bash", "tool_input": {}}, allowed) is None


def test_spawn_guard_runs_as_a_hook_and_exits_2_when_blocking() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "tools.agent_hooks.subagent_guard", "spawn", "haiku-runner"],
        cwd=ROOT, input=json.dumps(_agent_call("opus-designer")),
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 2 and "haiku-runner" in done.stderr


def test_failure_hook_tells_the_runner_to_stop_and_report() -> None:
    payload = {"tool_name": "Bash", "tool_input": {"command": "pytest"}, "error": "Exit code 1\nboom"}
    done = subprocess.run(
        [sys.executable, "-m", "tools.agent_hooks.subagent_guard", "failed"],
        cwd=ROOT, input=json.dumps(payload), capture_output=True, text=True, timeout=60,
    )
    output = json.loads(done.stdout)["hookSpecificOutput"]
    assert done.returncode == 0
    assert output["hookEventName"] == "PostToolUseFailure"
    assert "Exit code 1" in output["additionalContext"] and "stop now" in output["additionalContext"]


def test_agent_definitions_delegate_only_downhill() -> None:
    texts = {path.stem: path.read_text(encoding="utf-8") for path in AGENTS.glob("*.md")}
    assert set(texts) == {"opus-designer", "sonnet-builder", "haiku-runner"}
    for name, allowed in DOWNHILL.items():
        front = texts[name].split("---")[1]
        assert "disallowedTools: Agent" not in front
        assert f"subagent_guard spawn {' '.join(allowed)}\n" in front
    runner = texts["haiku-runner"].split("---")[1]
    assert "disallowedTools: Agent" in runner
    assert "PostToolUseFailure" in runner and "subagent_guard failed" in runner
    assert "effort: medium" in runner
