"""Read-only queue admission oracle with a synthetic monotonic clock."""
import pytest

from experiments.jobs import inbox


class Clock:
    now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.mark.parametrize(('states', 'events', 'limit', 'reason', 'elapsed'), [
    ([['pending'], ['pending'], ['running']], False, 120, 'all_running', 40),
    ([['pending']], False, 35, 'timeout', 35),
    ([['failed']], False, 120, 'all_terminal', 0),
    ([['unknown']], True, 120, 'attention', 0),
    ([['running']], True, 120, 'attention', 0),
    ([['running', 'pending'], ['running', 'running']], False, 120, 'all_running', 20),
    ([['complete', 'pending']], False, 120, 'terminal', 0),
])
def test_admission_transition_oracle(tmp_path, monkeypatch, states, events, limit, reason, elapsed):
    clock = Clock()
    reads = []

    def snapshot(state_dir, job_id, now):
        reads.append(job_id)
        poll = min(int(clock.now // 20), len(states) - 1)
        return {'job_id': job_id, 'status': states[poll][int(job_id)],
                'attention': events, 'unacknowledged_event_ids': ['event'] if events else []}

    monkeypatch.setattr(inbox, 'wait_snapshot', snapshot)
    result = inbox.wait_jobs(tmp_path, [str(i) for i in range(len(states[0]))], limit,
                             until_running=True, poll_seconds=20, clock=clock, sleep=clock.sleep)
    assert result['reason'] == reason
    assert result['attention'] is events
    assert clock.now == elapsed
    assert result['jobs'][0]['unacknowledged_event_ids'] == (['event'] if events else [])
    assert reads
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('status,events_error', [('unreachable', None), ('running', 'events unavailable'), ('running', '')])
def test_admission_does_not_accept_incomplete_metadata(tmp_path, monkeypatch, status, events_error):
    clock = Clock()
    monkeypatch.setattr(inbox, 'wait_snapshot', lambda *args: {
        'status': status, 'attention': False, 'events_error': events_error})
    result = inbox.wait_jobs(tmp_path, ['job'], 40, until_running=True,
                             poll_seconds=20, clock=clock, sleep=clock.sleep)
    assert result['reason'] == 'timeout'
    assert clock.now == 40


def test_admission_cli_routes_condition(tmp_path, monkeypatch, capsys):
    seen = {}

    def wait(state_dir, jobs, seconds, **kwargs):
        seen.update(state_dir=state_dir, jobs=jobs, seconds=seconds, **kwargs)
        return {'reason': 'all_running', 'attention': False, 'jobs': []}

    monkeypatch.setattr(inbox, 'wait_jobs', wait)
    assert inbox.main(['--state-dir', str(tmp_path), 'wait', 'job', '--until-running', '--max-seconds', '600']) == 0
    assert seen == {'state_dir': tmp_path, 'jobs': ['job'], 'seconds': 600, 'until_running': True}
    assert 'all_running' in capsys.readouterr().out


def test_admission_retains_45_minute_cap(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(inbox, 'wait_snapshot', lambda *args: {'status': 'pending', 'attention': False})
    result = inbox.wait_jobs(tmp_path, ['job'], 10000, until_running=True,
                             poll_seconds=600, clock=clock, sleep=clock.sleep)
    assert result['reason'] == 'timeout'
    assert clock.now == result['max_seconds'] == 2700
