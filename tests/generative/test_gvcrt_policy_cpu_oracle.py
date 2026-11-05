from pathlib import Path

import pytest

from experiments.generative import gvcrt_policy_cpu_oracle as oracle


def protected_config(tmp_path: Path) -> dict:
    groups = {'source_archive': 2, 'source_rgb': 18, 'source_yuv': 1, 'sender_reference': 1, 'ground_truth': 4}
    protected: dict[str, list[str]] = {}
    aliases = []
    for group, count in groups.items():
        protected[group] = []
        for i in range(count):
            p = tmp_path / f'{group}-{i}'
            p.write_bytes(b'never read during denial probes')
            protected[group].append(str(p))
            alias = tmp_path / f'alias-{group}-{i}'
            alias.symlink_to(p)
            aliases.append(str(alias))
    protected['aliases'] = aliases
    return {'denied_paths': protected}


def test_real_path_alias_coverage_is_exact(tmp_path):
    cfg = protected_config(tmp_path)
    oracle.validate_config(cfg)
    Path(cfg['denied_paths']['aliases'][0]).unlink()
    Path(cfg['denied_paths']['aliases'][0]).symlink_to(cfg['denied_paths']['source_rgb'][0])
    with pytest.raises(ValueError, match='aliases'):
        oracle.validate_config(cfg)


def test_host_loader_identity_requires_host_root_owned_cache(monkeypatch, tmp_path):
    class FakePath:
        def resolve(self, strict=False):
            return self
        def stat(self):
            return type('Stat', (), {'st_mode': 0o100644, 'st_uid': 65534, 'st_dev': 1, 'st_ino': 2})()
        def __str__(self):
            return '/etc/ld.so.cache'
    monkeypatch.setattr(oracle, 'Path', lambda _: FakePath())
    with pytest.raises(ValueError, match='host-parent'):
        oracle.loader_identity()
