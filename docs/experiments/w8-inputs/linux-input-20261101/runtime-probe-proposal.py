"""Proposed remote stdin probe, NOT RUN: inventory or qualify one cache.

Each invocation is read-only. Its caller must enforce the shared CPU/wall
budget across all hosts, SSH processes and local orchestration.
"""
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
from pathlib import Path

MODE = sys.argv[1]
if MODE not in ('inventory', 'qualify', 'namespace'):
    raise SystemExit('expected inventory, qualify or namespace')
resource.setrlimit(resource.RLIMIT_CPU, (3 if MODE == 'qualify' else 1,) * 2)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
root = Path('/local/users/emanuele/pointstream')
cache = root / 'cache/32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'
tree = cache / 'tree'
if MODE == 'namespace':
    unshare = '/usr/bin/unshare'
    if not os.path.isfile(unshare) or not os.access(unshare, os.X_OK):
        raise SystemExit('unshare unavailable')
    parent_net = os.readlink('/proc/self/ns/net')
    code = ("import json,os,socket,sys; "
            "parent=sys.argv[1]; current=os.readlink('/proc/self/ns/net'); "
            "s=socket.socket(socket.AF_INET,socket.SOCK_STREAM); s.close(); "
            "print(json.dumps({'parent_net':parent,'child_net':current,'different':parent != current,'socket_created':True})); "
            "sys.exit(0 if parent != current else 1)")
    os.execv(unshare, [unshare, '--user', '--map-root-user', '--net', '/usr/bin/python3', '-B', '-c', code, parent_net])
elif MODE == 'inventory':
    result = {'mode': MODE, 'host': os.uname().nodename, 'root_exists': root.is_dir(),
              'cache_tree_exists': tree.is_dir(), 'python_exists': (tree / 'bin/python').is_file()}
    if root.is_dir():
        fs = os.statvfs(root)
        result.update({'local_device': os.stat(root).st_dev, 'root_device': os.stat('/').st_dev,
                       'separate_filesystem': os.stat(root).st_dev != os.stat('/').st_dev,
                       'free_bytes': fs.f_bavail * fs.f_frsize})
    print(json.dumps(result))
else:
    # Called with the cached environment's own interpreter, not system Python.
    if Path(sys.executable).resolve() != (tree / 'bin/python').resolve():
        raise SystemExit('wrong interpreter')
    import numpy
    import cv2
    if not all(Path(module.__file__).resolve().is_relative_to(tree.resolve()) for module in (numpy, cv2)):
        raise SystemExit('import resolves outside cached environment')
    packages = {}
    for name in ('numpy', 'opencv-python-headless'):
        dist = importlib.metadata.distribution(name)
        if not Path(dist.locate_file('')).resolve().is_relative_to(tree.resolve()):
            raise SystemExit(f'{name} metadata resolves outside cached environment')
        record = dist.read_text('RECORD')
        if record is None:
            raise SystemExit(f'{name} RECORD missing')
        packages[name] = {'version': dist.version,
                          'record_sha256': hashlib.sha256(record.encode()).hexdigest()}
    print(json.dumps({'mode': MODE, 'host': os.uname().nodename, 'executable': sys.executable,
                      'prefix': sys.prefix, 'numpy_import_version': numpy.__version__,
                      'numpy_path': numpy.__file__, 'opencv_import_version': cv2.__version__,
                      'opencv_path': cv2.__file__, 'metadata': packages}))
