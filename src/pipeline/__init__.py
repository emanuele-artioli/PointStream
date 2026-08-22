"""The encoder, decoder, reconstruction and stage DAG.

Depends on ``contracts`` only. Backends are injected; this package never
imports ``src.components`` and never looks up a registry. The runner (C3)
is what binds a named backend to a stage callable.
"""
