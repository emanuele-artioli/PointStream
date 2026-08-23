"""The pipeline layer: reconstruction, residual, encoder, and the stage DAG.

Depends on ``contracts`` only. Components reach it by injection — this package
never imports a backend, never inspects a class name, and never asks which
generator was chosen. The runner (C3) is the layer that binds registries
and named backends to stage callables.
"""
