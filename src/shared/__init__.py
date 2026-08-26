"""Condemned pre-rewrite leftovers (BP22). Not a layer.

Still here because Stream B is live on training, ``src.transport.disk`` still
imports the old schemas, and pre-rewrite scripts still call eval/dataset helpers.
``src.decoder`` is gone. ``dwpose_draw`` and run-summary invariants have moved out.
"""
