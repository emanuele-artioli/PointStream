"""Drive a shipped tier config through the runner and score the result.

This package is the caller `src/runner` was built to serve: it loads a config
file, hands the runner source chunks, and writes what came back. It imports the
runner as a library — no subprocess, no stdout scrape.
"""
