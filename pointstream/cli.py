"""
Command Line Interface for PointStream
"""
import argparse
import sys
from pathlib import Path

def run_server():
    """Entry point for pointstream-server command"""
    # Import here to avoid circular imports
    from pointstream.scripts.run_server import main
    main()

def run_client():
    """Entry point for pointstream-client command"""
    from pointstream.scripts.run_client import main
    main()

def train():
    """Entry point for pointstream-train command"""
    from pointstream.scripts.train import main
    main()

def simple_train():
    """Entry point for pointstream-simple-train command"""
    from pointstream.scripts.simple_train import main
    main()

if __name__ == "__main__":
    print("Use the installed console scripts: pointstream-server, pointstream-client, pointstream-train, or pointstream-simple-train")
    sys.exit(1)
