#!/usr/bin/env python3
"""
PointStream Main Entry Point

This is the main entry point for the PointStream pipeline.
It imports the server from the scripts package and runs the main function.
"""

import sys
from pathlib import Path

# Add the current directory to Python path so we can import from scripts
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the main server function
from scripts.server import main

if __name__ == "__main__":
    main()
