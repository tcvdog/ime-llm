#!/usr/bin/env python3
"""IME prototype: entry point."""

import sys
import os

# Ensure we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_CONFIG
from engine import Engine
from gui import IMEGUI


def main():
    # Init engine
    engine = Engine()

    # Load saved cache
    engine.load_cache()

    # Start GUI
    gui = IMEGUI(engine=engine)
    gui.run()


if __name__ == "__main__":
    main()
