#!/usr/bin/env python3
"""IME prototype: entry point."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from engine import Engine
from gui import IMEGUI


def main():
    engine = Engine(config=load_config())
    gui = IMEGUI(engine=engine)
    gui.run()


if __name__ == "__main__":
    main()
