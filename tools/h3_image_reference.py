#!/usr/bin/env python3
"""Reference-dump harness for the MiniMax-H3-Image path of h3.c.

Entry point only; the stages live in the h3_image_ref package next to this
file, and `--help` prints their full description. Run from the repository root
in an environment that has diffusers (with the MiniMax-H3-Image classes),
transformers, torch and safetensors.
"""

import os
import sys

# python -I leaves the script directory off sys.path, so add it explicitly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h3_image_ref.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
