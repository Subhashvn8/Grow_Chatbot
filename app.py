"""Vercel entrypoint.

Vercel looks for a FastAPI app named ``app`` at the repository root. This
module exposes the same FastAPI instance used by ``grow-ui`` locally, so the
hosted page and the CLI still share one ask path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from grow_faq.config import load_settings
from grow_faq.ui.app import app as grow_app, configure

configure(load_settings())

app = grow_app
