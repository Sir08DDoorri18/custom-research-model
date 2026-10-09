"""Starts the Chainlit UI. Used by `python -m research serve`.

Chainlit's CLI calls nest_asyncio.apply() on import, which breaks asyncio on Python 3.14
(anyio then finds "no running event loop" on every request). The UI doesn't need it, so it is
switched off before Chainlit is imported.

Saved conversations need a logged-in user; research/app.py logs everyone in as the one local
user, and Chainlit signs that login with CHAINLIT_AUTH_SECRET, kept in .cache/auth_secret.
"""
import os
import sys

import nest_asyncio

nest_asyncio.apply = lambda *args, **kwargs: None

from chainlit.cli import cli  # noqa: E402

if __name__ == "__main__":
    from research.history import auth_secret
    os.environ.setdefault("CHAINLIT_AUTH_SECRET", auth_secret())
    cli(["run", *sys.argv[1:]])
