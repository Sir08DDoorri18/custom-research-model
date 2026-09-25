"""Starts the Chainlit UI. Used by `python -m research serve`.

Chainlit's CLI calls nest_asyncio.apply() on import, which breaks asyncio on Python 3.14
(anyio then finds "no running event loop" on every request). The UI doesn't need it, so it is
switched off before Chainlit is imported.
"""
import sys

import nest_asyncio

nest_asyncio.apply = lambda *args, **kwargs: None

from chainlit.cli import cli  # noqa: E402

if __name__ == "__main__":
    cli(["run", *sys.argv[1:]])
