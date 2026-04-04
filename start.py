import sys
import weakref

# AnyIO Python 3.14 None-Task Patch
class TaskStateDict:
    def __init__(self):
        self.wd = weakref.WeakKeyDictionary()
        self.none_state = None

    def __getitem__(self, key):
        if key is None:
            if self.none_state is None:
                raise KeyError(None)
            return self.none_state
        return self.wd[key]

    def __setitem__(self, key, value):
        if key is None:
            self.none_state = value
        else:
            self.wd[key] = value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return default

if sys.version_info >= (3, 13):
    import anyio._backends._asyncio
    anyio._backends._asyncio._task_states = TaskStateDict()
    
    import anyio._core._eventloop
    from anyio._backends._asyncio import AsyncIOBackend
    original_get_backend = anyio._core._eventloop.get_async_backend
    def patched_get_backend():
        try:
            return original_get_backend()
        except anyio._core._eventloop.NoEventLoopError:
            return AsyncIOBackend()
    anyio._core._eventloop.get_async_backend = patched_get_backend

import chainlit.cli

if __name__ == '__main__':
    sys.argv = ["chainlit", "run", "app.py", "--port", "8001"]
    sys.exit(chainlit.cli.cli())
