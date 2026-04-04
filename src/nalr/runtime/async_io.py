from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Callable
import weakref


class AsyncIOWorker:
    def __init__(self, name: str) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._lock = Lock()
        self._tail: Future[None] | None = None
        self._last_error: BaseException | None = None
        self._finalizer = weakref.finalize(
            self,
            self._executor.shutdown,
            wait=False,
            cancel_futures=False,
        )

    def submit(
        self,
        fn: Callable[[], None],
        *,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        def wrapped() -> None:
            try:
                fn()
            except Exception as exc:  # pragma: no cover
                with self._lock:
                    self._last_error = exc
                if on_error is not None:
                    try:
                        on_error(exc)
                    except Exception:
                        pass

        with self._lock:
            self._tail = self._executor.submit(wrapped)

    def flush(self, *, raise_on_error: bool = False) -> None:
        with self._lock:
            tail = self._tail
        if tail is not None:
            tail.result()
        if raise_on_error:
            with self._lock:
                error = self._last_error
                self._last_error = None
            if error is not None:
                raise error

    def close(self, *, wait: bool = False) -> None:
        if self._finalizer.alive:
            self._finalizer.detach()
        self._executor.shutdown(wait=wait, cancel_futures=False)
