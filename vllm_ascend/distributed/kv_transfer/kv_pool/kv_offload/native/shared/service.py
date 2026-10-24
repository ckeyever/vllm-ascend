# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Authenticated local multiprocessing transport and parent-owned service."""

from __future__ import annotations

import multiprocessing
import shutil
import tempfile
import threading
import uuid
from contextlib import suppress
from multiprocessing.connection import Client, Listener
from pathlib import Path

from .coordinator import SharedCache

RPC_TIMEOUT = 60
SERVICE_START_TIMEOUT = 60
SERVICE_STOP_TIMEOUT = 30
RPC_METHODS = frozenset(
    {
        "register_worker",
        "layout",
        "reserve_store",
        "forget_operation",
        "commit_store",
        "ack_store",
        "abort_store",
        "lookup",
        "prepare_load",
        "release_scope",
        "release_load",
        "validate_transfer",
        "touch",
        "stats",
        "take_events",
        "reset",
        "drain",
    }
)


class SharedClient:
    def __init__(self, connection: dict) -> None:
        self.connection = connection

    def call(self, method: str, *args):
        address = self.connection["address"]
        authkey = bytes.fromhex(self.connection["authkey"])
        with Client(address, family="AF_UNIX", authkey=authkey) as connection:
            connection.send((method, args))
            if not connection.poll(RPC_TIMEOUT):
                raise RuntimeError(f"Shared KV coordinator timed out during {method}")
            success, result = connection.recv()
        if not success:
            kind, message = result
            if kind == "ValueError":
                raise ValueError(message)
            raise RuntimeError(message)
        return result


def _serve(connection: dict, settings: dict, ready) -> None:
    cache = None
    listener = None
    try:
        cache = SharedCache(**settings)
        listener = Listener(connection["address"], family="AF_UNIX", authkey=bytes.fromhex(connection["authkey"]))
        ready.send((True, None))
        ready.close()
        while True:
            with listener.accept() as client:
                method, args = client.recv()
                try:
                    if method == "shutdown":
                        cache.close()
                        client.send((True, None))
                        break
                    if method not in RPC_METHODS:
                        raise ValueError("Unknown shared KV operation")
                    result = getattr(cache, method)(*args)
                    client.send((True, result))
                except Exception as exc:
                    client.send((False, (type(exc).__name__, str(exc))))
    except BaseException as exc:
        if not ready.closed:
            ready.send((False, str(exc)))
        raise
    finally:
        if listener is not None:
            listener.close()
        if cache is not None:
            cache.close()


class SharedService:
    """Owned by CoreEngineProcManager, not the startup-only launch context."""

    def __init__(self, settings: dict, context=None) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="ascend-kv-", dir="/dev/shm")
        runtime_dir = self._directory.name
        self._disk_directory: Path | None = None
        disk_root = settings.pop("disk_root", None)
        instance = uuid.uuid4().hex
        if disk_root:
            self._disk_directory = Path(disk_root).resolve() / ("ascend-" + instance)
        self.connection = {"address": str(Path(runtime_dir) / "control"), "authkey": uuid.uuid4().hex}
        settings = settings | {
            "runtime_dir": runtime_dir,
            "pool_epoch": instance,
            "disk_dir": str(self._disk_directory) if self._disk_directory is not None else None,
        }
        context = context or multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        self.process = context.Process(
            target=_serve, args=(self.connection, settings, sender), name="AscendSharedKVCache"
        )
        self._stopped = False
        try:
            self.process.start()
            sender.close()
            if not receiver.poll(SERVICE_START_TIMEOUT):
                raise RuntimeError("Shared KV coordinator failed to start")
            success, error = receiver.recv()
            if not success:
                raise RuntimeError(error)
        except BaseException:
            self.close()
            raise
        finally:
            receiver.close()
            sender.close()

    def close(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.process.pid is not None:
            if self.process.is_alive():
                # Authentication/SSD drain can block too: keep parent cleanup bounded.
                def request_shutdown():
                    with suppress(OSError, EOFError, RuntimeError):
                        SharedClient(self.connection).call("shutdown")

                threading.Thread(target=request_shutdown, daemon=True).start()
            self.process.join(SERVICE_STOP_TIMEOUT)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(SERVICE_STOP_TIMEOUT)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
        self._directory.cleanup()
        if self._disk_directory is not None and self._disk_directory.exists():
            # This uniquely named directory was created by this instance only.
            shutil.rmtree(self._disk_directory)
