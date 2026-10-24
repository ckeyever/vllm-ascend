# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Instance-owned CPU/SSD cache state; deliberately independent of torch."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import threading
import uuid
from collections import Counter, OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Entry:
    key: bytes
    slot: int
    generation: int
    token: str
    owner: int
    state: str = "writing"
    pins: set[str] = field(default_factory=set)
    completed_workers: set[int] = field(default_factory=set)


@dataclass
class DiskEntry:
    token: str
    state: str = "writing"
    checksum: str = ""
    readers: int = 0


class SharedCache:
    """One authoritative index, allocator and set of I/O queues per instance.

    Only the coordinator process owns this object. RPC threads and I/O
    completions serialize short state changes with _lock; DMA never runs here.
    """

    def __init__(
        self,
        runtime_dir: str,
        cpu_bytes: int,
        dp_size: int,
        world_size: int,
        disk_dir: str | None = None,
        disk_bytes: int = 0,
        read_threads: int = 4,
        write_threads: int = 2,
        enable_events: bool = False,
        pool_epoch: str | None = None,
    ) -> None:
        if min(cpu_bytes, dp_size, world_size, read_threads, write_threads) <= 0 or disk_bytes < 0:
            raise ValueError("Shared cache sizes and I/O thread counts must be positive")
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.region_path = str(self.runtime_dir / "kv.mmap")
        self.disk_dir = Path(disk_dir) if disk_dir else None
        if self.disk_dir is not None:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
        self._epoch = pool_epoch or uuid.uuid4().hex
        self.cpu_bytes = cpu_bytes
        self.disk_bytes = disk_bytes
        self.dp_size = dp_size
        self.world_size = world_size
        self._lock = threading.RLock()
        self._workers: dict[tuple[int, int], dict] = {}
        self._identity: str | None = None
        self._layout: dict | None = None
        self._region: mmap.mmap | None = None
        self._entries: OrderedDict[bytes, Entry] = OrderedDict()
        self._disk: OrderedDict[bytes, DiskEntry] = OrderedDict()
        self._free: deque[int] = deque()
        self._generations: list[int] = []
        self._quarantined: set[int] = set()
        self._tickets: dict[str, tuple[Entry, str]] = {}
        self._scopes: dict[tuple[int, str], dict[bytes, str]] = {}
        self._operations: dict[tuple[int, str], dict] = {}
        self._read_pool = ThreadPoolExecutor(read_threads, thread_name_prefix="shared-kv-read")
        self._write_pool = ThreadPoolExecutor(write_threads, thread_name_prefix="shared-kv-write")
        self._futures: set[Future] = set()
        self._stats: Counter = Counter()
        self._events: list[dict] | None = [] if enable_events else None
        self._closed = False

    def register_worker(self, dp: int, rank: int, schema: dict) -> dict | None:
        with self._lock:
            if not 0 <= dp < self.dp_size or not 0 <= rank < self.world_size:
                raise ValueError("Worker is outside the shared instance's DP/TP/PP topology")
            identity = json.dumps(schema["identity"], sort_keys=True)
            if self._identity is not None and self._identity != identity:
                raise ValueError("Cross-DP KV model, groups, hash seed or configuration mismatch")
            self._identity = identity
            existing = self._workers.get((dp, rank))
            if existing is not None and existing != schema:
                raise ValueError("Worker changed its registered KV layout")
            for (other_dp, other_rank), other in self._workers.items():
                if other_rank == rank and other != schema:
                    raise ValueError(f"KV layout mismatch between DP{other_dp} and DP{dp}, worker {rank}")
            self._workers[dp, rank] = schema
            if self._layout is None and len(self._workers) == self.dp_size * self.world_size:
                self._allocate_region()
            return self._layout

    def _allocate_region(self) -> None:
        offsets = []
        offset = 0
        for rank in range(self.world_size):
            offsets.append(offset)
            offset += sum(self._workers[0, rank]["page_sizes"])
        stride = (offset + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        if stride == 0 or self.cpu_bytes < stride:
            raise ValueError("cpu_bytes_to_use cannot hold one shared KV chunk")
        chunks = self.cpu_bytes // stride
        total_bytes = chunks * stride
        free_bytes = os.statvfs(self.runtime_dir).f_bavail * os.statvfs(self.runtime_dir).f_frsize
        if free_bytes < total_bytes:
            raise ValueError(f"Shared KV region needs {total_bytes} bytes, only {free_bytes} available")
        fd = os.open(self.region_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, total_bytes)
            self._region = mmap.mmap(fd, total_bytes)
        finally:
            os.close(fd)
        schema_bytes = json.dumps([self._workers[0, rank] for rank in range(self.world_size)], sort_keys=True).encode()
        self._layout = {
            "path": self.region_path,
            "stride": stride,
            "chunks": chunks,
            "offsets": offsets,
            "schema_digest": hashlib.sha256(schema_bytes).hexdigest(),
        }
        self._generations = [0] * chunks
        self._free.extend(range(chunks))

    def layout(self) -> dict | None:
        with self._lock:
            return self._layout

    def _allocate(self, key: bytes, owner: int, state: str = "writing") -> Entry | None:
        if not self._free:
            victim = next((e for e in self._entries.values() if e.state == "ready" and not e.pins), None)
            if victim is None:
                return None
            del self._entries[victim.key]
            self._free.append(victim.slot)
            self._stats["dram_evictions"] += 1
            self._emit(victim.key, "CPU", True)
        slot = self._free.popleft()
        self._generations[slot] += 1
        entry = Entry(key, slot, self._generations[slot], uuid.uuid4().hex, owner, state)
        assert self._layout is not None and self._region is not None
        start = slot * self._layout["stride"]
        # Every group owns a complete row; clear unused group components/padding.
        self._region[start : start + self._layout["stride"]] = bytes(self._layout["stride"])
        self._entries[key] = entry
        return entry

    def _descriptor(self, entry: Entry) -> dict:
        return {
            "pool_epoch": self._epoch,
            "key": entry.key,
            "slot": entry.slot,
            "generation": entry.generation,
            "token": entry.token,
        }

    def reserve_store(self, dp: int, operation: str, keys: list[bytes]) -> dict:
        with self._lock:
            if self._layout is None or self._closed:
                raise RuntimeError("Shared KV cache is not ready")
            operation_key = (dp, operation)
            if operation_key in self._operations:
                return self._operations[operation_key]
            writes, statuses = [], []
            for key in keys:
                entry = self._entries.get(key)
                disk = self._disk.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
                    statuses.append("exists" if entry.state == "ready" else "pending")
                    self._stats["deduplicated_stores"] += 1
                    if entry.state == "ready":
                        self._cascade(entry)
                elif disk is not None and disk.state == "ready":
                    self._disk.move_to_end(key)
                    statuses.append("exists")
                    self._stats["deduplicated_stores"] += 1
                else:
                    entry = self._allocate(key, dp)
                    if entry is None:
                        statuses.append("capacity")
                    else:
                        writes.append(self._descriptor(entry))
                        statuses.append("write")
            result = {"writes": writes, "statuses": statuses}
            self._operations[operation_key] = result
            return result

    def forget_operation(self, dp: int, operation: str) -> None:
        with self._lock:
            self._operations.pop((dp, operation), None)

    def _matches(self, descriptor: dict) -> Entry | None:
        entry = self._entries.get(descriptor["key"])
        if entry is None or descriptor.get("pool_epoch") != self._epoch:
            return None
        if (entry.slot, entry.generation, entry.token) != (
            descriptor["slot"],
            descriptor["generation"],
            descriptor["token"],
        ):
            return None
        return entry

    def commit_store(self, dp: int, descriptors: list[dict]) -> None:
        with self._lock:
            for descriptor in descriptors:
                entry = self._matches(descriptor)
                if entry is None or entry.owner != dp or entry.state != "writing":
                    continue
                if entry.completed_workers != set(range(self.world_size)):
                    raise RuntimeError("Cannot publish shared KV before every TP/PP worker completes")
                entry.state = "ready"
                self._stats["dram_stores"] += 1
                self._emit(entry.key, "CPU", False)
                self._cascade(entry)

    def ack_store(self, dp: int, rank: int, descriptors: list[dict]) -> None:
        with self._lock:
            if not 0 <= rank < self.world_size:
                raise ValueError("Invalid shared KV worker rank")
            for descriptor in descriptors:
                entry = self._matches(descriptor)
                if entry is not None and entry.owner == dp and entry.state == "writing":
                    entry.completed_workers.add(rank)

    def abort_store(self, dp: int, descriptors: list[dict]) -> None:
        with self._lock:
            for descriptor in descriptors:
                entry = self._matches(descriptor)
                if entry is not None and entry.owner == dp and entry.state == "writing":
                    del self._entries[entry.key]
                    # A timeout/abort does NOT prove that old NPU DMA has stopped.
                    self._quarantined.add(entry.slot)
                    self._stats["quarantined_slots"] += 1

    def _track(self, future: Future) -> None:
        self._futures.add(future)
        future.add_done_callback(self._finished)

    def _finished(self, future: Future) -> None:
        with self._lock:
            self._futures.discard(future)

    def _disk_path(self, key: bytes) -> Path:
        assert self.disk_dir is not None
        return self.disk_dir / (key.hex() + ".bin")

    def _make_disk_space(self) -> bool:
        assert self._layout is not None
        if not self.disk_bytes:
            return True
        stride = self._layout["stride"]
        while (len(self._disk) + 1) * stride > self.disk_bytes:
            victim = next(((k, d) for k, d in self._disk.items() if d.state == "ready" and not d.readers), None)
            if victim is None:
                return False
            key, _ = victim
            try:
                self._disk_path(key).unlink(missing_ok=True)
                self._disk_path(key).with_suffix(".json").unlink(missing_ok=True)
            except OSError:
                self._stats["ssd_write_errors"] += 1
                return False
            del self._disk[key]
            self._stats["ssd_evictions"] += 1
            self._emit(key, "STORAGE", True)
        return True

    def _cascade(self, entry: Entry) -> None:
        if self.disk_dir is None or entry.key in self._disk or self._closed:
            return
        if not self._make_disk_space():
            self._stats["ssd_capacity_skips"] += 1
            return
        token = uuid.uuid4().hex
        self._disk[entry.key] = DiskEntry(token)
        entry.pins.add(token)
        self._track(self._write_pool.submit(self._write_disk, entry, token))

    def _write_disk(self, entry: Entry, token: str) -> None:
        assert self._region is not None and self._layout is not None
        path = self._disk_path(entry.key)
        temporary = path.with_name(path.name + "." + token + ".tmp")
        manifest = path.with_suffix(".json")
        manifest_tmp = temporary.with_suffix(".json.tmp")
        stride = self._layout["stride"]
        view = memoryview(self._region)[entry.slot * stride : (entry.slot + 1) * stride]
        success = False
        try:
            checksum = hashlib.sha256(view).hexdigest()
            with temporary.open("xb", buffering=0) as stream:
                remaining = view
                while remaining:
                    written = stream.write(remaining)
                    if not written:
                        raise OSError("Short shared KV cache write")
                    remaining = remaining[written:]
                remaining.release()
            manifest_tmp.write_text(
                json.dumps(
                    {
                        "pool_epoch": self._epoch,
                        "schema_digest": self._layout["schema_digest"],
                        "key": entry.key.hex(),
                        "bytes": stride,
                        "sha256": checksum,
                    }
                )
            )
            with self._lock:
                disk = self._disk.get(entry.key)
                if disk is not None and disk.token == token:
                    os.replace(temporary, path)
                    os.replace(manifest_tmp, manifest)
                    disk.state = "ready"
                    disk.checksum = checksum
                    self._stats["ssd_stores"] += 1
                    self._emit(entry.key, "STORAGE", False)
                    success = True
        except OSError:
            with self._lock:
                self._stats["ssd_write_errors"] += 1
        finally:
            view.release()
            # Cleanup failure must not strand a DRAM pin or an SSD reservation.
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            with suppress(OSError):
                manifest_tmp.unlink(missing_ok=True)
            with self._lock:
                if not success and self._disk.get(entry.key, DiskEntry("")).token == token:
                    self._disk.pop(entry.key, None)
                    with suppress(OSError):
                        path.unlink(missing_ok=True)
                    with suppress(OSError):
                        manifest.unlink(missing_ok=True)
                entry.pins.discard(token)

    def lookup(self, dp: int, scope: str, keys: list[bytes], allow_disk: bool = True) -> list[dict]:
        with self._lock:
            tickets = self._scopes.setdefault((dp, scope), {})
            results = []
            for key in keys:
                entry = self._entries.get(key)
                disk = self._disk.get(key)
                if entry is not None and entry.state == "ready":
                    self._entries.move_to_end(key)
                    ticket = tickets.get(key)
                    if ticket is None:
                        ticket = uuid.uuid4().hex
                        tickets[key] = ticket
                        self._tickets[ticket] = (entry, "lookup")
                        entry.pins.add(ticket)
                    results.append({"status": "hit", "ticket": ticket, **self._descriptor(entry)})
                    self._stats["dram_hits"] += 1
                elif entry is not None:
                    results.append({"status": "pending"})
                elif allow_disk and disk is not None and disk.state == "ready":
                    entry = self._allocate(key, dp, "promoting")
                    if entry is None:
                        results.append({"status": "miss"})
                    else:
                        disk.readers += 1
                        self._disk.move_to_end(key)
                        self._track(self._read_pool.submit(self._read_disk, entry, disk))
                        results.append({"status": "pending", "source": "disk"})
                else:
                    results.append({"status": "miss"})
            return results

    def _read_disk(self, entry: Entry, disk: DiskEntry) -> None:
        assert self._region is not None and self._layout is not None
        stride = self._layout["stride"]
        view = memoryview(self._region)[entry.slot * stride : (entry.slot + 1) * stride]
        success = False
        try:
            path = self._disk_path(entry.key)
            metadata = json.loads(path.with_suffix(".json").read_text())
            if metadata != {
                "pool_epoch": self._epoch,
                "schema_digest": self._layout["schema_digest"],
                "key": entry.key.hex(),
                "bytes": stride,
                "sha256": disk.checksum,
            }:
                raise OSError("Shared SSD KV manifest mismatch")
            if path.stat().st_size != stride:
                raise OSError("Shared SSD KV file size mismatch")
            with path.open("rb", buffering=0) as stream:
                offset = 0
                while offset < stride:
                    count = stream.readinto(view[offset:])
                    if not count:
                        raise OSError("Short shared SSD KV read")
                    offset += count
            if hashlib.sha256(view).hexdigest() != disk.checksum:
                raise OSError("Shared SSD KV checksum mismatch")
            success = True
        except (OSError, ValueError):
            with self._lock:
                self._stats["ssd_read_errors"] += 1
        finally:
            view.release()
            with self._lock:
                disk.readers -= 1
                if success:
                    entry.state = "ready"
                    self._stats["ssd_promotions"] += 1
                    self._emit(entry.key, "CPU", False)
                else:
                    self._entries.pop(entry.key, None)
                    self._free.append(entry.slot)
                    self._disk.pop(entry.key, None)
                    with suppress(OSError):
                        self._disk_path(entry.key).unlink(missing_ok=True)
                    with suppress(OSError):
                        self._disk_path(entry.key).with_suffix(".json").unlink(missing_ok=True)

    def prepare_load(self, dp: int, scope: str, keys: list[bytes]) -> list[dict]:
        with self._lock:
            scoped = self._scopes.get((dp, scope), {})
            if any(key not in scoped for key in keys):
                raise RuntimeError("Shared cache HIT must have a live lookup ticket")
            result = []
            for key in keys:
                ticket = scoped.pop(key)
                entry, kind = self._tickets[ticket]
                if kind != "lookup":
                    raise RuntimeError("Lookup ticket was already consumed")
                self._tickets[ticket] = (entry, "load")
                result.append({"lease": ticket, **self._descriptor(entry)})
            return result

    def release_scope(self, dp: int, scope: str) -> None:
        with self._lock:
            for ticket in self._scopes.pop((dp, scope), {}).values():
                item = self._tickets.pop(ticket, None)
                if item is not None:
                    item[0].pins.discard(ticket)

    def release_load(self, leases: list[str]) -> None:
        with self._lock:
            for lease in leases:
                item = self._tickets.get(lease)
                if item is not None and item[1] == "load":
                    del self._tickets[lease]
                    item[0].pins.discard(lease)

    def validate_transfer(self, dp: int, descriptors: list[dict], store: bool) -> None:
        with self._lock:
            for descriptor in descriptors:
                entry = self._matches(descriptor)
                if entry is None:
                    raise RuntimeError("Stale shared KV slot or generation")
                if store and (entry.owner != dp or entry.state != "writing"):
                    raise RuntimeError("DP does not own this shared KV write")
                if not store and (entry.state != "ready" or descriptor["lease"] not in entry.pins):
                    raise RuntimeError("Shared KV load has no live read lease")

    def touch(self, keys: list[bytes]) -> None:
        with self._lock:
            for key in keys:
                if key in self._entries:
                    self._entries.move_to_end(key)
                if key in self._disk:
                    self._disk.move_to_end(key)

    def _emit(self, key: bytes, medium: str, removed: bool) -> None:
        if self._events is not None:
            self._events.append({"keys": [key], "medium": medium, "removed": removed})

    def take_events(self, dp: int) -> list[dict]:
        with self._lock:
            if dp != 0 or self._events is None:
                return []
            result, self._events = self._events, []
            return result

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats) | {
                "dram_chunks": len(self._entries),
                "ssd_chunks": len(self._disk),
                "read_tickets": len(self._tickets),
                "pending_io": len(self._futures),
                "quarantined_slots": len(self._quarantined),
            }

    def reset(self) -> None:
        with self._lock:
            if self._tickets or self._futures or any(e.state != "ready" for e in self._entries.values()):
                raise RuntimeError("Shared KV reset requires all DP transfers and lookups to be idle")
            self._free.extend(e.slot for e in self._entries.values())
            self._entries.clear()
            for key in self._disk:
                self._disk_path(key).unlink(missing_ok=True)
                self._disk_path(key).with_suffix(".json").unlink(missing_ok=True)
            self._disk.clear()
            self._operations.clear()
            if self._events is not None:
                self._events.clear()

    def drain(self) -> None:
        while True:
            with self._lock:
                futures = list(self._futures)
            if not futures:
                return
            for future in futures:
                future.result()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._write_pool.shutdown(wait=True)
        self._read_pool.shutdown(wait=True)
        if self._region is not None:
            self._region.close()
            self._region = None
