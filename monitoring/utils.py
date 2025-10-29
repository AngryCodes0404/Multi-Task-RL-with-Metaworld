import os
import threading
import time
from typing import Any, TYPE_CHECKING, Optional

import flax.struct
import flax.traverse_util
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, Float, PyTree
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import HistogramProto, Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

if TYPE_CHECKING:
    from metaworld_types import LogDict


class Histogram(flax.struct.PyTreeNode):
    data: Float[npt.NDArray | Array, "..."] | None = None
    np_histogram: tuple | None = None


class _TensorBoardWriter:
    """Lightweight tensorboard writer that does not depend on torch."""

    def __init__(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self._writer = EventFileWriter(log_dir)
        self._lock = threading.Lock()
        self._logdir = self._writer.get_logdir()
        print(f"[tensorboard] Event files will be written under: {self._logdir}")
        # Write file_version event like standard SummaryWriter
        self._write_event(Event(file_version="brain.Event:2"))

    def _write_event(self, event: Event) -> None:
        with self._lock:
            self._writer.add_event(event)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        event = Event(
            wall_time=time.time(),
            step=int(step),
            summary=Summary(value=[Summary.Value(tag=tag, simple_value=float(value))]),
        )
        self._write_event(event)

    def add_histogram_from_array(
        self, tag: str, values: npt.NDArray[np.floating], step: int, bins: int = 64
    ) -> None:
        counts, bin_edges = np.histogram(values, bins=bins)
        self.add_histogram_raw(tag, counts, bin_edges, step)

    def add_histogram_raw(
        self,
        tag: str,
        counts: npt.NDArray,
        bin_edges: npt.NDArray,
        step: int,
    ) -> None:
        counts_np = np.asarray(counts, dtype=np.float64)
        bin_edges_np = np.asarray(bin_edges, dtype=np.float64)

        if counts_np.size == 0 or counts_np.sum() == 0:
            return

        bin_mids = (bin_edges_np[:-1] + bin_edges_np[1:]) * 0.5

        hist = HistogramProto()
        hist.min = float(bin_edges_np[0])
        hist.max = float(bin_edges_np[-1])
        hist.num = float(counts_np.sum())
        hist.sum = float((bin_mids * counts_np).sum())
        hist.sum_squares = float(((bin_mids ** 2) * counts_np).sum())
        hist.bucket_limit.extend(bin_edges_np[1:].tolist())
        hist.bucket.extend(counts_np.tolist())

        event = Event(
            wall_time=time.time(),
            step=int(step),
            summary=Summary(value=[Summary.Value(tag=tag, histo=hist)]),
        )
        self._write_event(event)

    def flush(self) -> None:
        with self._lock:
            self._writer.flush()

    def close(self) -> None:
        with self._lock:
            self._writer.close()


# Global TensorBoard writer for training
_tensorboard_writer: Optional[_TensorBoardWriter] = None

def set_tensorboard_writer(log_dir: str) -> None:
    """Set up global TensorBoard writer for training logging."""
    global _tensorboard_writer
    os.makedirs(log_dir, exist_ok=True)
    print(f"Initializing TensorBoard writer at: {log_dir}")
    _tensorboard_writer = _TensorBoardWriter(log_dir)

def close_tensorboard_writer() -> None:
    """Close the global TensorBoard writer."""
    global _tensorboard_writer
    if _tensorboard_writer is not None:
        _tensorboard_writer.close()
        _tensorboard_writer = None

def log(logs: dict, step: int) -> None:
    """Log function for TensorBoard."""
    if _tensorboard_writer is not None:
        for key, value in logs.items():
            if isinstance(value, Histogram):
                # Convert histogram to TensorBoard format
                if value.data is not None:
                    data_np = np.asarray(value.data)
                    _tensorboard_writer.add_histogram_from_array(key, data_np, step)
                elif value.np_histogram is not None:
                    counts, bin_edges = value.np_histogram
                    _tensorboard_writer.add_histogram_raw(
                        key, counts, bin_edges, step
                    )
            elif isinstance(value, (int, float)):
                _tensorboard_writer.add_scalar(key, value, step)
            elif hasattr(value, 'item'):  # JAX/NumPy scalars
                _tensorboard_writer.add_scalar(key, value.item(), step)
            elif hasattr(value, '__array__'):  # JAX/NumPy arrays
                # Convert JAX arrays to NumPy for TensorBoard
                value_np = np.array(value)
                if value_np.ndim == 0:  # Scalar array
                    _tensorboard_writer.add_scalar(key, value_np.item(), step)
                else:  # Multi-dimensional array - log as histogram
                    _tensorboard_writer.add_histogram_from_array(key, value_np, step)
        _tensorboard_writer.flush()


def get_logs(
    name: str,
    data: Float[npt.NDArray | Array, "..."],
    axis: int | None = None,
    hist: bool = True,
    std: bool = True,
) -> "LogDict":
    ret: "LogDict" = {
        f"{name}_mean": jnp.mean(data, axis=axis),
        f"{name}_min": jnp.min(data, axis=axis),
        f"{name}_max": jnp.max(data, axis=axis),
    }
    if std:
        ret[f"{name}_std"] = jnp.std(data, axis=axis)
    if hist:
        ret[f"{name}"] = Histogram(data.reshape(-1))

    return ret


def prefix_dict(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def pytree_histogram(pytree: PyTree, bins: int = 64) -> dict[str, Histogram]:
    flat_dict = flax.traverse_util.flatten_dict(pytree, sep="/")
    ret = {}
    for k, v in flat_dict.items():
        if isinstance(v, tuple):  # For activations
            v = v[0]
        ret[k] = Histogram(np_histogram=jnp.histogram(v, bins=bins))  # pyright: ignore[reportArgumentType]
    return ret
