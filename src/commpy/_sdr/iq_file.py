"""IQ recording file interoperability (raw binary and SigMF).

Two on-disk formats for complex baseband IQ samples:

- Raw binary: a headerless dump of interleaved real/imaginary `complex64`
  (or `complex128`) values, directly compatible with GNU Radio's
  `blocks.file_sink`/`blocks.file_source` (GNU Radio's default `gr_complex`
  is `complex64`).
- SigMF (https://github.com/sigmf/SigMF): a `<name>.sigmf-data` raw sample
  file plus a `<name>.sigmf-meta` JSON sidecar describing sample rate,
  datatype, and capture metadata, per the SigMF core namespace.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

_SIGMF_DATATYPES: dict[np.dtype[Any], str] = {
    np.dtype(np.complex64): 'cf32_le',
    np.dtype(np.complex128): 'cf64_le',
}
_SIGMF_DTYPE_FROM_DATATYPE: dict[str, np.dtype[Any]] = {
    v: k for k, v in _SIGMF_DATATYPES.items()
}


def write_iq(path: str | Path, samples: ArrayLike, dtype: DTypeLike = np.complex64) -> None:
    """Write complex baseband samples to a raw binary IQ file.

    The output is a headerless dump of interleaved real/imaginary values,
    directly readable by GNU Radio's `blocks.file_source` (or any tool
    expecting a raw `complex64`/`complex128` sample stream).

    Args:
        path: Output file path.
        samples: Complex baseband samples.
        dtype: On-disk sample dtype; `np.complex64` (GNU Radio's default) or
            `np.complex128`.

    Raises:
        ValueError: If `dtype` is not `complex64` or `complex128`.
    """
    dtype_arr = np.dtype(dtype)
    if dtype_arr not in _SIGMF_DATATYPES:
        msg = f'dtype must be complex64 or complex128, got {dtype_arr}.'
        raise ValueError(msg)
    np.asarray(samples, dtype=dtype_arr).tofile(str(path))


def read_iq(path: str | Path, dtype: DTypeLike = np.complex64) -> NDArray[np.complexfloating]:
    """Read complex baseband samples from a raw binary IQ file.

    Args:
        path: Input file path.
        dtype: On-disk sample dtype the file was written with.

    Returns:
        Complex baseband samples.
    """
    return np.fromfile(str(path), dtype=dtype)


def write_sigmf(  # noqa: PLR0913 -- each parameter is a distinct, standard SigMF metadata field
    path: str | Path,
    samples: ArrayLike,
    sample_rate: float,
    center_freq: float = 0.0,
    *,
    dtype: DTypeLike = np.complex64,
    description: str = '',
    author: str = '',
) -> None:
    """Write a SigMF recording (`.sigmf-data` + `.sigmf-meta` sidecar).

    Args:
        path: Base path (extension-less); writes `<path>.sigmf-data` and
            `<path>.sigmf-meta`.
        samples: Complex baseband samples.
        sample_rate: Sample rate, in Hz.
        center_freq: RF center frequency of the capture, in Hz.
        dtype: On-disk sample dtype; `np.complex64` or `np.complex128`.
        description: Free-text recording description.
        author: Recording author.

    Raises:
        ValueError: If `dtype` is not `complex64` or `complex128`.
    """
    dtype_arr = np.dtype(dtype)
    if dtype_arr not in _SIGMF_DATATYPES:
        msg = f'dtype must be complex64 or complex128, got {dtype_arr}.'
        raise ValueError(msg)
    base = str(path)
    write_iq(f'{base}.sigmf-data', samples, dtype=dtype_arr)

    meta: dict[str, Any] = {
        'global': {
            'core:datatype': _SIGMF_DATATYPES[dtype_arr],
            'core:sample_rate': float(sample_rate),
            'core:version': '1.0.0',
            'core:description': description,
            'core:author': author,
        },
        'captures': [
            {'core:sample_start': 0, 'core:frequency': float(center_freq)},
        ],
        'annotations': [],
    }
    Path(f'{base}.sigmf-meta').write_text(json.dumps(meta, indent=2))


def read_sigmf(path: str | Path) -> tuple[NDArray[np.complexfloating], dict[str, Any]]:
    """Read a SigMF recording (`.sigmf-data` + `.sigmf-meta` sidecar).

    Args:
        path: Base path (extension-less), or the path to either the
            `.sigmf-data` or `.sigmf-meta` file.

    Returns:
        `(samples, metadata)`: complex baseband samples and the parsed
        `.sigmf-meta` JSON as a dict.

    Raises:
        ValueError: If the metadata's `core:datatype` is not a supported
            complex format.
    """
    base = str(path)
    for suffix in ('.sigmf-data', '.sigmf-meta'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break

    meta = json.loads(Path(f'{base}.sigmf-meta').read_text())
    datatype = meta['global']['core:datatype']
    if datatype not in _SIGMF_DTYPE_FROM_DATATYPE:
        msg = f'Unsupported SigMF core:datatype {datatype!r}; expected cf32_le or cf64_le.'
        raise ValueError(msg)
    dtype = _SIGMF_DTYPE_FROM_DATATYPE[datatype]
    samples = read_iq(f'{base}.sigmf-data', dtype=dtype)
    return samples, meta


__all__ = ['read_iq', 'read_sigmf', 'write_iq', 'write_sigmf']
