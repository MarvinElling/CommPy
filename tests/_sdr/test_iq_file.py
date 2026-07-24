"""Tests for commpy IQ file interoperability (raw binary and SigMF)."""

import numpy as np
import pytest

from commpy import read_iq, read_sigmf, write_iq, write_sigmf


def test_raw_iq_roundtrip_complex64(tmp_path, rng):
    samples = (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)).astype(np.complex64)
    path = tmp_path / 'recording.iq'
    write_iq(path, samples, dtype=np.complex64)
    recovered = read_iq(path, dtype=np.complex64)
    np.testing.assert_array_equal(recovered, samples)


def test_raw_iq_roundtrip_complex128(tmp_path, rng):
    samples = (rng.standard_normal(500) + 1j * rng.standard_normal(500)).astype(np.complex128)
    path = tmp_path / 'recording.iq'
    write_iq(path, samples, dtype=np.complex128)
    recovered = read_iq(path, dtype=np.complex128)
    np.testing.assert_array_equal(recovered, samples)


def test_write_iq_rejects_non_complex_dtype(tmp_path):
    with pytest.raises(ValueError, match='dtype'):
        write_iq(tmp_path / 'recording.iq', np.ones(10), dtype=np.float32)


def test_sigmf_roundtrip_default_dtype(tmp_path, rng):
    samples = (rng.standard_normal(2000) + 1j * rng.standard_normal(2000)).astype(np.complex64)
    base = tmp_path / 'capture'
    write_sigmf(base, samples, sample_rate=1e6, center_freq=915e6, description='test', author='ci')

    recovered, meta = read_sigmf(base)
    np.testing.assert_array_equal(recovered, samples)
    assert meta['global']['core:datatype'] == 'cf32_le'
    assert meta['global']['core:sample_rate'] == 1e6
    assert meta['global']['core:description'] == 'test'
    assert meta['global']['core:author'] == 'ci'
    assert meta['captures'][0]['core:frequency'] == 915e6


def test_sigmf_roundtrip_complex128(tmp_path, rng):
    samples = (rng.standard_normal(200) + 1j * rng.standard_normal(200)).astype(np.complex128)
    base = tmp_path / 'capture'
    write_sigmf(base, samples, sample_rate=2e6, dtype=np.complex128)

    recovered, meta = read_sigmf(base)
    np.testing.assert_array_equal(recovered, samples)
    assert meta['global']['core:datatype'] == 'cf64_le'


def test_read_sigmf_accepts_data_or_meta_path(tmp_path, rng):
    samples = (rng.standard_normal(10) + 1j * rng.standard_normal(10)).astype(np.complex64)
    base = tmp_path / 'capture'
    write_sigmf(base, samples, sample_rate=1e6)

    from_data = read_sigmf(f'{base}.sigmf-data')[0]
    from_meta = read_sigmf(f'{base}.sigmf-meta')[0]
    np.testing.assert_array_equal(from_data, samples)
    np.testing.assert_array_equal(from_meta, samples)


def test_write_sigmf_rejects_non_complex_dtype(tmp_path):
    with pytest.raises(ValueError, match='dtype'):
        write_sigmf(tmp_path / 'capture', np.ones(10), sample_rate=1e6, dtype=np.int16)


def test_read_sigmf_rejects_unsupported_datatype(tmp_path):
    base = tmp_path / 'capture'
    (tmp_path / 'capture.sigmf-data').write_bytes(b'\x00' * 8)
    (tmp_path / 'capture.sigmf-meta').write_text(
        '{"global": {"core:datatype": "ri16_le"}, "captures": [], "annotations": []}',
    )
    with pytest.raises(ValueError, match='core:datatype'):
        read_sigmf(base)
