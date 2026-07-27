"""Tests for the commpy-mcp server.

The tool implementations are plain functions returning JSON-serializable dicts,
so they are tested directly (no `mcp` package needed). Server registration is
tested separately and skipped when the optional `mcp` extra is absent.
"""

import asyncio

import pytest

from commpy import mcp_server


def test_list_capabilities_reports_codes_and_modulations():
    caps = mcp_server.list_capabilities()
    assert set(caps['fec_codes']) == {'ldpc', 'polar', 'turbo'}
    assert set(caps['modulations']) == {'psk', 'qam', 'pam'}
    assert 'awgn' in caps['channels']


def test_channel_capacity_awgn_and_bsc():
    awgn = mcp_server.channel_capacity('awgn', snr_db=10.0)
    assert awgn['capacity_bits_per_use'] > 3.0  # ~3.46 bits/use at 10 dB
    bsc = mcp_server.channel_capacity('bsc', error_probability=0.5)
    assert bsc['capacity_bits_per_use'] == pytest.approx(0.0, abs=1e-9)  # p=0.5 -> zero capacity
    with pytest.raises(ValueError, match='channel'):
        mcp_server.channel_capacity('bogus')


def test_run_ber_simulation_returns_decreasing_points():
    result = mcp_server.run_ber_simulation('psk', 4, [2.0, 10.0], target_errors=50, max_bits=40_000)
    points = result['points']
    assert result['modulation'] == 'psk'
    assert len(points) == 2
    assert all(0.0 <= p['error_rate'] <= 1.0 for p in points)
    assert points[1]['error_rate'] < points[0]['error_rate']
    with pytest.raises(ValueError, match='modulation'):
        mcp_server.run_ber_simulation('bogus', 4, [5.0])


@pytest.mark.parametrize('code', ['ldpc', 'polar', 'turbo'])
def test_run_coded_ber_simulation_reports_code_and_points(code):
    result = mcp_server.run_coded_ber_simulation(
        code, [3.0], target_errors=10, max_blocks=40,
    )
    assert result['code'] == code
    assert result['n'] > result['k'] > 0
    assert 0.0 < result['rate'] < 1.0
    assert len(result['points']) == 1
    with pytest.raises(ValueError, match='code'):
        mcp_server.run_coded_ber_simulation('bogus', [3.0])


def test_build_server_registers_all_tools():
    pytest.importorskip('mcp.server.fastmcp')  # skip when the mcp extra is absent
    server = mcp_server.build_server()
    tool_names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert tool_names == {
        'list_capabilities', 'channel_capacity', 'run_ber_simulation', 'run_coded_ber_simulation',
    }
