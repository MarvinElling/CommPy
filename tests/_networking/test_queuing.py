"""Tests for commpy.MM1Queue / MM1KQueue / MMcQueue."""

import pytest

from commpy import MM1KQueue, MM1Queue, MMcQueue


def test_mm1_known_values():
    # lambda=1, mu=2 -> rho=0.5, L=1, Lq=0.5, W=1, Wq=0.5 (textbook example).
    q = MM1Queue(arrival_rate=1.0, service_rate=2.0)
    assert q.utilization == pytest.approx(0.5)
    assert q.mean_number_in_system == pytest.approx(1.0)
    assert q.mean_number_in_queue == pytest.approx(0.5)
    assert q.mean_wait_in_system == pytest.approx(1.0)
    assert q.mean_wait_in_queue == pytest.approx(0.5)


def test_mm1_littles_law():
    q = MM1Queue(arrival_rate=3.0, service_rate=5.0)
    assert q.mean_number_in_system == pytest.approx(q.arrival_rate * q.mean_wait_in_system)
    assert q.mean_number_in_queue == pytest.approx(q.arrival_rate * q.mean_wait_in_queue)


def test_mm1_state_probabilities_sum_to_one():
    q = MM1Queue(arrival_rate=2.0, service_rate=5.0)
    total = sum(q.state_probability(n) for n in range(2000))
    assert total == pytest.approx(1.0, abs=1e-6)


def test_mm1_rejects_unstable_queue():
    with pytest.raises(ValueError, match='unstable'):
        MM1Queue(arrival_rate=5.0, service_rate=5.0)
    with pytest.raises(ValueError, match='unstable'):
        MM1Queue(arrival_rate=6.0, service_rate=5.0)


def test_mm1_rejects_non_positive_rates():
    with pytest.raises(ValueError, match='positive'):
        MM1Queue(arrival_rate=0.0, service_rate=5.0)


def test_mm1k_state_probabilities_sum_to_one():
    # Overloaded (arrival_rate > service_rate) is fine here: finite capacity is always stable.
    q = MM1KQueue(arrival_rate=3.0, service_rate=2.0, capacity=10)
    total = sum(q.state_probability(n) for n in range(q.capacity + 1))
    assert total == pytest.approx(1.0)


def test_mm1k_rho_equals_one_uniform_distribution():
    q = MM1KQueue(arrival_rate=4.0, service_rate=4.0, capacity=5)
    for n in range(q.capacity + 1):
        assert q.state_probability(n) == pytest.approx(1.0 / (q.capacity + 1))


def test_mm1k_littles_law_with_effective_arrival_rate():
    q = MM1KQueue(arrival_rate=3.0, service_rate=2.0, capacity=8)
    lam_eff = q.effective_arrival_rate
    assert q.mean_number_in_system == pytest.approx(lam_eff * q.mean_wait_in_system)
    assert q.mean_number_in_queue == pytest.approx(lam_eff * q.mean_wait_in_queue)


def test_mm1k_effective_arrival_rate_never_exceeds_arrival_rate():
    q = MM1KQueue(arrival_rate=5.0, service_rate=2.0, capacity=3)
    assert q.effective_arrival_rate <= q.arrival_rate


def test_mm1k_approaches_mm1_for_large_capacity():
    mm1 = MM1Queue(arrival_rate=1.0, service_rate=2.0)
    mm1k = MM1KQueue(arrival_rate=1.0, service_rate=2.0, capacity=200)
    assert mm1k.mean_number_in_system == pytest.approx(mm1.mean_number_in_system, rel=1e-4)
    assert mm1k.mean_wait_in_system == pytest.approx(mm1.mean_wait_in_system, rel=1e-4)


def test_mm1k_rejects_invalid_capacity():
    with pytest.raises(ValueError, match='capacity'):
        MM1KQueue(arrival_rate=1.0, service_rate=2.0, capacity=0)


def test_mmc_single_server_matches_mm1():
    mm1 = MM1Queue(arrival_rate=2.0, service_rate=5.0)
    mmc = MMcQueue(arrival_rate=2.0, service_rate=5.0, n_servers=1)
    assert mmc.mean_number_in_system == pytest.approx(mm1.mean_number_in_system)
    assert mmc.mean_number_in_queue == pytest.approx(mm1.mean_number_in_queue)
    assert mmc.mean_wait_in_system == pytest.approx(mm1.mean_wait_in_system)
    assert mmc.mean_wait_in_queue == pytest.approx(mm1.mean_wait_in_queue)


def test_mmc_littles_law():
    q = MMcQueue(arrival_rate=8.0, service_rate=3.0, n_servers=4)
    assert q.mean_number_in_system == pytest.approx(q.arrival_rate * q.mean_wait_in_system)
    assert q.mean_number_in_queue == pytest.approx(q.arrival_rate * q.mean_wait_in_queue)


def test_mmc_more_servers_reduces_wait():
    q_slow = MMcQueue(arrival_rate=8.0, service_rate=3.0, n_servers=4)
    q_fast = MMcQueue(arrival_rate=8.0, service_rate=3.0, n_servers=8)
    assert q_fast.mean_wait_in_queue < q_slow.mean_wait_in_queue


def test_mmc_rejects_unstable_queue():
    with pytest.raises(ValueError, match='unstable'):
        MMcQueue(arrival_rate=20.0, service_rate=2.0, n_servers=4)  # 20 >= 4*2=8, unstable


def test_mmc_erlang_c_in_valid_range():
    q = MMcQueue(arrival_rate=8.0, service_rate=3.0, n_servers=4)
    assert 0.0 <= q.erlang_c <= 1.0
    assert 0.0 <= q.state_probability_empty <= 1.0
