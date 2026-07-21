"""M/M/1-family queuing models: closed-form performance metrics.

Pure analytical formulas (no discrete-event simulation) for the classic
Markovian queuing systems: `MM1Queue` (single server, infinite capacity),
`MM1KQueue` (single server, finite capacity `K`), and `MMcQueue` (`c`
servers, infinite capacity, Erlang-C). Kept intentionally small and
independent of the rest of the package.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MM1Queue:
    """M/M/1 queue: Poisson arrivals, exponential service, a single server, infinite capacity."""

    arrival_rate: float
    service_rate: float

    def __post_init__(self) -> None:
        """Validate the queue is well-formed and stable.

        Raises:
            ValueError: If rates are non-positive, or the queue is unstable
                (`arrival_rate >= service_rate`).
        """
        if self.arrival_rate <= 0 or self.service_rate <= 0:
            msg = 'arrival_rate and service_rate must be positive.'
            raise ValueError(msg)
        if self.arrival_rate >= self.service_rate:
            msg = 'unstable queue: arrival_rate must be < service_rate.'
            raise ValueError(msg)

    @property
    def utilization(self) -> float:
        """Server utilization `rho = lambda / mu`."""
        return self.arrival_rate / self.service_rate

    @property
    def mean_number_in_system(self) -> float:
        """`L = rho / (1 - rho)`."""
        rho = self.utilization
        return rho / (1 - rho)

    @property
    def mean_number_in_queue(self) -> float:
        """`Lq = rho**2 / (1 - rho)`."""
        rho = self.utilization
        return rho**2 / (1 - rho)

    @property
    def mean_wait_in_system(self) -> float:
        """`W = 1 / (mu - lambda)`."""
        return 1.0 / (self.service_rate - self.arrival_rate)

    @property
    def mean_wait_in_queue(self) -> float:
        """`Wq = rho / (mu - lambda)`."""
        return self.utilization / (self.service_rate - self.arrival_rate)

    def state_probability(self, n: int) -> float:
        """`P(n in system) = (1 - rho) * rho**n`.

        Raises:
            ValueError: If `n < 0`.
        """
        if n < 0:
            msg = 'n must be non-negative.'
            raise ValueError(msg)
        rho = self.utilization
        return (1 - rho) * rho**n


@dataclass(frozen=True)
class MM1KQueue:
    """M/M/1/K queue: single server, finite capacity `K` (includes the customer in service)."""

    arrival_rate: float
    service_rate: float
    capacity: int

    def __post_init__(self) -> None:
        """Validate the queue is well-formed.

        Raises:
            ValueError: If rates are non-positive or `capacity < 1`.
        """
        if self.arrival_rate <= 0 or self.service_rate <= 0:
            msg = 'arrival_rate and service_rate must be positive.'
            raise ValueError(msg)
        if self.capacity < 1:
            msg = 'capacity must be >= 1.'
            raise ValueError(msg)

    @property
    def utilization(self) -> float:
        """Traffic intensity `rho = lambda / mu` (may be >= 1: finite capacity is always stable)."""
        return self.arrival_rate / self.service_rate

    def state_probability(self, n: int) -> float:
        """`P(n in system)` for `n` in `[0, capacity]`.

        Raises:
            ValueError: If `n` is out of `[0, capacity]`.
        """
        if not 0 <= n <= self.capacity:
            msg = f'n must be in [0, {self.capacity}].'
            raise ValueError(msg)
        rho = self.utilization
        k = self.capacity
        if math.isclose(rho, 1.0):
            return 1.0 / (k + 1)
        p0 = (1 - rho) / (1 - rho**(k + 1))
        return p0 * rho**n

    @property
    def blocking_probability(self) -> float:
        """`P(K in system)`: probability an arriving customer is blocked (system full)."""
        return self.state_probability(self.capacity)

    @property
    def effective_arrival_rate(self) -> float:
        """`lambda_eff = lambda * (1 - blocking_probability)`: throughput actually admitted."""
        return self.arrival_rate * (1 - self.blocking_probability)

    @property
    def mean_number_in_system(self) -> float:
        """`L = sum_{n=0}^{K} n * P(n)`."""
        return sum(n * self.state_probability(n) for n in range(self.capacity + 1))

    @property
    def mean_number_in_queue(self) -> float:
        """`Lq = L - (1 - P(0))` (mean number in system minus mean number in service)."""
        return self.mean_number_in_system - (1 - self.state_probability(0))

    @property
    def mean_wait_in_system(self) -> float:
        """`W = L / lambda_eff` (Little's law, using the *effective* admitted arrival rate)."""
        return self.mean_number_in_system / self.effective_arrival_rate

    @property
    def mean_wait_in_queue(self) -> float:
        """`Wq = Lq / lambda_eff`."""
        return self.mean_number_in_queue / self.effective_arrival_rate


@dataclass(frozen=True)
class MMcQueue:
    """M/M/c queue: `c` identical servers, infinite capacity (Erlang-C model)."""

    arrival_rate: float
    service_rate: float
    n_servers: int

    def __post_init__(self) -> None:
        """Validate the queue is well-formed and stable.

        Raises:
            ValueError: If rates/`n_servers` are non-positive, or the queue
                is unstable (`arrival_rate >= n_servers * service_rate`).
        """
        if self.arrival_rate <= 0 or self.service_rate <= 0:
            msg = 'arrival_rate and service_rate must be positive.'
            raise ValueError(msg)
        if self.n_servers < 1:
            msg = 'n_servers must be >= 1.'
            raise ValueError(msg)
        if self.arrival_rate >= self.n_servers * self.service_rate:
            msg = 'unstable queue: arrival_rate must be < n_servers * service_rate.'
            raise ValueError(msg)

    @property
    def offered_load(self) -> float:
        """Offered load `a = lambda / mu` (in Erlangs; average number of customers in service)."""
        return self.arrival_rate / self.service_rate

    @property
    def utilization(self) -> float:
        """Per-server utilization `rho = lambda / (c * mu)`."""
        return self.arrival_rate / (self.n_servers * self.service_rate)

    @property
    def state_probability_empty(self) -> float:
        """`P0`: probability the system is empty."""
        a = self.offered_load
        c = self.n_servers
        terms = sum(a**n / math.factorial(n) for n in range(c))
        last_term = (a**c / math.factorial(c)) / (1 - self.utilization)
        return 1.0 / (terms + last_term)

    @property
    def erlang_c(self) -> float:
        """Erlang-C formula: probability an arriving customer must wait (all servers busy)."""
        a = self.offered_load
        c = self.n_servers
        rho = self.utilization
        return (a**c / (math.factorial(c) * (1 - rho))) * self.state_probability_empty

    @property
    def mean_number_in_queue(self) -> float:
        """`Lq = P_wait * rho / (1 - rho)`."""
        rho = self.utilization
        return self.erlang_c * rho / (1 - rho)

    @property
    def mean_wait_in_queue(self) -> float:
        """`Wq = Lq / lambda`."""
        return self.mean_number_in_queue / self.arrival_rate

    @property
    def mean_wait_in_system(self) -> float:
        """`W = Wq + 1/mu`."""
        return self.mean_wait_in_queue + 1.0 / self.service_rate

    @property
    def mean_number_in_system(self) -> float:
        """`L = Lq + a` (queue plus the average number in service)."""
        return self.mean_number_in_queue + self.offered_load


__all__ = ['MM1KQueue', 'MM1Queue', 'MMcQueue']
