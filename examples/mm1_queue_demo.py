"""M/M/1 vs. M/M/2 queuing performance comparison.

Demonstrates commpy.MM1Queue / MMcQueue and Little's law.
"""

from commpy import MM1Queue, MMcQueue


def main() -> None:
    """Run the M/M/1 vs M/M/2 queuing comparison demo."""
    arrival_rate = 8.0  # customers/hour
    service_rate = 5.0  # customers/hour, per server

    single = MM1Queue(arrival_rate=arrival_rate, service_rate=service_rate * 2)
    two_servers = MMcQueue(arrival_rate=arrival_rate, service_rate=service_rate, n_servers=2)

    print(f'Arrival rate: {arrival_rate}/hour')
    print(f'\nOne fast server (rate={service_rate * 2}/hour):')
    print(f'  utilization        = {single.utilization:.2f}')
    print(f'  mean wait in queue  = {single.mean_wait_in_queue * 60:.1f} min')
    print(f'  mean number waiting = {single.mean_number_in_queue:.2f}')

    print(f'\nTwo servers (rate={service_rate}/hour each, same total capacity):')
    print(f'  utilization        = {two_servers.utilization:.2f}')
    print(f'  P(must wait)        = {two_servers.erlang_c:.2f}')
    print(f'  mean wait in queue  = {two_servers.mean_wait_in_queue * 60:.1f} min')
    print(f'  mean number waiting = {two_servers.mean_number_in_queue:.2f}')

    # Little's law: L = lambda * W, verified for both models.
    assert abs(single.mean_number_in_queue - arrival_rate * single.mean_wait_in_queue) < 1e-9
    two_lq_check = arrival_rate * two_servers.mean_wait_in_queue
    assert abs(two_servers.mean_number_in_queue - two_lq_check) < 1e-9
    print("\nOK: Little's law (L = lambda * W) holds for both models.")


if __name__ == '__main__':
    main()
