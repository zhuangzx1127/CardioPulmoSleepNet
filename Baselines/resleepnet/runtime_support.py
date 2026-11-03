"""Startup environment and TF 2.9 strategy shutdown compatibility."""
import atexit
import os
import sys
from multiprocessing.pool import ThreadPool


def configure_runtime():
    """Run before TensorFlow/NumPy imports so libgomp receives a valid value."""
    raw = os.environ.get('OMP_NUM_THREADS')
    if raw is None:
        os.environ['OMP_NUM_THREADS'] = '1'
        return
    parts = [part.strip() for part in raw.split(',')]
    if all(part.isascii() and part.isdigit() and int(part) > 0 for part in parts):
        os.environ['OMP_NUM_THREADS'] = ','.join(str(int(part)) for part in parts)
    else:
        print(f'Invalid OMP_NUM_THREADS={raw!r}; using 1.', file=sys.stderr, flush=True)
        os.environ['OMP_NUM_THREADS'] = '1'


def register_strategy_cleanup(strategy):
    """Close only thread pools owned by this strategy before interpreter teardown.

    TF 2.9 CollectiveAllReduce keeps a persistent multiprocessing ThreadPool;
    letting its destructor run after module globals are cleared can emit the
    Pool.__del__/queues.put NoneType.dumps traceback. Newer versions without
    these pools need no cleanup. Do not alter reduction or optimizer behavior.
    """
    pools = []
    extended = strategy.extended
    for name in ('_collective_ops', '_cross_device_ops', '_inferred_cross_device_ops',
                 '_host_cross_device_ops'):
        pool = getattr(getattr(extended, name, None), '_pool', None)
        if isinstance(pool, ThreadPool) and all(pool is not previous for previous in pools):
            pools.append(pool)
    closed = False

    def close():
        nonlocal closed
        if closed:
            return
        for pool in pools:
            pool.close()
            pool.join()
        closed = True

    atexit.register(close)
    return close
