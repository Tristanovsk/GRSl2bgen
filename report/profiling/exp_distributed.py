"""Process.run() on a local dask.distributed cluster: n_workers processes x threads each.

Usage: python report/profiling/exp_distributed.py <L2A product> <scratch output .nc> <n_workers> <threads_per_worker>
"""
import sys, time, logging, warnings
warnings.filterwarnings('ignore'); logging.getLogger().setLevel(logging.WARNING)
import GRSl2bgen
if __name__ == '__main__':
    l2a, out, nw, tpw = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    t = time.perf_counter()
    GRSl2bgen.Process(l2a, out, n_workers=nw, threads_per_worker=tpw).run()
    print(f'workers={nw} threads_per_worker={tpw} wall={time.perf_counter() - t:.1f}s', flush=True)
