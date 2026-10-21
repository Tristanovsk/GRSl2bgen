"""One full-tile Process.run() with input chunks C x C and N dask threads: wall time.

Usage: python report/profiling/bench_chunks.py <L2A product> <scratch output .nc> <C> <N>
(peak memory: run it with /usr/bin/time -f %M). The measurements of the report were made
before `Product(spatial_chunk=...)` existed, by rechunking the input after opening; this
version uses the option, with the same effect.
"""
import sys, time, logging, warnings
warnings.filterwarnings('ignore'); logging.getLogger().setLevel(logging.WARNING)
import GRSl2bgen
import GRSl2bgen.process
from GRSl2bgen.product import Product

l2a, out, c, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
GRSl2bgen.process.Product = lambda obj, level=0: Product(obj, level=level, spatial_chunk=c)
t = time.perf_counter()
GRSl2bgen.Process(l2a, out, n_threads=n).run()
print(f'chunks={c} threads={n} wall={time.perf_counter() - t:.1f}s', flush=True)
