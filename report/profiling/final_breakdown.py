"""Time of each stage of Process with the default settings (persist_output=True, 8 threads).

Usage: python report/profiling/final_breakdown.py <L2A product> <scratch output .nc>
"""
import sys, time, logging, warnings, os
warnings.filterwarnings('ignore'); logging.getLogger().setLevel(logging.WARNING)
import dask
from dask.diagnostics import ResourceProfiler
import GRSl2bgen
l2a, out = sys.argv[1:3]
with dask.config.set(scheduler='threads', num_workers=8):
    p = GRSl2bgen.Process(l2a, out)
    t = time.perf_counter(); p.execute(); print(f'execute (graph only): {time.perf_counter() - t:.1f} s')
    l2b = p.l2b
    print(f'graph tasks: {len(dict(l2b.l2b_prod.__dask_graph__()))}')
    with ResourceProfiler(dt=0.5) as rp:
        t = time.perf_counter(); l2b.l2b_prod = l2b.l2b_prod.persist(); print(f'compute L2B in memory: {time.perf_counter() - t:.1f} s')
    print(f'  mean CPU during compute: {sum(r.cpu for r in rp.results) / len(rp.results):.0f} %')
    t = time.perf_counter(); enc = l2b._packing_encoding(); print(f'packing min/max (in memory): {time.perf_counter() - t:.1f} s')
    t = time.perf_counter(); l2b.export_to_netcdf(out); print(f'NetCDF write incl. min/max: {time.perf_counter() - t:.1f} s')
    p.l2a.close(); os.remove(out)
