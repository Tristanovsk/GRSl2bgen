"""Write time and size of a 2048 x 2048 subset of an L2B NetCDF at zlib levels 5, 1 and 0.

Usage: python report/profiling/bench_complevel.py <L2B .nc> <scratch output .nc>
"""
import sys, time, os, warnings, logging
warnings.filterwarnings('ignore'); logging.getLogger().setLevel(logging.WARNING)
import xarray as xr
from GRSl2bgen.output import L2bProduct
src, out = sys.argv[1:3]
ds = xr.open_dataset(src).isel(x=slice(1500, 3548), y=slice(1500, 3548)).load()   # 2048 x 2048, in memory
for lev in (5, 1, 0):
    enc = {}
    for v in ds.data_vars:
        e = {k: ds[v].encoding[k] for k in ('dtype', 'scale_factor', 'add_offset', '_FillValue') if k in ds[v].encoding}
        if lev: e.update(zlib=True, complevel=lev)
        enc[v] = e
    t = time.perf_counter(); ds.to_netcdf(out, encoding=enc); dt = time.perf_counter() - t
    print(f'complevel {lev}: {dt:.1f} s, {os.path.getsize(out)/1e6:.1f} MB', flush=True); os.remove(out)
