"""Write the full-tile L2B (in memory) with several codecs: time and size.

Usage: python report/profiling/bench_write.py <L2B .nc> <scratch path prefix>
"""
import sys, time, os, shutil, warnings, logging
warnings.filterwarnings('ignore'); logging.getLogger().setLevel(logging.WARNING)
import xarray as xr, dask
src, tmp = sys.argv[1:3]
ds = xr.open_dataset(src).load()
base = {v: {k: ds[v].encoding[k] for k in ('dtype', 'scale_factor', 'add_offset', '_FillValue') if k in ds[v].encoding} for v in ds.data_vars}
def size(p): return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) if os.path.isdir(p) else os.path.getsize(p)
for name, extra in [('nc zlib5', dict(zlib=True, complevel=5)), ('nc zlib5+shuffle', dict(zlib=True, complevel=5, shuffle=True)),
                    ('nc zlib1+shuffle', dict(zlib=True, complevel=1, shuffle=True)), ('nc zlib1', dict(zlib=True, complevel=1))]:
    enc = {v: {**e, **extra, 'chunksizes': (1024, 1024) if ds[v].ndim == 2 else (ds[v].shape[0], 1024, 1024)} for v, e in base.items()}
    out = tmp + '.nc'; t = time.perf_counter(); ds.to_netcdf(out, encoding=enc); dt = time.perf_counter() - t
    print(f'{name:18s} {dt:6.1f} s {size(out)/1e6:7.1f} MB', flush=True); os.remove(out)
dsc = ds.chunk({'y': 1024, 'x': 1024})
with dask.config.set(scheduler='threads', num_workers=8):
    out = tmp + '.zarr'; t = time.perf_counter(); dsc.to_zarr(out, encoding=base, mode='w', zarr_format=3); dt = time.perf_counter() - t
print(f'{"zarr (default zstd)":18s} {dt:6.1f} s {size(out)/1e6:7.1f} MB', flush=True); shutil.rmtree(out)
