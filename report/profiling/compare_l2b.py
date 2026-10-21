"""Compare an L2B product with the reference: max difference in packing steps per variable.

Usage: python report/profiling/compare_l2b.py <new L2B .nc> <reference L2B .nc>
"""
import sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, xarray as xr
new, ref = xr.open_dataset(sys.argv[1]), xr.open_dataset(sys.argv[2])
worst = 0.
for v in ref.data_vars:
    a, b = new[v].values, ref[v].values
    if not np.array_equal(np.isnan(a), np.isnan(b)) if a.dtype.kind == 'f' else False:
        print(f'{v}: NaN pattern differs ({int((np.isnan(a) != np.isnan(b)).sum())} px)'); worst = np.inf; continue
    step = float(ref[v].encoding.get('scale_factor', 0)) or 1.
    d = np.nanmax(np.abs(a.astype(float) - b.astype(float))) / step if a.size else 0.
    worst = max(worst, d)
    if d > 1: print(f'{v}: max diff {d:.2f} packing steps')
print(f'max difference over all variables: {worst:.2f} packing steps')
