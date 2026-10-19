"""Full-tile Process.run() with experimental patches: nogil (numba kernels without the GIL),
f32 (Rrs and sza decoded to float32). Usage: exp_opt.py <L2A> <out.nc> <patches, e.g. nogil,f32 or none>"""
import sys, time, logging, warnings
warnings.filterwarnings('ignore'); logging.getLogger().setLevel(logging.WARNING)
import numpy as np
from numba import njit
import GRSl2bgen, GRSl2bgen.sam as sam, GRSl2bgen.euclidean as euc
import GRSl2bgen.process
from GRSl2bgen.product import Product
l2a, out, patches = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
if 'nogil' in patches:
    sam._sam_serial = njit(nogil=True)(sam._sam_core)
    euc._euclid_serial = njit(nogil=True)(euc._euclid_core)
if 'f32' in patches:
    def product_f32(obj, level=0):
        p = Product(obj, level=level)
        for v in ('Rrs', 'sza'):
            p.raster[v] = p.raster[v].astype(np.float32)
        return p
    GRSl2bgen.process.Product = product_f32
t = time.perf_counter(); GRSl2bgen.Process(l2a, out).run()
print(f'patches={patches} wall={time.perf_counter() - t:.1f}s', flush=True)
