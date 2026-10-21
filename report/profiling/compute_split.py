"""Compute cost of each part of the chain from the L2A, current defaults (1024 chunks, 8 threads).

Usage: python report/profiling/compute_split.py <L2A product>
"""
import sys, time, logging, warnings
warnings.filterwarnings('ignore'); logging.getLogger().setLevel(logging.WARNING)
import dask, xarray as xr
from dask.diagnostics import ResourceProfiler
from GRSl2bgen import Product, Chl, Spm, Cdom, Transparency, OWT_process
from GRSl2bgen.mask import QC_L2A

def run(name, *arrays):
    with ResourceProfiler(dt=0.25) as rp:
        t = time.perf_counter(); dask.compute(*arrays); dt = time.perf_counter() - t
    cpu = sum(r.cpu for r in rp.results) / max(len(rp.results), 1)
    print(f'{name:45s} {dt:6.1f} s   CPU {cpu:4.0f} %', flush=True)

with dask.config.set(scheduler='threads', num_workers=8):
    prod = Product(sys.argv[1])
    r = prod.raster.where(prod.raster['mask'] == 0)
    run('read + decode + mask (Rrs, sza)', r['Rrs'].sum(), r['sza'].sum())
    owt = OWT_process(r); owt.execute()
    run('OWT, 4 databases (incl. read)', *[owt.output[v].data for v in owt.output.data_vars])
    spm = Spm(r, xowt_prod=owt.xowt_Cordeiro2022SPM); spm.process()
    run('SPM, 4 outputs (incl. read + Cordeiro OWT)', *[spm.output[v].data for v in spm.output.data_vars])
    run('  SPM_OWTblend_Ta2025 only', spm.output['SPM_OWTblend_Ta2025'].data)
    chl = Chl(r, xowt_prod=owt.xowt_spyrakos2018); chl.process()
    run('Chl-a, 4 outputs (incl. read + Spyrakos OWT)', *[chl.output[v].data for v in chl.output.data_vars])
    cd = Cdom(r, chl=chl.output['Chla_OWTblend_Ta2025']); cd.process()
    run('CDOM, 3 outputs (incl. read + Chl-a)', *[cd.output[v].data for v in cd.output.data_vars])
    tr = Transparency(r); tr.process()
    run('Transparency, 2 outputs (incl. read)', *[tr.output[v].data for v in tr.output.data_vars])
    qc = QC_L2A(r); qc.process()
    run('QC flags (incl. read)', *[f.data for f in qc.flags.values()])
