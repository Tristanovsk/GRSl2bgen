"""Profile the GRSl2bgen chain on one L2A product.

Part 1: baseline Process (execute + export), dask Profiler aggregated by task prefix.
Part 2: per-module cost with the masked input and the OWT results persisted.
Part 3: per-output-variable cost (each variable computed alone, inputs persisted).

Usage::

    python report/profiling/profile_l2bgen.py <L2A product> <scratch output .nc> [parts]

`parts` is any combination of 1, 2, 3 (default 123). Set the environment variable
CHUNK (e.g. CHUNK=1024) to rechunk the input spatially for parts 2-3. The results are
printed and saved to <scratch output .nc>.profile.json.
"""
import json
import logging
import os
import sys
import time
import warnings
from collections import defaultdict

import dask
import numpy as np
import xarray as xr
from dask.diagnostics import Profiler, ResourceProfiler
from dask.utils import key_split

import GRSl2bgen
from GRSl2bgen import Chl, Spm, Cdom, Transparency, OWT_process, Product, L2bProduct
from GRSl2bgen.mask import QC_L2A

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.WARNING)

L2A = sys.argv[1]
OUT = sys.argv[2]
parts = sys.argv[3] if len(sys.argv) > 3 else '123'
results = {}


def tic():
    return time.perf_counter()


def by_prefix(prof):
    """Summed task duration (s) per dask task prefix."""
    agg = defaultdict(float)
    cnt = defaultdict(int)
    for r in prof.results:
        agg[key_split(r.key)] += r.end_time - r.start_time
        cnt[key_split(r.key)] += 1
    return sorted(((k, round(v, 1), cnt[k]) for k, v in agg.items()), key=lambda x: -x[1])


def report(name, value):
    results[name] = value
    print(name, json.dumps(value) if not isinstance(value, (int, float)) else value, flush=True)


# ---------------------------------------------------------------- part 1
if '1' in parts:
    proc = GRSl2bgen.Process(L2A, l2b_path=OUT)
    t = tic(); proc.execute(); report('1_execute_graph_s', round(tic() - t, 1))
    l2b = proc.l2b
    report('1_n_tasks', len(dict(l2b.l2b_prod.__dask_graph__())))
    with Profiler() as p1, ResourceProfiler(dt=1) as r1:
        t = tic(); enc = l2b._packing_encoding(); report('1_pass1_minmax_s', round(tic() - t, 1))
    report('1_pass1_by_prefix', by_prefix(p1)[:25])
    report('1_pass1_peak_mem_GB', round(max(r.mem for r in r1.results) / 1e3, 2))
    report('1_pass1_mean_cpu_pct', round(np.mean([r.cpu for r in r1.results]), 0))
    with Profiler() as p2, ResourceProfiler(dt=1) as r2:
        t = tic(); l2b.export_to_netcdf(OUT); report('1_pass2_write_s', round(tic() - t, 1))
    report('1_pass2_by_prefix', by_prefix(p2)[:25])
    report('1_pass2_peak_mem_GB', round(max(r.mem for r in r2.results) / 1e3, 2))
    report('1_pass2_mean_cpu_pct', round(np.mean([r.cpu for r in r2.results]), 0))
    proc.l2a.close()
    os.remove(OUT)

# ---------------------------------------------------------------- parts 2-3
if '2' in parts or '3' in parts:
    prod = Product(L2A)
    if os.environ.get("CHUNK"):
        c = int(os.environ["CHUNK"]); prod.raster = prod.raster.chunk({"y": c, "x": c})
        report("chunk", c)
    t = tic()
    raster = prod.raster.where(prod.raster['mask'] == 0).persist()
    dask.wait(raster) if hasattr(dask, 'wait') else None
    raster['Rrs'].data.sum().compute()  # make sure the persist is materialised
    report('2_read_mask_persist_s', round(tic() - t, 1))

    def timed(name, obj):
        t = tic(); out = dask.persist(obj)[0]
        dask.compute(*[v.data for v in out.data_vars.values() if v.chunks is not None])
        report(name, round(tic() - t, 1))
        return out

    t = tic(); qc = QC_L2A(raster); qc.process()
    timed('2_qc_flags_s', xr.Dataset(qc.flags))

    owt = OWT_process(raster)
    t = tic()
    owt.execute()  # graph only
    report('2_owt_execute_call_s', round(tic() - t, 1))
    xowt = {}
    for name in ('xowt_spyrakos2018', 'xowt_bi2024', 'xowt_Ta2025', 'xowt_Cordeiro2022SPM'):
        xowt[name] = timed(f'2_owt_{name[5:]}_s', getattr(owt, name))

    t = tic()
    spm = Spm(raster, xowt_prod=xowt['xowt_Cordeiro2022SPM']); spm.process()
    report('2_spm_process_call_s', round(tic() - t, 1))
    t = tic()
    chl = Chl(raster, xowt_prod=xowt['xowt_spyrakos2018']); chl.process()
    report('2_chl_process_call_s', round(tic() - t, 1))
    t = tic()
    trans = Transparency(raster); trans.process()
    report('2_transparency_process_call_s', round(tic() - t, 1))
    spm_out = timed('2_spm_all_s', spm.output)
    chl_out = timed('2_chl_all_s', chl.output)
    t = tic()
    cdom = Cdom(raster, chl=chl_out.get('Chla_OWTblend_Ta2025')); cdom.process()
    report('2_cdom_process_call_s', round(tic() - t, 1))
    cdom_out = timed('2_cdom_all_s', cdom.output)
    trans_out = timed('2_transparency_all_s', trans.output)

    if '3' in parts:
        per_var = {}
        eager = []
        for mod, ds in (('spm', spm.output), ('chl', chl.output), ('cdom', cdom.output),
                        ('transparency', trans.output)):
            for v in ds.data_vars:
                if ds[v].chunks is None:
                    eager.append(f'{mod}:{v}')
                    continue
                t = tic(); ds[v].data.compute(); per_var[f'{mod}:{v}'] = round(tic() - t, 1)
                print('  ', mod, v, per_var[f'{mod}:{v}'], flush=True)
        report('3_per_variable_s', dict(sorted(per_var.items(), key=lambda x: -x[1])))
        report('3_eager_numpy_variables', eager)

    # export cost alone: everything persisted, then pass 1 + pass 2
    l2b = L2bProduct(prod, [xr.merge([xowt[k] for k in xowt], compat='no_conflicts', join='outer'),
                            chl_out, spm_out, cdom_out, trans_out],
                     flags={**qc.flags, **trans.flags}, flag_descriptions=qc.flag_descriptions)
    l2b.l2b_prod = l2b.l2b_prod.persist()
    t = tic(); l2b.export_to_netcdf(OUT); report('2_export_only_persisted_s', round(tic() - t, 1))
    os.remove(OUT)
    prod.close()

json.dump(results, open(OUT + '.profile.json', 'w'), indent=1)
