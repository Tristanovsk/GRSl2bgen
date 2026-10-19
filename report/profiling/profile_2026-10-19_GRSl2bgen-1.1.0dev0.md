# GRSl2bgen profiling — full Sentinel-2 tile

| | |
|---|---|
| Date | 2026-10-19 |
| GRSl2bgen version | 1.1.0dev0 (git `0915e4f`, branch `develop`) |
| Input L2A | `S2A_MSIL2AGRS_20210211T104151_N0500_R008_T31TFJ_20230608T124831_V3.1.0` (GRS Zarr pyramid, level 0) |
| Image size | 5490 x 5490 pixels at 20 m, 11 bands, stored chunks 256 x 256 |
| Output | NetCDF (default `Process` settings: threaded dask scheduler, `persist_input=False`) |
| Machine | Intel Xeon Silver 4210 @ 2.20 GHz, 40 logical CPUs, 62 GB RAM |
| Environment | Python 3.12.4, dask 2026.3.0, xarray 2026.9.0, numpy 2.3.5, numba 0.64.0, zarr 3.4.1, netCDF4 1.7.2 |
| Scripts | [`profile_l2bgen.py`](profile_l2bgen.py), [`bench_complevel.py`](bench_complevel.py), [`bench_chunks.py`](bench_chunks.py), [`final_breakdown.py`](final_breakdown.py), [`compare_l2b.py`](compare_l2b.py) |
| Results after optimization | section 7, GRSl2bgen git `3370f88` |

## Summary

- One full tile takes **310 s** (peak memory 6.4 GB) with the current code.
- The machine is barely used: about **2 of the 40 cores** (mean CPU ≈ 190 %). The input is
  read with its stored 256 x 256 chunks, so the dask graph has **195,000 small tasks**,
  dominated by Python overhead and GIL contention.
- The processing chain is **evaluated three times**: partly in `Process.execute()`
  (eager `.compute()` calls), then once for the min/max of the int16 packing, then once
  more to write the file.
- Rechunking the input to 1024 x 1024 pixels alone brings one tile to **119 s
  (2.6 x faster)**, with outputs identical to the reference L2B (within the packing
  precision).
- After the phase 1 optimizations (section 7): **93 s per tile (3.3 x faster), peak
  memory 5.9 GB** (was 6.4 GB), outputs identical to the reference within one packing
  step.

## 1. Baseline: `Process.execute()` + `write_output()`

Measured with the dask `Profiler` and `ResourceProfiler` (the profilers add overhead: the
sum is 420 s instead of 310 s without them).

| Step | Wall time | Peak memory | Mean CPU | What is computed |
|---|---|---|---|---|
| `execute()` | 98.7 s | | | graph construction, plus the eager `.compute()` of Chl-a (`chlorophyll_a.py:133`, including the Spyrakos2018 OWT) and of transparency (`transparency.py:104`) |
| export, pass 1 (`_packing_encoding`) | 94.3 s | 10.4 GB | 192 % | min/max of every packed variable: full evaluation of the chain |
| export, pass 2 (`to_netcdf`) | 227.4 s | 12.1 GB | 186 % | full evaluation of the chain again, int16 packing, zlib compression (level 5), write |

Graph size: 194,988 tasks.

Summed task time by dask task type (seconds of task time, all threads):

| Task type | Pass 1 | Pass 2 | Main source |
|---|---|---|---|
| `store-map` (NetCDF write + compression) | — | 2991 | `to_netcdf`, serialized by the NetCDF library lock |
| `where` | 1642 | 1932 | water mask, validity ranges, OWT masks, blending |
| `getitem` | 392 | 578 | band selections (`Rrs.sel(wl=...)`) |
| `sum-sum-aggregate` | 234 | 382 | OWT blending (sum over `Nclasses`) |
| `truediv` | 165 | 239 | band ratios, blending weights |
| `chunk_min` / `chunk_max` | 149 / 148 | 146 / 145 | packing ranges |
| `rechunk-*` | 176 | 217 | `wl` rechunk and dask/zarr chunk mismatches |

Task times exceed the wall time because they are summed over the threads; with a mean
CPU of ~190 %, most of that time is threads waiting on each other.

## 2. Cost per module (masked input persisted in memory)

Each module is computed once, with its inputs (masked L2A raster, OWT results) already
in memory, so that each row shows the cost of the module itself.

| Module | Chunks 256 x 256 (stored) | Chunks 1024 x 1024 | Speed-up |
|---|---|---|---|
| Read L2A + water mask (`mask == 0`) | 30.7 s | 25.4 s | 1.2 x |
| QC of the L2A (SWIR module flags) | 1.6 s | 0.3 s | 5 x |
| OWT Spyrakos2018 (SAM, 3 classes) | 5.2 s | 4.5 s | 1.2 x |
| OWT Bi2024 (SAM) | 7.9 s | 1.7 s | 4.6 x |
| OWT Tarasenko2025 (SAM) | 2.7 s | 2.1 s | 1.3 x |
| OWT Cordeiro2022SPM (Euclidean, 2 classes) | 2.3 s | 2.2 s | 1.0 x |
| SPM (4 outputs) | 46.2 s | 5.1 s | 9 x |
| Chl-a (4 outputs, eager in `process()`) | 39.4 s | 7.8 s | 5 x |
| CDOM (3 outputs) | 32.2 s | 11.9 s | 2.7 x |
| Transparency (2 outputs, eager in `process()`) | 27.7 s | 12.8 s | 2.2 x |
| NetCDF export of the persisted L2B (2 passes) | 89.6 s | 54.2 s | 1.7 x |
| **Total** | **285.5 s** | **128.0 s** | **2.2 x** |

The OWT classifications (numba kernels) are cheap; the cost is in the band maths of the
retrieval modules and in the export.

## 3. Cost per output variable (256 x 256 chunks, inputs persisted)

Each variable computed alone (its shared intermediates are recomputed, so the values do
not add up to the module totals):

| Variable | Time |
|---|---|
| `SPM_OWTblend_Ta2025` | 31.9 s |
| `adg443_QAAv6` | 17.8 s |
| `adg443_QAAv6Chl` | 14.0 s |
| `SPM_obs2co` | 5.1 s |
| `TURB_dogliotti` | 5.0 s |
| `acdom_B15` | 2.8 s |
| `SPM_nechad` | 2.4 s |

`Chla_OC2nasa`, `Chla_M09B`, `Chla_NIRB`, `Chla_OWTblend_Ta2025`, `Kd_par` and
`Zsd_Lee15` are already numpy arrays after `process()` (eager `.compute()`); their cost
is in the module rows above (Chl-a 39.4 s, transparency 27.7 s).

## 4. NetCDF compression

2048 x 2048 subset of the L2B product, all data in memory:

| zlib level | Write time | File size |
|---|---|---|
| 5 (current) | 3.9 s | 3.8 MB |
| 1 | 2.5 s | 5.2 MB |
| 0 (none) | 1.4 s | 247.6 MB |

## 5. End-to-end check of the chunk size

Unmodified `Process.run()`, with only the input rechunked to 1024 x 1024 after opening
(`Product._open_zarr` patched from outside):

| Chunks | Wall time | Peak memory | Outputs |
|---|---|---|---|
| 256 x 256 (current) | 310 s | 6.4 GB | reference |
| 1024 x 1024 | 119 s | 9.6 GB | all variables equal to the reference within 2 packing steps |

The peak memory of this check was first reported as 17.8 GB: that measurement included
the comparison script, which loads both full L2B products. Measured alone (section 7):
9.6 GB with 40 threads.

## 6. Optimization opportunities

By expected gain:

1. **Rechunk the Zarr input to larger spatial chunks** in `Product` (one line).
   Measured: 310 s → 119 s per tile, same outputs (see section 7 for the memory).
2. **Evaluate the chain once instead of three times.**
   - (a) Fixed packing ranges per variable (from the existing `range` attributes): no
     min/max pass at all, and the same `scale_factor`/`add_offset` in every product, so
     products can be compared on their raw values. Changes the packing of the products.
   - (b) Or persist the L2B before export (`export(persist=True)` exists already), for a
     few more GB of RAM.
   - In both cases, remove the eager `.compute()` in `chlorophyll_a.py:133` and
     `transparency.py:104`, which today act as a cache between the two export passes.
3. **Chl-a blending on a 4-D array** (`chlorophyll_a.py:226`):
   `algo(self.Rrs.where(mask))` broadcasts the 11-band cube over the 3 best classes
   (`wl x Nclasses x y x x`, ~8 GB in float64 for a full tile) before each of the 4
   algorithm groups. The algorithms are per pixel, so `algo(self.Rrs).where(mask)`
   gives the same result on a 2-D band (SPM blending already does this).
4. **NetCDF writing** is the largest remaining cost (54 s with everything in memory): the
   NetCDF library writes and compresses one variable at a time. zlib level 1 instead of 5
   writes ~35 % faster for ~37 % larger files; Zarr output compresses in parallel.
5. **QAA v6 is computed three times** (`cdom.py:197` and `cdom.py:266` with the same
   settings, `transparency.py:186`): sharing one result saves a few seconds. *Not
   needed after item 2 (section 7): in a single dask graph the identical QAA v6
   tasks are merged automatically.*

Items 1, 2b and 3 keep the outputs unchanged and can be checked with the test suite
(`tests/test_process.py`); item 2a changes the packing of the products.

## 7. Phase 1 optimizations: results

Same tile, same machine, `Process.run()` with the default settings after each change.
Each output is compared with the reference L2B produced by the original code
(`compare_l2b.py`: maximum difference in int16 packing steps).

### Chunk size and number of threads

Input rechunked from outside the code (`bench_chunks.py`):

| Chunks | Threads | Wall time | Peak memory |
|---|---|---|---|
| 512 x 512 | 8 | 155.8 s | 5.5 GB |
| 512 x 512 | 16 | 151.7 s | 5.7 GB |
| 512 x 512 | 40 | 152.0 s | 6.7 GB |
| 1024 x 1024 | 8 | 121.2 s | 5.7 GB |
| 1024 x 1024 | 16 | 119.0 s | 6.1 GB |
| 1024 x 1024 | 40 | 118.4 s | 9.6 GB |

More threads hardly change the run time but increase the memory: 1024 x 1024 chunks and
8 threads were chosen as defaults.

### Step by step

| Step | Commit | Change | Wall time | Peak memory | Difference to reference |
|---|---|---|---|---|---|
| 0 | `0915e4f` | original code | 310 s | 6.4 GB | — |
| 1 | `12b66c5` | input rechunked to 1024 x 1024 (`Product(spatial_chunk=1024)`), 8 dask threads (`Process(n_threads=8)`) | 120.1 s | 5.6 GB | 0 steps |
| 2 | `956d949` | Chl-a OWT blending masks the per-pixel estimate, not the broadcast `Rrs` | 118.0 s | 5.6 GB | 0 steps |
| 3 | `77d92cc` | chain evaluated once: no eager `.compute()` in Chl-a and transparency, L2B persisted before export (`Process(persist_output=True)`) | 93.0 s | 6.9 GB | 0 steps |
| 4 | `3370f88` | geophysical variables converted to `float32` before packing | 92.4 s | 5.9 GB | ≤ 1 step on 0.1-0.2 % of the water pixels |

**Overall: 310 s → 93 s per tile (3.3 x faster), peak memory 6.4 → 5.9 GB.** The test
suite passes after each step.

### Where the time goes now

`final_breakdown.py`, default settings (`3370f88`):

| Stage | Time |
|---|---|
| `execute()`: graph construction only (28,199 tasks, was 194,988) | 3.5 s |
| computation of the L2B product in memory (mean CPU ≈ 400 %) | 36.2 s |
| int16 packing ranges (min/max, data in memory) | 2.7 s |
| NetCDF write (packing, zlib level 5) | ≈ 50 s |

Writing the NetCDF file is now more than half of the run time. The next steps are those
of phase 2, which change the product format and need a decision: fixed packing ranges
per variable (item 2a, would also remove the need to hold the L2B in memory), a lower
zlib level or Zarr output (item 4).
