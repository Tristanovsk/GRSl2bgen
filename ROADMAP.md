# GRSl2bgen roadmap

## OWT and bloom detection (beyond Tarasenko et al.)

Not for the current article: the article keeps the Ta2025 classification (spectral angle, without type 5a) and the
superbloom criterion. This section records why visible blooms are missed and how the OWT part could be upgraded.

### Diagnosis (22 Nov 2026)

Test: `grstbx/notebook/case_study/tarasenko/analysis/owt_library_test.py` (outputs in `owt_library_test/`), four
windows of tile 30TWT at 20 m (GRS 3.1.1 L2A): 2021-09-03, 2021-08-14, 2018-07-31, 2019-07-16. "Visible bloom" =
red-edge extent (NDCI > 1/11 on a 3x3 median, Rrs705 > 0.0015, groups >= 9 px). Visual reference: 10 m L1C RGB/ReGB
composites (`analysis/l1c_regb_10m.py`, `/data/satellite/Sentinel-2/l1c_regb_10m/30TWT/index.html`).

1. **The detection gate, not the OWT, misses most blooms.** The paper method detects 8-29 % of the visible red-edge
   area. The missed area fails the superbloom criterion: Rrs705 <= 0.005 (diluted blooms, 68-83 % of the missed area
   in 2021) or Rrs705/Rrs665 <= 1.8 (46-75 % in 2018-2019, e.g. the discoloured water of the Vilaine plume). Only
   6-29 % of it is superbloom assigned to a water type, mostly **5b** (82-87 %) and 4b; 3-10 % is in patches below
   0.04 km2.
2. **The distance is not the lever.** Euclidean distance of area-normalised spectra vs spectral angle (Ta2025):
   same blooms, 4-29 % less detected area, no new bloom (same on the 189-image sample of 21 Nov: 168 vs 186 events,
   `analysis/distance_test/`). Euclidean top-3 + SAM = SAM.
3. **Green blooms have no red edge.** On 2021-09-03, the green swirls offshore of the Loire (107 km2 in the window)
   have a 560 nm peak (560/490 = 1.9) and OC2 chl-a ~8 mg m-3 (background 1.0, diluted bloom margins 7.5), but
   Rrs665-783 ~0 or negative (-0.0006 sr-1 at 740 nm, also in the background: over-correction of the red/NIR by the
   atmospheric correction offshore). No red-edge gate can see them, and the OWT puts them in 4a/4b.
4. **The invRrs library as built is not robust.** 14 invRrs water types + 42 HABLAB bloom spectra (14 species x chl-a
   20/200/600, one moderate background): the species of the detected pixels are unstable (cyanobacteria and "other
   HABLAB species" dominate, L. chlorophorum almost never chosen); without the superbloom gate (log-Rrs Euclidean),
   76-87 % of the bloom-class pixels are outside the visible blooms (turbid plumes classified as blooms). The bloom
   and water spectra were simulated over different backgrounds, so the class competition reflects the background
   (SPM, CDOM) rather than the species.

### Plan

1. **Two stages: detect, then identify.**
   - Detection and intensity of the bloom from continuous indices with spatial coherence, replacing the fixed
     superbloom thresholds: red edge (NDCI or MCI) for dense blooms, blue-green information (560/490 or an OC chl-a
     anomaly relative to the local background) for green and diluted blooms; the refined extent (NDCI growth,
     chl_min) becomes part of this stage.
   - Species identification only on the detected pixels, with a bloom-type library and a confidence (angle, gap to
     the second class), "unidentified bloom" allowed.
2. **Reduce Ta2025 for identification.** Merge the near-identical water types (3a/3b: 2.9 deg, 4a/4b: 1.8 deg); 5a
   already left out; 5b absorbs superbloom pixels (29 % of the missed area on 2019-07-16) but removing it adds false
   cyanobacteria on vegetated shores (paper test): handle it in stage 1 (bottom / shallow mask, see GRSshallowat
   `optically_shallow` flag) rather than by exclusion.
3. **invRrs library v2.**
   - Joint grid simulated with the same model and SRFs: species (HABLAB IOPs) x chl-a (log grid, ~5-600 mg m-3) x
     background (SPM, a_cdom) matching the water types, so that a bloom and a water type are compared on the same
     background.
   - Hierarchical classification: water type (background) first, then bloom species within the matching
     background.
   - Distance with uncertainty: Mahalanobis or likelihood using the ensemble spread
     (`notebook/tarasenko/data/water_types_invRrs_s2_ensemble.nc`, 200 members) and the Rrs uncertainty of GRS, giving
     memberships instead of a hard class.
   - Validation: HABLAB test set (`hablab_invRrs_testset_s2.nc`), training images, REPHY matchups (incl. L. polyedra
     2021-2022, which Ta2025 labels L. chlorophorum).
4. **Atmospheric correction.** Negative Rrs at 665-783 nm offshore (2021-09-03) remove the red edge of moderate
   blooms: check the GRS 3.1.1 red/NIR residual on clear water (GRS roadmap); meanwhile, offset or band weighting in
   the classification.
5. **Evaluation set.** Label bloom / no bloom (and green / red-edge) windows from the 10 m composites and the ReGB
   catalogue (`/data/satellite/Sentinel-2/regb_catalogue/30TWT`), then score each configuration by precision and
   recall per window instead of by eye; extend `owt_library_test.py` to it.
