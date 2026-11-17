# OWT update for Tarasenko, Gernez & Harmel (30TWT)

Notebooks for updating the OWT part of the article (manuscript: `grstbx/notebook/case_study/tarasenko`).
They start from the GRS 3.1.1 **L2A** products of `s3://grstbx-s3/sentinel-2/l2a/30TWT`, read from the local copy
`/data/satellite/s3/grstbx-s3` or through the STAC catalogue (AWS profile `grstbx-local`). Kernel: `py12`
(GRSl2bgen, grstbx, invRrs, hablab_tbx).

| notebook | content | outputs |
|---|---|---|
| `00_data_inventory_30TWT` | L2A inventory, images of the article, EMODnet DTM, geometry | `data/inventory_l2a_30TWT.csv`, `data/training_images_article.csv`, `data/geometry_30TWT.json` |
| `01_TA25_owt_update` | Ta2025 rebuilt from Bi 2024 + the table of the article; bloom-type medians recomputed with GRS 3.1.1 on the training images | `data/owt_tarasenko2025_v2.nc` |
| `02_invRrs_owt_hablab` | HABLAB species x Chl-a 20/200/600 (invRrs `TwoModes`), separability, test set | `data/owt_hablab_invRrs_{s2,hyper}.nc`, `data/hablab_invRrs_testset_s2.nc` |
| `03_invRrs_owt_water_types` | 14 water types, oligotrophic to extremely turbid (invRrs `TwoModesSi`), to replace Cordeiro 2022 | `data/owt_water_types_invRrs_{s2,hyper}.nc` |
| `04_l2a_reduction_30TWT` | one pass over the L2A images: candidate bloom pixels, water subsample, class counts, depth | `cache/30TWT/img/*.npz`, `cache/30TWT/records.csv` |
| `05_owt_sensitivity` | spectral range, second SAM for HAB / massive blooms, shallow mask, libraries | `cache/30TWT/sensitivity_*`, `figures/sens_*.png` |
| `06_test_new_owt_30TWT` | new libraries on the images (water types vs Cordeiro, Ta2025 v1 / v2 / + HABLAB) | `figures/wt_*.png` |
| `07_article_figures` | figures of the manuscript from the L2A cache (`COPY_TO_ARTICLE` to copy them) | `figures/*.png`, `figures/numbers_30TWT_l2a.json` |

Shared code: `ta_common.py`. Run the notebooks in order: 04 reads the libraries of 01 to 03, and 05 to 07 read the 04 cache.
`TA_MAX_IMAGES=2` (environment variable) limits 04 to a few images for a quick test.

Nothing here modifies the package: the new libraries can be passed to `GRSl2bgen.OWT(raster, xowt=...)`, and
they go to `GRSl2bgen/data/` only once validated.
