Batch processing
================

Process full Sentinel-2 tiles of GRS L2A products (Zarr pyramids of GRS 3.x or NetCDF files
of GRS 2.x) into L2B products, for one or several tiles and a date range: scan the L2A
folder, leave out incomplete products, keep one GRS version per product, run the
processing chain on each of them (skipping those already processed and logging failures
without stopping the batch) and save a summary table. The last section gives a
command-line alternative for long batches, with one process per product.

.. toctree::
   :maxdepth: 2
   :caption: Batch processing

   grsl2bgen_batch_processing
