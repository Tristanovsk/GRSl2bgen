Batch processing
================

Process a time series of GRS L2A products of one site or tile into L2B products:
list the products found on disk, select a date range, run the processing chain on
each of them (skipping those already processed and logging failures without stopping
the batch) and save a summary table. The last section gives a command-line
alternative for long batches, with one process per product.

.. toctree::
   :maxdepth: 2
   :caption: Batch processing

   grsl2bgen_batch_processing
