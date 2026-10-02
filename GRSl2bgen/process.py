"""Processing chain from an L2A product to an L2B product."""


import os
import logging
import contextlib

from . import Product, L2bProduct
from . import Chl, Spm, Cdom, Transparency, OWT_process

opj = os.path.join


@contextlib.contextmanager
def _dask_client(n_workers=None, threads_per_worker=1, memory_limit='auto'):
    """Optionally start a local dask.distributed cluster for the ``with`` block.

    ``dask.distributed`` is imported lazily, so importing this module stays
    cheap. With ``n_workers=None`` nothing is started and dask's default
    threaded scheduler is used.

    Parameters
    ----------
    n_workers : int, optional
        Number of worker processes. None = no cluster.
    threads_per_worker : int, optional
        Threads per worker (default 1; HDF5/netCDF4 reads are serialized
        within a process, so several single-threaded workers read in parallel).
    memory_limit : str or int, optional
        Memory limit per worker (default ``'auto'``).

    Yields
    ------
    dask.distributed.Client or None
    """
    if n_workers is None:
        yield None
        return
    from dask.distributed import Client, LocalCluster
    with LocalCluster(n_workers=n_workers,
                      threads_per_worker=threads_per_worker,
                      memory_limit=memory_limit) as cluster:
        with Client(cluster) as client:
            logging.info(f'dask dashboard: {client.dashboard_link}')
            yield client


class Process():
    """Processing chain from an L2A product to an L2B product.

    Loads the L2A product, restricts it to water pixels, derives the OWT
    classification, SPM, Chl-a, CDOM and transparency parameters, assembles
    them into an L2B product, and writes it to NetCDF.

    Parameters
    ----------
    l2a_obj : str or xarray.Dataset
        L2A product: file/directory path or loaded dataset (see `Product`).
    l2b_path : str, optional
        Output NetCDF path (default ``'./l2b_product.nc'``). A path ending in
        ``.zarr`` is written as a Zarr store, anything else as NetCDF.
    n_workers : int, optional
        If given, `run` executes the chain on a local dask.distributed
        cluster with this many worker processes. Default None: dask's
        default threaded scheduler.
    threads_per_worker : int, optional
        Threads per worker when `n_workers` is set (default 1).
    pyramid : bool, optional
        For ``.zarr`` output, write a cloud-ready multiscale store (resolution
        levels as groups ``0``, ``1``, ... plus consolidated metadata). See
        ``L2bProduct.export_to_zarr_pyramid``. Default False.
    persist_input : bool, optional
        If True, the water-masked input raster is computed once and kept in
        memory, so that SPM, Chl-a, CDOM and transparency do not each re-read
        and re-mask it. Needs enough RAM for the masked raster (default
        False).


    Attributes
    ----------
    l2a_obj : str or xarray.Dataset
        Input L2A product.
    l2b_path : str
        Output file path.
    successful : bool
        True once `execute` has completed without error.
    l2b : L2bProduct
        Assembled L2B product; only available after `execute`.

    Examples
    --------
    >>> proc = Process('scene_L2A.nc', l2b_path='out/scene_L2B.nc')
    >>> proc.execute()
    >>> proc.write_output()
    """

    def __init__(self,
                 l2a_obj,
                 l2b_path='./l2b_product.nc',
                 n_workers=None,
                 threads_per_worker=1,
                 persist_input=False,
                 pyramid=False
                 ):
        self.l2a_obj = l2a_obj
        self.l2b_path = l2b_path
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.persist_input = persist_input
        self.pyramid = pyramid
        self.l2b = None
        self.successful = False

    def execute(self, ):
        """Run the full processing chain and build the L2B product.

        Steps:

        1. Load the L2A product (`Product`).
        2. Keep only valid water pixels (``mask == 0``).
        3. Derive OWT classification (`OWT_process`), SPM (`Spm`), Chl-a
           (`Chl`, using the Spyrakos et al. 2018 OWT SAM result), CDOM
           (`Cdom`) and transparency (`Transparency`).
        4. Merge the outputs into an `L2bProduct` stored in ``self.l2b``.

        Sets ``self.successful`` to True at the end. Exceptions are not
        caught.

        Raises
        ------
        KeyError
            If the L2A raster has no ``mask`` variable.

        """

        logging.info('import l2a product')
        l2a_obj = self.l2a_obj

        prod = Product(l2a_obj)

        # Apply water mask: restrict all processing to valid water pixels (mask == 0)
        raster = prod.raster.where(prod.raster['mask'] == 0)
        if self.persist_input:
            # evaluate the masked input once and share it between all modules
            raster = raster.persist()

        #  ----------------------
        # get OWT parameters
        # ----------------------
        logging.info('get OWT classification')
        owt_process = OWT_process(raster)
        owt_process.execute()

        # ----------------------
        # get SPM parameters
        # ----------------------
        logging.info('get SPM parameters')
        spm_prod = Spm(raster)
        spm_prod.process()

        # ----------------------
        # get Chl-a parameters
        # ----------------------
        logging.info('get Chl-a parameters')
        chl_prod = Chl(raster,xowt_prod=owt_process.xowt_spyrakos2018)
        chl_prod.process()

        # ----------------------
        # get CDOM parameters
        # ----------------------
        logging.info('get CDOM parameters')
        cdom_prod = Cdom(raster)
        cdom_prod.process()

        # ----------------------
        # get transparency parameters
        # ----------------------
        logging.info('get transparency parameters')
        trans_prod = Transparency(raster)
        trans_prod.process()

        logging.info('construct l2b product')
        l2_raster_list = [
            owt_process.output,
            chl_prod.output,
            spm_prod.output,
            cdom_prod.output,
            trans_prod.output]
        self.l2b = L2bProduct(prod, l2_raster_list)
        self.successful = True

    def write_output(self):
        """Export the L2B product to ``self.l2b_path`` (NetCDF, or Zarr if it ends in ``.zarr``).

        Must be called after `execute`.

        Raises
        ------
        AttributeError
            If `execute` has not been run (``self.l2b`` does not exist yet).
        """
        logging.info('export final l2b product')
        self.l2b.export(self.l2b_path, pyramid=self.pyramid)

    def run(self):
        """Run `execute` and `write_output` inside one dask scheduler context.

        Use this instead of calling the two methods separately when
        `n_workers` is set: the distributed cluster must stay alive until the
        file is written, because the L2B product is evaluated lazily and the
        computation happens in `write_output`.
        """
        with _dask_client(self.n_workers, self.threads_per_worker):
            self.execute()
            self.write_output()
