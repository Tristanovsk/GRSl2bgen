Output Product
==============

This page describes the content of the Level-2B product written by GRSl2bgen: the
``xarray.Dataset`` assembled by :py:class:`GRSl2bgen.output.L2bProduct` and exported to
NetCDF or Zarr (see :doc:`processing_chain` for the formats and the Zarr pyramids, and
:doc:`algorithms` for the equations).

Overview
--------

Read the product with ``xarray``; packed values are unpacked automatically:

.. code-block:: python

   import xarray as xr

   ds = xr.open_dataset('S2B_MSIL2B_<...>.nc')                  # NetCDF
   # ds = xr.open_zarr('S2B_MSIL2B_<...>.zarr')                 # Zarr
   # ds = xr.open_zarr('S2B_MSIL2B_<...>.zarr', group='0')      # Zarr pyramid, level 0

Example (300 x 300 pixel subset of a Sentinel-2 image):

.. code-block:: text

   <xarray.Dataset>
   Dimensions:                 (Nclasses: 3, y: 300, x: 300)
   Coordinates:
     * Nclasses                (Nclasses) int64 0 1 2
     * x                       (x) float64 ...
     * y                       (y) float64 ...
       time                    datetime64[ns] 2024-07-06T09:15:59
       spatial_ref             int64 0
   Data variables:
       owt_dist_Spyrakos2018   (Nclasses, y, x) float32 ...
       owt_index_Spyrakos2018  (Nclasses, y, x) float32 ...
       owt_dist_Bi2024         (y, x) float32 ...
       owt_index_Bi2024        (y, x) float32 ...
       owt_dist_Ta2025         (y, x) float32 ...
       owt_index_Ta2025        (y, x) float32 ...
       owt_dist_Cordeiro2022SPM  (Nclasses, y, x) float32 ...
       owt_index_Cordeiro2022SPM (Nclasses, y, x) float32 ...
       Chla_OC2nasa            (y, x) float64 ...
       Chla_M09B               (y, x) float64 ...
       Chla_NIRB               (y, x) float64 ...
       Chla_OWTblend_Ta2025    (y, x) float64 ...
       SPM_obs2co              (y, x) float64 ...
       TURB_dogliotti          (y, x) float64 ...
       SPM_nechad              (y, x) float64 ...
       SPM_OWTblend_Ta2025     (y, x) float32 ...
       acdom_B15               (y, x) float64 ...
       adg443_QAAv6            (y, x) float64 ...
       adg443_QAAv6Chl         (y, x) float64 ...
       Kd_par                  (y, x) float64 ...
       Zsd_Lee15               (y, x) float64 ...
       flags                   (y, x) uint32 ...
       flags_l2b               (y, x) uint32 ...
       mask                    (y, x) uint8 ...
   Attributes: (73)
       ...

All geophysical variables are defined on valid water pixels only (``mask == 0``); every
other pixel is NaN.

Dimensions and coordinates
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Name
     - Type
     - Description
   * - ``y``, ``x``
     - dimension, ``float64``
     - Projected coordinates of the pixel centres (m), same grid as the input L2A
       product (e.g. UTM for Sentinel-2).
   * - ``Nclasses``
     - dimension, ``int64``
     - Rank of the OWT class, ``0`` = best match, ``1`` = second best, ``2`` = third
       best. Only used by the Spyrakos et al. (2018) variables (3 ranks) and the
       Cordeiro (2022) variables (2 ranks, the third one is NaN).
   * - ``time``
     - scalar, ``datetime64``
     - Acquisition date and time.
   * - ``spatial_ref``
     - scalar, ``int64``
     - CF grid mapping: coordinate reference system (``crs_wkt``) and
       ``GeoTransform``. Every variable refers to it through its ``grid_mapping``
       attribute, so the product can be opened directly in GIS software.

Data variables
--------------

The "Valid range" column gives the filter applied by the processor: values outside are
set to NaN (or to 0 when noted).

Optical Water Types
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 27 16 12 45

   * - Variable
     - Dimensions
     - Unit
     - Description
   * - ``owt_index_Spyrakos2018``
     - (Nclasses, y, x)
     - --
     - Class number (1--13) of the three best-matching OWT of Spyrakos et al. (2018),
       ordered by increasing spectral angle.
   * - ``owt_dist_Spyrakos2018``
     - (Nclasses, y, x)
     - rad
     - Spectral angle :math:`\theta` (:eq:`sam`) to each of these classes,
       :math:`0 \le \theta \le \pi`.
   * - ``owt_index_Bi2024``
     - (y, x)
     - --
     - Class number (1--10) of the best-matching OWT of Bi and Hieronymi (2024).
   * - ``owt_dist_Bi2024``
     - (y, x)
     - rad
     - Spectral angle to this class.
   * - ``owt_index_Ta2025``
     - (y, x)
     - --
     - Class number (1--16) of the best-matching OWT used in Tarasenko et al. (2025).
   * - ``owt_dist_Ta2025``
     - (y, x)
     - rad
     - Spectral angle to this class.
   * - ``owt_index_Cordeiro2022SPM``
     - (Nclasses, y, x)
     - --
     - Class number (1--4) of the two best-matching SPM OWT of Cordeiro (2022),
       ordered by increasing Euclidean distance. The ``Nclasses`` dimension is shared
       with Spyrakos2018, so the third entry is NaN.
   * - ``owt_dist_Cordeiro2022SPM``
     - (Nclasses, y, x)
     - sr\ :sup:`-1`
     - Euclidean distance :math:`d` (:eq:`euclidean`) to each of these classes.

The meaning of each class number is stored in the ``definition`` attribute of the
``owt_index_*`` variables:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Database
     - Classes (index: name)
   * - Spyrakos2018
     - 1: hypereutrophic; 2: common case; 3: clear; 4: turbid with organic content;
       5: sediment-laden; 6: balanced optical effects at shorter wavelengths;
       7: highly productive cyanobacteria-dominated; 8: productive with cyanobacteria;
       9: OWT2 with higher :math:`\Rrs` at shorter wavelengths; 10: CDOM-rich;
       11: CDOM-rich with cyanobacteria; 12: turbid with cyanobacteria;
       13: very clear blue waters.
   * - Bi2024
     - 1: type 1; 2: type 2; 3: type 3a; 4: type 3b; 5: type 4a; 6: type 4b;
       7: type 5a; 8: type 5b; 9: type 6; 10: type 7 (see the
       :doc:`tutorials/grsl2bgen_owt_minimal` tutorial for their description).
   * - Ta2025
     - 1--10: Bi2024 types 1, 2, 3a, 3b, 4a, 4b, 5a, 5b, 6, 7;
       11: Cyanobacteria; 12: Mesodinium; 13: Dinoflagellates2; 14: red;
       15: Dinoflagellates1; 16: Lepidodinium.
   * - Cordeiro2022SPM
     - 1: clear; 2: moderate; 3: turbid; 4: very turbid.

Chlorophyll-a
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 12 24 40

   * - Variable
     - Unit
     - Valid range
     - Description
   * - ``Chla_OC2nasa``
     - mg m\ :sup:`-3`
     - ]0, 1200[
     - NASA OC2, OCTS parameterisation, 490/560 nm ratio (:eq:`oc2`); oligotrophic
       waters.
   * - ``Chla_M09B``
     - mg m\ :sup:`-3`
     - ]0, 1200[
     - Moses et al. (2009) three-band red-edge index, 665/705/740 nm (:eq:`m09b`);
       turbid and eutrophic waters.
   * - ``Chla_NIRB``
     - mg m\ :sup:`-3`
     - ]0, 1200[
     - Martin et al. (2025) 705/443 nm ratio (:eq:`nirb`); coastal (case-2) waters.
   * - ``Chla_OWTblend_Ta2025``
     - mg m\ :sup:`-3`
     - ]0, 1200[
     - OWT-blended Chl-a (:eq:`chl-blend`): Spyrakos2018 classes, recipe of Tavares
       et al. (2025), 2 best classes blended (``n_classes_blended`` attribute); only
       the classes with a valid estimate are averaged.

Suspended particulate matter and turbidity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 12 24 40

   * - Variable
     - Unit
     - Valid range
     - Description
   * - ``SPM_nechad``
     - mg L\ :sup:`-1`
     - [0, 2000], negative values set to 0
     - Nechad et al. (2010), 665 nm (:eq:`nechad`); low to moderately turbid waters.
   * - ``SPM_obs2co``
     - mg L\ :sup:`-1`
     - [0, 2000], negative values set to 0
     - OBS2CO switching algorithm, 665 and 865 nm (:eq:`switch`).
   * - ``TURB_dogliotti``
     - FNU
     - [0, 2000], negative values set to 0
     - Dogliotti et al. (2015) switching algorithm, 665 nm (:eq:`switch`).
   * - ``SPM_OWTblend_Ta2025``
     - mg L\ :sup:`-1`
     - ]0, 2000[, plus a validity range per class algorithm
     - OWT-blended SPM: Cordeiro2022SPM classes, 2 best classes blended
       (``n_classes_blended`` attribute) with Jiang et al. (2021), Zhang et al. (2014)
       and Binding et al. (2010) (:eq:`jiang-green` to :eq:`binding`); only the
       classes with a valid estimate are averaged.

CDOM and transparency
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 12 24 40

   * - Variable
     - Unit
     - Valid range
     - Description
   * - ``acdom_B15``
     - m\ :sup:`-1`
     - ]0, 30[
     - CDOM absorption coefficient at 440 nm, Brezonik et al. (2015), 490/740 nm ratio
       (:eq:`cdom`).
   * - ``adg443_QAAv6``
     - m\ :sup:`-1`
     - ]0, 30[
     - Detritus and CDOM absorption at 443 nm, QAA v6 adapted to MSI (443/490 nm
       decomposition, :eq:`qaa-adg`), with ``Chla_OWTblend_Ta2025`` as first guess for
       the phytoplankton absorption ratio.
   * - ``adg443_QAAv6Chl``
     - m\ :sup:`-1`
     - ]0, 30[
     - Detritus and CDOM absorption at 443 nm, QAA v6 non-water absorption minus the
       phytoplankton absorption of Bricaud et al. (1998) computed from
       ``Chla_OWTblend_Ta2025`` (:eq:`qaa-adg-chl`); not validated yet.
   * - ``Kd_par``
     - m\ :sup:`-1`
     - ]0, 3000[
     - Diffuse attenuation coefficient of PAR, Roy and Das (2022) (:eq:`kdpar`).
   * - ``Zsd_Lee15``
     - m
     - ]0, 100[
     - Secchi disk depth, Lee et al. (2015) (:eq:`zsd`), from QAA v6 IOPs and the
       Kd model of Lee et al. (2013) at the most transparent band (443--665 nm);
       only written when the L2A product has the solar zenith angle ``sza``.

.. note::

   The ``range`` attribute of the geophysical variables gives the bounds of the
   validity filter applied, listed in the tables above (e.g. ``[0, 1200]`` for Chl-a,
   ``[0, 2000]`` for SPM and turbidity, ``[0, 30]`` for the CDOM products, ``[0, 3000]``
   for ``Kd_par``, ``[0, 100]`` for ``Zsd_Lee15``). Products written by GRSl2bgen
   versions earlier than 1.1.0 may carry different, informative values (e.g.
   ``[0, 2000]`` for Chl-a, ``[0, 200]`` for ``Kd_par``).

Quality flags and mask
~~~~~~~~~~~~~~~~~~~~~~

``flags`` and ``mask`` are copied from the input GRS L2A product (or set to 0 if the
input has none) and are stored without packing. The flags set by GRSl2bgen are in a
separate bit field, ``flags_l2b`` (see below).

``mask`` (``uint8``): ``0`` = valid water pixel, processed; ``1`` = rejected pixel
(land, cloud, glint, negative reflectance, ...).

``flags`` (``uint32``) is a bit field: bit :math:`i` is set when the condition
``flag_names[i]`` is met. A pixel can have several flags. With the current GRS
flags:

.. list-table::
   :header-rows: 1
   :widths: 8 12 30 50

   * - Bit
     - Value
     - Name
     - Description
   * - 0
     - 1
     - ``nodata``
     - No data in the input image
   * - 1
     - 2
     - ``cloud_p06``
     - Low-confidence cloud mask (s2cloudless, probability threshold 0.6)
   * - 2
     - 4
     - ``cloud_p08``
     - High-confidence cloud mask (s2cloudless, probability threshold 0.8)
   * - 3
     - 8
     - ``water_swir_visible_index``
     - Water from a normalized SWIR/visible index (490 and 1610 nm)
   * - 4
     - 16
     - ``water_red_visible_index``
     - Water from a normalized NIR/visible index (490 and 842 nm); can fail in turbid
       waters
   * - 5
     - 32
     - ``thin_cirrus``
     - Thin cirrus, from the cirrus band
   * - 6
     - 64
     - ``opac_cirrus``
     - Opaque cirrus, from the cirrus band
   * - 7
     - 128
     - ``high_swir``
     - High SWIR reflectance (bright cloud, strong reflection, ...)
   * - 8
     - 256
     - ``surfwater_land``
     - Land, from the SurfWater input file
   * - 9
     - 512
     - ``surfwater_water``
     - Water, from the SurfWater input file
   * - 10
     - 1024
     - ``surfwater_cloud_and_shadow``
     - Cloud and shadow, from the SurfWater input file
   * - 18
     - 262144
     - ``neg_rrs``
     - Negative :math:`\Rrs` in the blue or green bands

The thresholds used by GRS are given in the ``flag_descriptions`` attribute of the
product, and the fraction of the image covered by each flag in the global attributes
``flag_<name>``. Decode a flag with:

.. code-block:: python

   names = list(ds.flags.attrs['flag_names'])
   bit = names.index('neg_rrs')
   neg_rrs = (ds.flags.astype('int64') >> bit) & 1      # 1 where the flag is set

For example, ``flags = 536 = 8 + 16 + 512`` means that both water indices and the
SurfWater mask detect water.

``flags_l2b`` (``uint32``) is the bit field of the flags set by GRSl2bgen, with the
same ``flag_names`` and ``flag_descriptions`` attributes and the same decoding:

.. list-table::
   :header-rows: 1
   :widths: 8 12 30 50

   * - Bit
     - Value
     - Name
     - Description
   * - 1
     - 2
     - ``swir_module_low``
     - SWIR module of the L2A reflectance,
       :math:`\sqrt{\Rrs(1610)^2 + \Rrs(2200)^2}` (nearest bands), above 0.002 sr\ :sup:`-1`
   * - 2
     - 4
     - ``swir_module_medium``
     - SWIR module above 0.005 sr\ :sup:`-1`
   * - 3
     - 8
     - ``swir_module_high``
     - SWIR module above 0.01 sr\ :sup:`-1`
   * - 12
     - 4096
     - ``zsd_a_below_aw``
     - ``Zsd_Lee15``: QAA v6 total absorption below the pure water absorption at the
       transparent band, set to :math:`a_w`

Water is black in the SWIR: a high SWIR module points to residual sun glint, adjacency
effect, thin cloud or a land/water mixed pixel. The three SWIR flags are cumulative (a
pixel above 0.01 sr\ :sup:`-1` has bits 1, 2 and 3 set); they are computed on the
processed water pixels (``mask == 0``) by :py:class:`~GRSl2bgen.mask.QC_L2A`, whose
thresholds can be changed (the values used are given in ``flag_descriptions``).

Variable attributes
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Content
   * - ``units``
     - Physical unit (``radians`` for the SAM ``owt_dist_*`` variables, ``sr-1`` for
       ``owt_dist_Cordeiro2022SPM``).
   * - ``description``
     - Short description of the retrieval and of the bands used.
   * - ``applicability``
     - Water types for which the algorithm is designed.
   * - ``coef``
     - Coefficients of the algorithm.
   * - ``reference``, ``references``
     - Bibliographic reference(s).
   * - ``range``
     - Indicative range (see the note above).
   * - ``definition``
     - (``owt_index_*``) meaning of each class number.
   * - ``n_classes_blended``
     - (``Chla_OWTblend_Ta2025``) number of OWT classes blended.
   * - ``flag_names``, ``flag_descriptions``
     - (``flags``, ``flags_l2b``, ``mask``) name and description of each bit.
   * - ``grid_mapping``
     - ``spatial_ref`` (written on export).

Global attributes
-----------------

The global attributes of the input L2A product are copied: sensor and acquisition
metadata (``constellation``, ``satellite``, ``instrument``, ``tile``,
``acquisition_date``, ``orbit_direction``, ``FOOTPRINT``, ``PRODUCT_URI``, ...), sun
geometry (``mean_solar_zenith_angle``, ``mean_solar_azimuth``), GRS processing settings
and flag statistics (``flag_<name>``). GRSl2bgen adds:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Content
   * - ``processor``
     - ``GRSl2bgen_<version>`` used to generate the product.
   * - ``processing_time``
     - Local processing date and time (ISO 8601).

For Zarr pyramids, the root group also holds the ``multiscales`` attribute
(see :doc:`processing_chain`).

Storage and packing
-------------------

On export, every geophysical variable is packed into a 16-bit integer. For each
variable, the scale factor :math:`s` and offset :math:`o` are computed from the minimum
and maximum :math:`[v_{\min}, v_{\max}]` of the variable **in this product**:

.. math::

   s = \frac{v_{\max} - v_{\min}}{2^{16} - 2}, \qquad o = v_{\min} + (2^{15} - 1)\, s

and the stored integer :math:`p` relates to the physical value :math:`v` by

.. math::

   v = p\, s + o, \qquad p \in [-32767, 32767]

so that :math:`v_{\min}` is stored as :math:`-32767` and :math:`v_{\max}` as
:math:`32767`. The lowest integer, :math:`-32768`, is reserved for
``_FillValue`` (NaN). ``xarray`` (and most NetCDF readers) apply this conversion
automatically. A constant variable is stored with :math:`s = 1`, :math:`o = v_{\min}`.

The quantisation step is :math:`s \approx (v_{\max} - v_{\min}) / 65534` (maximum
error :math:`s/2`), e.g. 0.02 mg m\ :sup:`-3` for a Chl-a field ranging from 0 to
1200 mg m\ :sup:`-3`. Because :math:`s` and :math:`o` differ from one product to
another, compare products after unpacking, not on the raw integers.

.. note::

   Products written by GRSl2bgen versions earlier than this fix used
   :math:`s = (v_{\max} - v_{\min}) / (2^{16} - 1)` and :math:`o = v_{\min} + 2^{15} s`,
   which packs :math:`v_{\min}` onto the fill value: in those products, pixels equal to
   the minimum of a variable (e.g. SPM or turbidity equal to 0) read back as NaN.

The OWT class numbers ``owt_index_*`` are not packed: they are stored as ``uint8``
with ``_FillValue = 0`` (class numbers start at 1) and read back as exact integers
(``float32``, NaN where no class is assigned).

.. note::

   Products written by earlier GRSl2bgen versions packed ``owt_index_*`` like the
   geophysical variables, so their class numbers read back as e.g. 8.99998 instead of 9:
   round them before comparing with a class number.

``flags``, ``flags_l2b`` and ``mask`` keep their integer type. In memory, before export,
all the geophysical variables are ``float32``; read back from the file, the packed
variables are unpacked by ``xarray`` as ``float64``.

.. _stac-item:

Product folder, log and STAC Item
---------------------------------

With ``--product_dir``, the CLI writes each product in its own folder, together with
the log of the run and, with ``--stac``, its `STAC <https://stacspec.org>`_ Item:

.. code-block:: text

   <odir>/S2B_MSIL2B_<...>_V3.1.0/
       S2B_MSIL2B_<...>_V3.1.0.nc      (or .zarr/)   product
       S2B_MSIL2B_<...>_V3.1.0.log                   log of the run (see Log Format)
       S2B_MSIL2B_<...>_V3.1.0.json                  STAC Item

One folder per product is the layout expected on object storage (one S3 prefix per
item): a product is copied, listed or deleted as a whole, and two products of the same
date never share a folder. Without ``--product_dir`` the three files are written side by
side in the output directory.

The Item is written last, once the product and its log are complete (the product
itself is written under a temporary name and renamed when complete): an existing Item
marks a finished product, and ``--no_clobber`` with ``--stac`` skips a product only
when its Item exists. When uploading the folder, upload the Item last as well.

The Item holds:

- ``geometry`` and ``bbox``: the product grid in longitude/latitude; ``datetime``: the
  acquisition time of the L1C;
- the ``data`` asset (role ``data``) and the ``log`` asset (role ``metadata``), with
  links relative to the Item (``./<name>.nc``), so that the folder can be moved or
  uploaded unchanged;
- extensions: `projection <https://github.com/stac-extensions/projection>`_
  (``proj:code``, ``proj:shape``, ``proj:transform``, ``proj:bbox``),
  `processing <https://github.com/stac-extensions/processing>`_ (``processing:level``
  ``L2B``, ``processing:software`` with the GRSl2bgen and GRS versions,
  ``processing:datetime``, ``processing:lineage``),
  `file <https://github.com/stac-extensions/file>`_ (``file:size`` and
  ``file:checksum``, SHA2-256 multihash, of the NetCDF and the log; none for a Zarr
  store), `eo <https://github.com/stac-extensions/eo>`_ (``eo:cloud_cover`` of the L1C)
  and `grid <https://github.com/stac-extensions/grid>`_ (``grid:code``, MGRS tile);
- a ``derived_from`` link to the L2A input, with its path as given to the CLI.

From Python, :py:func:`GRSl2bgen.stac.write_stac_item` writes the Item of any L2B
product::

   from GRSl2bgen.stac import write_stac_item

   write_stac_item('out/l2b.nc', l2a=l2a_path, log_file='out/l2b.log')

.. note::

   The log and the ``derived_from`` link hold the local paths of the run. Before
   publishing the folders in a public bucket, point ``derived_from`` to the published
   L2A Item, or keep the logs in a private location.
