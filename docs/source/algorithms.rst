Algorithm Description
=====================

This page gives the mathematical formulation of every retrieval performed by
GRSl2bgen, as implemented in the code. Coefficients are the default values
used by :py:class:`GRSl2bgen.process.Process`; each section links to the
function that implements it.

Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Symbol
     - Definition
     - Unit
   * - :math:`\lambda`
     - Wavelength (central wavelength of the sensor band)
     - nm
   * - :math:`\Rrs(\lambda)`
     - Remote-sensing reflectance from the GRS L2A product (variable ``Rrs``)
     - sr\ :sup:`-1`
   * - :math:`R_\lambda`
     - Shorthand for :math:`\Rrs(\lambda)`, e.g. :math:`R_{665} = \Rrs(665)`
     - sr\ :sup:`-1`
   * - :math:`\rho_w(\lambda)`
     - Water-leaving reflectance, :math:`\rho_w = \pi\,\Rrs`
     - --
   * - :math:`\mathbf{R}`
     - Pixel spectrum, vector :math:`[\Rrs(\lambda_1), \dots, \Rrs(\lambda_N)]`
     - sr\ :sup:`-1`
   * - :math:`\mathbf{R}_j`
     - Mean spectrum of Optical Water Type :math:`j`
     - database dependent
   * - :math:`\theta_j`, :math:`d_j`
     - Spectral angle / Euclidean distance between :math:`\mathbf{R}` and :math:`\mathbf{R}_j`
     - rad / sr\ :sup:`-1`

For Sentinel-2/MSI, the band centres are rounded to 443, 490, 560, 665, 705,
740, 783, 842, 865 nm.

Water masking
-------------

All retrievals are restricted to valid water pixels, i.e. pixels whose
L2A ``mask`` variable equals 0 (see :py:meth:`GRSl2bgen.process.Process.execute`).
Every other pixel is set to NaN before processing.

Optical Water Types (OWT)
-------------------------

Each pixel spectrum :math:`\mathbf{R}` is compared with the :math:`M` mean
spectra :math:`\mathbf{R}_j` of an OWT database. The database spectra are
linearly interpolated on the image wavelengths, and only bands within
:math:`350 \le \lambda \le 800` nm are used (:py:class:`GRSl2bgen.owt.OWT`).

Spectral Angle Mapper
~~~~~~~~~~~~~~~~~~~~~

The default similarity measure is the spectral angle (:py:mod:`GRSl2bgen.sam`):

.. math::
   :label: sam

   \theta_j = \arccos\left(
       \frac{\sum_{i=1}^{N} \Rrs(\lambda_i)\, R_j(\lambda_i)}
            {\sqrt{\sum_{i=1}^{N} \Rrs(\lambda_i)^2}\;
             \sqrt{\sum_{i=1}^{N} R_j(\lambda_i)^2}}
   \right), \qquad j = 1, \dots, M

:math:`\theta_j \in [0, \pi]`; the smaller the angle, the closer the spectral
*shapes*. The angle does not depend on the magnitude of either spectrum, so
pixel spectra can be compared directly with the standardised (unit-area)
spectra of Spyrakos et al. (2018). Pixels with a non-finite band or a null
spectrum are left undefined.

Euclidean distance
~~~~~~~~~~~~~~~~~~

Alternatively (``spectral_distance='euclidean'``, :py:mod:`GRSl2bgen.euclidean`):

.. math::
   :label: euclidean

   d_j = \sqrt{\sum_{i=1}^{N} \left[\Rrs(\lambda_i) - R_j(\lambda_i)\right]^2}

Unlike :eq:`sam`, :eq:`euclidean` is sensitive to the reflectance magnitude,
so the class spectra must be in the same unit as the pixel spectra.

Class ranking
~~~~~~~~~~~~~

The classes are sorted by increasing distance and the :math:`K` best are
kept:

.. math::

   \theta_{(1)} \le \theta_{(2)} \le \dots \le \theta_{(K)},
   \qquad c_k = \text{index of the class of rank } k

They are stored as ``owt_dist_<database>`` (:math:`\theta_{(k)}`) and
``owt_index_<database>`` (:math:`c_k`, 1-based) along the ``Nclasses``
dimension.

.. list-table:: OWT classifications computed by the processing chain
   :header-rows: 1
   :widths: 25 15 15 45

   * - Database
     - Classes :math:`M`
     - Kept :math:`K`
     - Spectra
   * - Spyrakos et al. (2018)
     - 13
     - 3
     - Standardised :math:`\Rrs` (``m_nRrs``), inland waters
   * - Bi et al. (2024)
     - 10
     - 1
     - :math:`\Rrs` (``m_Rrs``)
   * - Tarasenko et al. (2025)
     - see ``owt_tarasenko2025.nc``
     - 1
     - Standardised :math:`\Rrs` (``m_nRrs``)

Chlorophyll-a
-------------

Implemented in :py:class:`GRSl2bgen.chlorophyll_a.Chl`. Every Chl-a product is
kept only within :math:`0 < \mathrm{Chl} < 1200` mg m\ :sup:`-3`; values
outside are set to NaN.

NASA OC2 (``Chla_OC2nasa``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Blue-green band-ratio polynomial, OCTS parameterisation, for oligotrophic
waters:

.. math::
   :label: oc2

   X = \log_{10}\left(\frac{R_{490}}{R_{560}}\right), \qquad
   \mathrm{Chl} = 10^{\,\sum_{i=0}^{4} a_i X^i}

with :math:`a = [0.2236,\ -1.8296,\ 1.9094,\ -2.9481,\ -0.1718]`.

Moses et al. 2009 three-band (``Chla_M09B``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Red-edge three-band index, for turbid and eutrophic waters:

.. math::
   :label: m09b

   \mathrm{Chl} = a_0 \left(\frac{1}{R_{665}} - \frac{1}{R_{705}}\right) R_{740} + a_1,
   \qquad a_0 = 232.329,\; a_1 = 23.174

NIR-blue ratio (``Chla_NIRB``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Martin et al. (2025), for coastal (case-2) waters:

.. math::
   :label: nirb

   \mathrm{Chl} = a_0 \left(\frac{R_{705}}{R_{443}}\right)^{a_1},
   \qquad a_0 = 11.2,\; a_1 = 1.7

OWT-blended Chl-a (``Chla_OWTblend_Ta2025``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following Tavares et al. (2025), each Spyrakos et al. (2018) class is assigned
the algorithm best suited to it, and the estimates of the :math:`K` best
classes of the pixel (:math:`K = 2` by default) are averaged with weights
inversely proportional to their spectral angle
(:py:meth:`~GRSl2bgen.chlorophyll_a.Chl.owt_blending`):

.. math::
   :label: chl-blend

   \mathrm{Chl}_{\mathrm{blend}} =
   \frac{\sum_{k=1}^{K} w_k\, \mathrm{Chl}_{\mathcal{A}(c_k)}}
        {\sum_{k=1}^{K} w_k},
   \qquad w_k = \frac{1}{\theta_{(k)} + \varepsilon},
   \quad \varepsilon = 10^{-6}

where :math:`\mathcal{A}(c)` is the algorithm assigned to class :math:`c`:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Spyrakos OWT
     - Algorithm :math:`\mathcal{A}`
     - Equation
   * - 1, 6, 10
     - Gons et al. (2005)
     - :eq:`gons`
   * - 2, 4, 5, 11, 12
     - NDCI (Mishra & Mishra, 2012)
     - :eq:`ndci`
   * - 7, 8
     - Gilerson et al. (2010), two-band
     - :eq:`gilerson`
   * - 3, 9, 13
     - NASA OC2
     - :eq:`oc2`

**Gons et al. (2005).** The backscattering coefficient is first estimated from
the NIR band, then Chl-a from the red-edge ratio:

.. math::
   :label: gons

   b_b = \frac{1.61\, R_{783}}{0.082 - 0.6\, R_{783}}, \qquad
   \mathrm{Chl} = \frac{\dfrac{R_{705}}{R_{665}}\left[a_w(708) + b_b\right]
                        - a_w(665) - b_b^{\,p}}{a^*_{\phi}}

with :math:`a_w(665) = 0.425` m\ :sup:`-1`, :math:`a_w(708) = 0.704`
m\ :sup:`-1`, :math:`p = 1.063` and :math:`a^*_\phi = 0.016` m\ :sup:`2` mg\ :sup:`-1`.

**Normalized Difference Chlorophyll Index (NDCI).**

.. math::
   :label: ndci

   \mathrm{NDCI} = \frac{R_{705} - R_{665}}{R_{705} + R_{665}}, \qquad
   \mathrm{Chl} = 14.039 + 86.115\,\mathrm{NDCI} + 194.325\,\mathrm{NDCI}^2

**Gilerson et al. (2010), two-band.**

.. math::
   :label: gilerson

   \mathrm{Chl} = \left(\frac{0.7864\, R_{705}/R_{665} - 0.4245}{0.022}\right)^{1.124}

Suspended particulate matter and turbidity
------------------------------------------

Implemented in :py:class:`GRSl2bgen.suspended_particulate_matter.Spm`. All
three products share the semi-analytical form of Nechad et al. (2010), written
with :math:`\rho_w = \pi \Rrs`:

.. math::
   :label: nechad

   f(\rho_w;\, A, C) = \frac{A\, \rho_w}{1 - \rho_w / C}

Negative values are set to 0 and values above 2000 are set to NaN.

Nechad et al. 2010 (``SPM_nechad``, mg L\ :sup:`-1`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \mathrm{SPM} = f\left(\rho_w(665);\; A = 342.1,\; C = 0.19563\right)

Switching algorithms
~~~~~~~~~~~~~~~~~~~~

The two other products switch between a low- and a high-turbidity relationship,
with a linear transition between two thresholds :math:`s_0 < s_1` applied to a
switching variable :math:`x` (see table):

.. math::
   :label: switch

   P =
   \begin{cases}
     P_{\mathrm{low}} & x \le s_0 \\
     (1 - w)\, P_{\mathrm{low}} + w\, P_{\mathrm{high}} & s_0 < x < s_1 \\
     P_{\mathrm{high}} & x \ge s_1
   \end{cases},
   \qquad w = \frac{x - s_0}{s_1 - s_0}

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 10 16

   * - Product
     - :math:`P_{\mathrm{low}}`
     - :math:`P_{\mathrm{high}}`
     - :math:`x`
     - :math:`[s_0, s_1]`
   * - ``SPM_obs2co`` (mg L\ :sup:`-1`), OBS2CO, calibrated on the GET
       radiometric database
     - :math:`f(\rho_w(665);\ 610.94,\ 0.2324)`
     - :math:`691.13\,\left(R_{865}/R_{665}\right)^{2.5411}`
     - :math:`R_{665}`
     - :math:`[0.07,\ 0.14]`
   * - ``TURB_dogliotti`` (FNU), Dogliotti et al. (2015)
     - :math:`f(\rho_w(665);\ 228.1,\ 0.1641)`
     - :math:`f(\rho_w(665);\ 3078.9,\ 0.2112)`
     - :math:`\rho_w(665)`
     - :math:`[0.05,\ 0.07]`

Coloured dissolved organic matter
---------------------------------

``acdom_B15`` (:py:class:`GRSl2bgen.cdom.Cdom`): CDOM absorption coefficient
at 440 nm from the band-ratio model of Brezonik et al. (2015):

.. math::
   :label: cdom

   a_{\mathrm{CDOM}}(440) = \exp\left[a_0 + a_1 \ln\left(\frac{R_{490}}{R_{740}}\right)\right],
   \qquad a_0 = 1.872,\; a_1 = -0.83

in m\ :sup:`-1`, kept within :math:`0 < a_{\mathrm{CDOM}}(440) < 30` m\ :sup:`-1`.

Transparency
------------

``Kd_par`` (:py:class:`GRSl2bgen.transparency.Transparency`): diffuse
attenuation coefficient of the photosynthetically available radiation, from
Roy and Das (2022, table 3, eq. 13):

.. math::
   :label: kdpar

   K_d(\mathrm{PAR}) = a_0 \exp\left[a_1 \left(R_{490} - R_{665}\right)\right],
   \qquad a_0 = 3.09,\; a_1 = -90.17

in m\ :sup:`-1`, kept within :math:`0 < K_d(\mathrm{PAR}) < 3000` m\ :sup:`-1`.

References
----------

- Bi, S., Hieronymi, M. (2024). Holistic optical water type classification
  for ocean, coastal, and inland waters. *Limnology and Oceanography*.
- Brezonik, P. L., Olmanson, L. G., Finlay, J. C., Bauer, M. E. (2015).
  Factors affecting the measurement of CDOM by remote sensing of optically
  complex inland waters. *Remote Sensing of Environment*, 157, 199--215.
  https://doi.org/10.1016/j.rse.2014.04.033
- Dogliotti, A. I., Ruddick, K. G., Nechad, B., Doxaran, D., Knaeps, E. (2015).
  A single algorithm to retrieve turbidity from remotely-sensed data in all
  coastal and estuarine waters. *Remote Sensing of Environment*, 156, 157--168.
- Gilerson, A. A., Gitelson, A. A., Zhou, J., et al. (2010). Algorithms for
  remote estimation of chlorophyll-a in coastal and inland waters using red
  and near infrared bands. *Optics Express*, 18, 24109--24125.
- Gons, H. J., Rijkeboer, M., Ruddick, K. G. (2005). Effect of a waveband shift
  on chlorophyll retrieval from MERIS imagery of inland and coastal waters.
  *Journal of Plankton Research*, 27, 125--127.
- Martin, S., Bryère, P., Gernez, P., Renosh, P. R., Doxaran, D. (2025).
  Towards reliable high-resolution satellite products for the monitoring of
  chlorophyll-a and suspended particulate matter in optically shallow coastal
  lagoons. *Remote Sensing*, 17, 3430. https://doi.org/10.3390/rs17203430
- Mishra, S., Mishra, D. R. (2012). Normalized difference chlorophyll index: a
  novel model for remote estimation of chlorophyll-a concentration in turbid
  productive waters. *Remote Sensing of Environment*, 117, 394--406.
- Moses, W. J., Gitelson, A. A., Berdnikov, S., Povazhnyy, V. (2009).
  Estimation of chlorophyll-a concentration in case II waters using MODIS and
  MERIS data: successes and challenges. *Environmental Research Letters*, 4, 1--8.
- Nechad, B., Ruddick, K. G., Park, Y. (2010). Calibration and validation of a
  generic multisensor algorithm for mapping of total suspended matter in
  turbid waters. *Remote Sensing of Environment*, 114, 854--866.
- Roy, A., Das, B. S. (2022). *Journal of Hydrology*.
- Spyrakos, E., O'Donnell, R., Hunter, P. D., et al. (2018). Optical types of
  inland and coastal waters. *Limnology and Oceanography*, 63, 846--870.
- Tavares, M. H., Guimarães, D., Roussillon, J., et al. (2025). A framework to
  retrieve water quality parameters in small, optically diverse freshwater
  ecosystems using Sentinel-2 MSI imagery. *Remote Sensing*, 17, 2729.
