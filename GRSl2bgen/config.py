"""Package-wide processing settings."""

# selection method for the wavelength coordinate ``wl`` (see xarray
# DataArray.sel): 'nearest' matches the requested band centres (e.g. 490 nm)
# to the closest wavelength of the sensor (e.g. 492.4 nm for S2A); None
# requires exact wavelengths
SEL_XARRAY_METHOD = 'nearest'

# classes left out of the Tarasenko et al. (2025) OWT classification of the L2B chain (labels of the ``owt``
# coordinate of ``data/owt_tarasenko2025.nc`` or 1-based positions); empty: the 16 classes. Can be set per run
# (``Process(ta2025_exclude=...)``, option ``--ta2025_exclude`` of the command line).
TA2025_EXCLUDE_CLASSES = ()
