"""Package-wide processing settings."""

# selection method for the wavelength coordinate ``wl`` (see xarray
# DataArray.sel): 'nearest' matches the requested band centres (e.g. 490 nm)
# to the closest wavelength of the sensor (e.g. 492.4 nm for S2A); None
# requires exact wavelengths
SEL_XARRAY_METHOD = 'nearest'
