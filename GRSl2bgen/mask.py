'''
Pixel flags of the L2B product.

The GRS L2A ``flags`` (cloud, water, cirrus, ... bits, see :doc:`/output_product`)
are copied unchanged into the L2B product. The flags set by GRSl2bgen are written to
a separate ``uint32`` bit field, ``flags_l2b``: bit ``i`` is set when the condition
``flag_names[i]`` is met, and ``flag_descriptions[i]`` describes it.

=====  ======================  ==============================================================
bit    name                    meaning
=====  ======================  ==============================================================
0      ``zsd_a_below_aw``      ``Zsd_Lee15``: QAA total absorption below the pure water
                               absorption at the transparent band, set to ``aw``
                               (`GRSl2bgen.transparency.Transparency`)
1      ``swir_module_low``     SWIR module of the L2A ``Rrs`` above the low threshold
                               (`QC_L2A`, default 0.002 sr-1)
2      ``swir_module_medium``  SWIR module above the medium threshold (default 0.005 sr-1)
3      ``swir_module_high``    SWIR module above the high threshold (default 0.01 sr-1)
=====  ======================  ==============================================================

A module adds a flag by registering its bit in `L2B_FLAGS` and returning a boolean
DataArray, which `l2b_flags` writes into ``flags_l2b`` (see `GRSl2bgen.process.Process`).
`Masking.create_mask` derives a binary good/bad mask from selected bits.
'''

import numpy as np
import xarray as xr
import pandas as pd

import logging

FLAG_NAME = 'flags'

FLAG_NAME_L2B = 'flags_l2b'

# L2B processing flags, written as the uint32 bit field ``flags_l2b``
# (the GRS L2A ``flags`` are copied unchanged).
# name: (bit, description)
L2B_FLAGS = {
    'zsd_a_below_aw': (0, 'Zsd_Lee15: QAA v6 total absorption below the pure water absorption '
                          'at the transparent band, set to aw'),
    # QC_L2A, descriptions completed with the thresholds used
    'swir_module_low': (1, 'SWIR module of the L2A Rrs above the low threshold'),
    'swir_module_medium': (2, 'SWIR module of the L2A Rrs above the medium threshold'),
    'swir_module_high': (3, 'SWIR module of the L2A Rrs above the high threshold'),
}


def l2b_flags(template, conditions, descriptions=None, number_of_flags=32):
    '''Build the ``flags_l2b`` bit field from the L2B processing flags (see `L2B_FLAGS`).

    Parameters
    ----------
    template : xarray.DataArray
        Array on the image grid (y, x).
    conditions : dict
        {flag name: boolean DataArray}, True where the bit is set.
    descriptions : dict, optional
        {flag name: description} replacing those of `L2B_FLAGS` (e.g. with
        the thresholds actually used).
    number_of_flags : int, optional
        Size of the flag name/description tables (default 32).

    Returns
    -------
    xarray.DataArray
        ``uint32`` ``flags_l2b`` with the attributes ``long_name``,
        ``flag_names`` and ``flag_descriptions`` (all registered flags, set
        or not).
    '''
    names = np.full(number_of_flags, 'None', dtype=object)
    flag_descriptions = np.full(number_of_flags, 'None', dtype=object)
    for name, (bit, description) in L2B_FLAGS.items():
        names[bit], flag_descriptions[bit] = name, description
    for name, description in (descriptions or {}).items():
        flag_descriptions[L2B_FLAGS[name][0]] = description

    flags = xr.zeros_like(template, dtype=np.uint32)
    for name, cond in conditions.items():
        bit = L2B_FLAGS[name][0]
        flags = flags | (cond.astype(np.uint32) * np.uint32(1 << bit))

    flags.attrs = {'long_name': 'flags computed by the L2B processing',
                   'flag_names': names.astype(str).tolist(),
                   'flag_descriptions': flag_descriptions.astype(str).tolist()}
    flags.name = FLAG_NAME_L2B
    return flags


class QC_L2A:
    '''Quality control of the L2A reflectance, written as bits of ``flags_l2b``.

    SWIR module ``sqrt(Rrs(1610)**2 + Rrs(2200)**2)`` (nearest bands): water is
    black in the SWIR, a high module points to residual glint, adjacency
    effect, thin cloud or a mixed land/water pixel. Three flags are set where
    the module exceeds each of the three thresholds (``swir_module_low`` <
    ``swir_module_medium`` < ``swir_module_high``, so a pixel above the
    highest threshold has the three bits set).

    Parameters
    ----------
    raster : xarray.Dataset
        L2A dataset with ``Rrs`` (sr-1) and a ``wl`` dimension.
    swir_module_thresholds : tuple of float, optional
        Three increasing thresholds (sr-1) of the SWIR module (default
        ``(0.002, 0.005, 0.01)``).
    swir_bands : tuple of float, optional
        The two SWIR wavelengths (nm), nearest available band used (default
        ``(1610, 2200)``).

    Attributes
    ----------
    swir_module : xarray.DataArray
        SWIR module (sr-1), set by `process`.
    flags : dict
        {flag name: boolean DataArray}, set by `process`.
    flag_descriptions : dict
        {flag name: description with the threshold used}, set by `process`.
    '''

    swir_flag_names = ('swir_module_low', 'swir_module_medium', 'swir_module_high')

    def __init__(self, raster, swir_module_thresholds=(0.002, 0.005, 0.01), swir_bands=(1610, 2200)):
        self.raster = raster
        self.swir_module_thresholds = swir_module_thresholds
        self.swir_bands = swir_bands
        self.swir_module = None
        self.flags = {}
        self.flag_descriptions = {}

    def process(self):
        '''Compute the SWIR module and fill `flags` and `flag_descriptions`.'''
        Rrs = self.raster['Rrs']
        b1, b2 = [Rrs.sel(wl=wl, method='nearest') for wl in self.swir_bands]
        # drop the per-band coordinates (wl, central_wavelength, ...)
        band_coords = [c for c in Rrs.coords if 'wl' in Rrs[c].dims]
        self.swir_module = np.sqrt(b1.drop_vars(band_coords) ** 2 + b2.drop_vars(band_coords) ** 2)
        self.swir_module.name = 'SWIR_module'

        bands = ' and '.join(f'{float(b.wl):.0f}' for b in (b1, b2))
        for name, threshold in zip(self.swir_flag_names, self.swir_module_thresholds):
            self.flags[name] = self.swir_module > threshold
            self.flag_descriptions[name] = (f'SWIR module sqrt(Rrs1**2 + Rrs2**2) of the L2A Rrs, bands '
                                            f'{bands} nm, above {threshold} sr-1')


class Settings:
    '''
    Settings used for the different masking procedures.
    '''

    def __init__(self):


        self.number_of_flags = 32


class Masking(Settings):
    '''
    Class dedicated to pixel classification and masking.
    '''

    def __init__(self, prod):
        '''

        :param prod: image raster containing the spectral bands
        '''
        Settings.__init__(self)
        self.prod = prod


        self.flag_descriptions = np.empty(self.number_of_flags,
                                          dtype='object')  # np.dtype('U', 1000))
        self.flag_names = np.empty(self.number_of_flags,
                                   dtype='object')  # np.dtype('U', 1000))

        # generate flag raster from nodata mask
        # self.create_flags()
        self.flags = xr.DataArray()
        self.flags.attrs['long_name'] = 'flags computed from l1c image'
        self.flag_stats = {}


    def get_stats(self):
        '''
        Compute image statistics for each flag and save them into dictionary

        :return:
        '''

        flag_value = 1
        for ii, flag_name in enumerate(self.prod[FLAG_NAME].flag_names):
            if flag_name != 'None':
                flag = ((self.prod[FLAG_NAME] & flag_value) != 0)
                flag_stat = float(flag.sum() / flag.count())
                self.flag_stats['flag_' + flag_name] = flag_stat
            flag_value = flag_value << 1
        self.prod.attrs.update(self.flag_stats)

    def print_stats(self):
        '''
        Provide pandas Dataframe with image statistics of each flag

        :return: dflags:: pandas Dataframe with statistics
        '''

        pflags = self.prod[FLAG_NAME]  # self.product[self.flag_ID]

        # construct dataframe:
        dflags = pd.DataFrame({'name': pflags.attrs["flag_names"]})
        dflags['description'] = pflags.attrs["flag_descriptions"]  # .split('\t')
        dflags['bit'] = dflags.index
        dflags = dflags[dflags.name != "None"]
        stats = []
        for name in dflags.name:
            stats.append(self.prod.attrs["flag_" + name])
        dflags['statistics'] = stats
        return dflags

    def process(self,
                output="prod"):
        '''
        Generate the flags raster and attributes

        :param output:
            - if None returns nothing but the raster is updated within the masking object
            - if "prod" returns the full raster updated with the flags variable and attributes
            - if "flags" returns xarray DataArray of the flags
              plus the dictionary of flags statistics
        :return: see :param output
        '''
        # apply the masking processors
        # TODO implement and call mask processing

        if isinstance(self.flags, xr.DataArray):
            self.prod[FLAG_NAME] = (('y', 'x'), self.flags.values)
        else:
            self.prod[FLAG_NAME] = (('y', 'x'), self.flags)

        # set the attributes with the names and description of the flags
        self.prod[FLAG_NAME].attrs['flag_descriptions'] = self.flag_descriptions.astype(str)
        self.prod[FLAG_NAME].attrs['flag_names'] = self.flag_names.astype(str)

        # compute flag statistics over the image
        self.get_stats()
        self.prod.attrs.update(self.flag_stats)

        if output == "prod":
            return self.prod
        elif output == "flags":
            flags = self.prod[FLAG_NAME]
            flags.attrs.update(self.flag_stats)
            return flags

    @staticmethod
    def create_mask(flags,
                    tomask=[0, 2],
                    tokeep=[3],
                    mask_name="mask",
                    _type=np.uint8
                    ):
        '''
        Create binary mask from bitmask flags, with selection of bitmask to mask or to keep (by bit number).
        The masking convention is: good pixels for mask == 0, bad pixels when mask == 1

        :param flags: xarray dataarray with bitmask flags
        :param tomask: array of bitmask flags used to mask
        :param tokeep: array of bitmask flags for which pixels are kept (= good quality)
        :param mask_name: name of the output mask
        :param _type: type of the array (uint8 is recommended)
        :return: mask

        Example of output mask

        >>> mask = create_mask(raster.flags,
        ...                    tomask = [0,2,11],
        ...                    tokeep = [3],
        ...                    mask_name="mask_from_flags" )
        <xarray.DataArray>
        'mask_from_flags'
        y: 5490x: 5490
        array([[1, 1, 1, ..., 1, 1, 1],
               [1, 1, 1, ..., 1, 1, 1],
               [1, 1, 1, ..., 1, 1, 1],
               ...,
               [0, 0, 0, ..., 1, 1, 1],
               [0, 0, 0, ..., 1, 1, 1],
               [0, 0, 0, ..., 1, 1, 1]], dtype=uint8)
        Coordinates:
            x           (x) float64 6e+05 6e+05 ... 7.098e+05 7.098e+05
            y           (y) float64 4.9e+06 4.9e+06 ... 4.79e+06
            spatial_ref () int64 0
            time        () datetime64[ns] 2021-05-12T10:40:21
            band        () int64 1
        Indexes: (2)
        Attributes:
        long_name:   binary mask from flags
        description: good pixels for mask == 0, bad pixels when mask == 1

        '''

        mask = xr.zeros_like(flags, dtype=_type)

        flag_value_tomask = 0
        flag_value_tokeep = 0

        if len(tomask) > 0:
            for bitnum in tomask:
                flag_value_tomask += 1 << bitnum

        if len(tokeep) > 0:
            for bitnum in tokeep:
                flag_value_tokeep += 1 << bitnum

        if (len(tokeep) > 0) & (len(tomask) > 0):
            mask = (((flags & flag_value_tomask) != 0) | ((flags & flag_value_tokeep) == 0)).astype(_type)
        elif (len(tokeep) > 0) | (len(tomask) > 0):
            if len(tokeep) > 0:
                mask = ((flags & flag_value_tokeep) == 0)
            else:
                mask = ((flags & flag_value_tomask) != 0)
        mask.attrs["long_name"] = "binary mask from flags"
        mask.attrs["description"] = "good pixels for mask == 0, bad pixels when mask == 1"
        mask.name = mask_name
        return mask
