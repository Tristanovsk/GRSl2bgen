'''
Module dedicated to pixel classification.
'''

import numpy as np
import xarray as xr
import pandas as pd

import logging

from s2cloudless import S2PixelCloudDetector

FLAG_NAME = 'flags'


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
