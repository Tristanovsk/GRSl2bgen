''' Executable to process Sentinel-2 L2A images into water quality paratmeters

Usage:
  GRSl2bgen <input_file> [-o <ofile>] [--odir <odir>]  [--no_clobber]
  GRSl2bgen -h | --help
  GRSl2bgen -v | --version

Options:
  -h --help        Show this screen.
  -v --version     Show version.

  <input_file>     Input file to be processed



  -o ofile         Full (absolute or relative) path to output L2 image directory.
  --odir odir      Ouput directory [default: ./]
  --no_clobber     Do not process <input_file> if <output_file> already exists.


  Example:
      img_dir=/your_path_for_image_folder
      GRSl2bgen $img_dir/S2B_MSIL2Agrs_20220731T103629_N0400_R008_T31TFJ_20220731T124834.nc -o $img_dir/L2B/S2B_MSIL2B_20220731T103629_N0400_R008_T31TFJ_20220731T124834.nc
'''

import os, sys
from docopt import docopt
import logging
from . import __package__, __version__
from . import Process


def main():
    args = docopt(__doc__, version=__package__ + '_' + __version__)
    print(args)

    file = args['<input_file>']
    noclobber = args['--no_clobber']

    ##################################
    # File naming convention
    ##################################

    outfile = args['-o']

    if outfile == None:
        outfile = file
        basename = os.path.basename(file)
        if basename[-3:] != '.nc':
            basename = basename + '.nc'
        basename = basename.replace('L2Agrs', 'L2B')
        odir = args['--odir']
        if odir == './':
            odir = os.getcwd()
    else:
        basename = os.path.basename(outfile)
        odir = os.path.dirname(outfile)

    if not os.path.exists(odir):
        os.makedirs(odir)

    outfile = os.path.join(odir, basename)

    if os.path.isfile(outfile) & noclobber:
        print('File ' + outfile + ' already processed; skip!')
        sys.exit()

    logging.info('call GRSl2bgen for the following paramater. File:' +
                 file + ', output file:' + outfile)
    process_ = Process(file, outfile)
    process_.execute()
    if process_.successful:
        process_.write_output()

    return


if __name__ == "__main__":
    main()
