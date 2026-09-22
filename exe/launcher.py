import yaml
import sys, os
import netCDF4 as nc
import logging
from logging.handlers import RotatingFileHandler

def get_outfile(file, output_dir, suffix, is_version_grs):
    basename = os.path.basename(file).rstrip(".nc")
    if is_version_grs:
        basename = "_".join(basename.split("_")[:-1])
    basename = basename.replace('L2AGRS', 'L2BWQ')
    basename = basename.replace('S2B', 'S2')
    basename = basename.replace('S2A', 'S2')
    return os.path.join(output_dir, basename + suffix + ".nc")


def main():
    #read config and prepare environment
    if(len(sys.argv)>1):
        config_file=sys.argv[1]
    else:
        config_file="/home/obs2co_l2bgen/exe/global_config.yml"

    with open(config_file, 'r') as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        
    
    from logging.handlers import RotatingFileHandler

    # file handle
    log_folder = os.path.dirname(data['logfile'])
    if not os.path.exists(log_folder):
            os.makedirs(log_folder)
            
    logger = logging.getLogger()
    file_handler = RotatingFileHandler(data['logfile'], 'a', 1000000, 1)
    formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d    %(levelname)s:%(filename)s::%(funcName)s:%(message)s', datefmt='%Y-%m-%dT%H:%M:%S')

    level = logging.getLevelName(data['level'])
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(data['level'])
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    
    from obs2co_l2bgen import process
    
    #get all config
    for key, value in data.items():
        if(value is not None and value!=''):
            data[key]=value 
        else:
            data[key]=None
    file = data["input_file"].rstrip("/")
    
    if not file:
        logging.error("Missing input file. Process stopped")
        exit(-1)
        
    # Verify existence of inputs
    if not os.path.isdir(file):
        logging.error("Input file doesn't exit. Process stopped")
        exit(-1)

    #prepare outfile
    suffix='_V'+str(data["chain_version"])
    output_dir = data['output_dir']
    if not output_dir:
        output_dir = "./"
    
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    outfile = get_outfile(file, output_dir, suffix, data['version_grs']) 

    # skip if already processed
    if os.path.isfile(outfile) & data["noclobber"]:
        logging.info('File ' + outfile + ' already processed; skip!')
        exit(-1)
    
    logging.info('call obs2co_l2bgen for the following parameters. File:' +
                 file + ', output file:' + outfile)
    
    try:
        process().execute(file, outfile)
        
    except Exception as inst:
        logging.info('-------------------------------')
        logging.info('error for file  ', inst, ' skip')
        logging.info('-------------------------------')
        with open(data["logfile"], "a") as myfile:
            myfile.write('error during l2bgen \n')


if __name__ == '__main__':
    main()
