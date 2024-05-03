# obs2co_l2bgen


## **Scientific code to process GRS L2A images** 
## Estimation of water quality parameters from remote sensing reflectance (Rrs) 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them

```
conda install -c conda-forge rasterio xarray dask rioxarray
```

### Installing

First, clone [the repository](https://gitlab.cnes.fr/waterquality/obs2co_l2bgen.git#) and execute the following command in the
local copy:

```
python setup.py install 
```



## Some example of the forward model:

![example files](illustration/test_Chla_OC2nasa.png)

## To generate a new kernel for Jupyter
```
python -m ipykernel install --user --name=obs2co
```

# Example
```
img_dir=/your_path_for_image_folder
obs2co_l2bgen $img_dir/ -o $img_dir/L2B/S2B_MSIL2B_20220731T103629_N0400_R008_T31TFJ_20220731T124834.nc
```

## Running the tests

### Launch with Docker on HAL:
```
qsub -q qdev -I -l walltime=4:00:00

/opt/bin/drunner run -it -v /datalake:/datalake artifactory.cnes.fr/obs2co-docker/l2bgen:1.0.2 python /home/obs2co_l2bgen/exe/launcher.py /home/obs2co_l2bgen/exe/global_config.yml
```

### Launch with Singularity on TREX:
```
module load singularity
singularity remote login -u <artifactory_user> -p <artifactory_token> docker://artifactory.cnes.fr
singularity cache clean -f
singularity pull docker://artifactory.cnes.fr/obs2co-docker/l2bgen:1.0.2

singularity exec --bind /datalake:/datalake l2bgen_1.0.2.sif python /home/obs2co_l2bgen/exe/launcher.py /home/obs2co_l2bgen/exe/global_config.yml
```

## Compile Docker image locally
First, you must get Dockerfile out of obs2co_l2bgen folder to have a structure as diplayed below.

head_folder  
 ├obs2co_l2bgen  
 └Dockerfile

Note that anything in this folder tree will be added to the Docker build context, so make it light.
You might consider removing the notebooks and all useless files and directories from the obs2co_l2bgen
folder to make the resulting image as light as possible (.git, illustration, notebook...).

Once all those requirements are met, you can compile the Docker image using the following command:
```
docker build -t obs2co_l2bgen:<version_tag> *path_to_head_folder* -f *path_to_Dockerfile*
```

When the compilation has ended, you can access the image with the command:
```
docker images
```

To run the Docker image in a container on a S2 raster you can use the run_docker.sh script as follow:
```
./run_docker.sh <image_ID> <L2A_raster_name> <L2A_raster_path> <desired_name_for_output> <desired_path_for_output>
```
Example:
```
./run_docker.sh l2bgen:V1_CNES \
S2B_L2Agrs_20220228T102849_N0400_R108_T31TFJ_20220228T123819 \
/DATA/grs_outputs \
S2B_L2Bgrs_20220228T102849_N0400_R108_T31TFJ_20220228T123819 \
/DATA/grs_outputs
```

The docker containers will be called l2bgen, which mean that you cannot currently launch multiple ones simultaneously.
You can adapt the sh script to modify this behaviour.

