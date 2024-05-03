#!/bin/bash
echo The ID of the Docker image is.............."$1"
echo The image to be processed is..............."$2"
echo The path to the image to be processed is..."$3"
echo The output file will be named.............."$4"
echo The output file will located at............"$5"

# Create and parametrize container
sudo docker run -d --name l2bgen $1
sudo docker exec l2bgen mkdir -p /home/L2A
sudo docker cp $3/$2 l2bgen:/home/L2A

# Launch processing and retreive result
sudo docker exec l2bgen obs2co_l2bgen /home/L2A/$2 -o $4
sudo docker cp l2bgen:/home/$4 $5/$4
echo Result written at location $5/$4

# Delete container
sudo docker kill l2bgen
sudo docker rm l2bgen
