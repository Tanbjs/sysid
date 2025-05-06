# DeepKoopman Operator Research

This project is dedicated to the research and development of the DeepKoopman operator. The DeepKoopman operator is a mathematical framework for analyzing dynamical systems using data-driven approaches.

# Build docker 

sudo docker build -t tf-gpu .

# Run docker and mount host to container

sudo docker run --rm --runtime=nvidia --gpus all -v .:/home/thesis_ws/sysid:rw,rshared -it tf-gpu bash
