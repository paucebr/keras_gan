#!/bin/bash

# execute with ./run free_gpu_device experiment_name
# check free gpu devices with nvidia-smi
# dont forget to change configuration file and user paths to save experiments

CUDA_VISIBLE_DEVICES=$1 python train.py -c config/camvid_adversarial_semseg.py -e $2 -s /home/pcebrian/experiments -l /home/pcebrian/experiments
