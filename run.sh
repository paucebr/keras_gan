#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python train.py -c config/camvid_adversarial_semseg.py -e $2 -s /home/pcebrian/experiments -l /home/pcebrian/experiments
