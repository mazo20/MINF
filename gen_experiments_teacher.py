#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/deeplab/datasets/data'
base_call = (f"python code/main.py  --data_root {DATA_HOME}/input --crop_val --batch_size 16  --gpu_id 0,1,2,3")

repeats = 1
strides = ['16', '32']
nets = ['deeplabv3plus_mobilenet', 'deeplabv3_mobilenet']
crop_size = ['513']


settings = [(crop, net, stride, rep) for crop in crop_size for net in nets for stride in strides
            for rep in range(repeats)]
nr_expts = len(crop_size) * len(nets) * repeats

nr_servers = 10
avg_expt_time = 300  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("experiment.txt", "w")

for i, (crop, net, stride, rep) in enumerate(settings):
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--model {net} "
        f"--crop_size {crop} "
        f"--output_stride {stride} "
        f"--random_seed {i} "
    )
    print(expt_call, file=output_file)

output_file.close()