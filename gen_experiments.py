#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/deeplab/datasets/data'
base_call = (f"python main.py  --data_root {DATA_HOME}/input --year 2012_aug --crop_val --batch_size 16 --output_stride 16")

repeats = 3
#learning_rates = [10, 1, 1e-1, 1e-2]
#gammas = [.4, .5, .6, .7, .8]
nets = ['deeplabv3plus_mobilenet', 'deeplabv3_mobilenet']
crop_size = ['128', '321', '513']



settings = [(crop, net, rep) for crop in crop_size for net in nets
            for rep in range(repeats)]
nr_expts = len(crop_size) * len(nets) * repeats

nr_servers = 10
avg_expt_time = 300  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("experiment.txt", "w")

for crop, net, rep in settings:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--network {net} "
        f"--crop_size {crop}"
    )
    print(expt_call, file=output_file)

output_file.close()