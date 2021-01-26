#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os
import itertools

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/deeplab/datasets/data'
base_call = (f"python code/main.py  --data_root {DATA_HOME}/input --crop_val  --gpu_id 0,1,2,3 ")

repeats = 1

config = {
    '--output_stride': ['16'],
    '--model': ['deeplabv3plus_resnet50', 'deeplabv3_resnet50'],
    # '': ['--separable_conv', ''],
    '--mode': ['teacher'],
    '--batch_size': ['16'], 
    '--crop_size': ['513'] * repeats,
    # '--teacher_ckpt': ['checkpoints/best_plusmobilenet.pth'],
    
}

keys, values = zip(*config.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

nr_expts = len(permutations_dicts)
nr_servers = 10
avg_expt_time = 5  # hours
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {((round(nr_expts / nr_servers) + 1) * avg_expt_time)} hrs')

output_file = open("experiment.txt", "w")

for i, dictionary in enumerate(permutations_dicts):
    call = base_call
    for key, value in dictionary.items():
        call += " " + key + " " + value
    call += " " + "--random_seed" + " " + str(i)
    print(call, file=output_file)
    
output_file.close()