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
base_call = (f"python code/main.py  --data_root {DATA_HOME}/input")

repeats = 2

config = {
    '--results_root': ['results/' + sys.argv[1]],
    '--output_stride': ['16', '32'],
    '--model': ['v3plus_resnet50', 'v3plus_mobilenet'],
    '--separable': ['none', 'grouped', 'bottleneck'],
    '--mode': ['teacher'],
    '--kernel_sharing': ['true', 'false'],
    # '--at_type': ['aspp-all', 'aspp-output', 'aspp-atrous'],
    # '--large_aspp': ['normal', 'medium', 'large', 'extra_large'],
    '--batch_size': ['16'], 
    '--crop_size': ['256'] * repeats,
    '--random_seed': ['1'],
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