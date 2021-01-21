#python gen_experiments.py 
run_experiment -b deeplab_arrayjob.sh -e experiment.txt \ --partition=PGR-Standard --cpus-per-task=8 --gres=gpu:4 --mem=20000 --exclude=damnii[06]
