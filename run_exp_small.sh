#USAGE: bash run_exp_small.sh <gen experiment file> <name of the experiment>
#EXAMPLE: bash run_exp_small.sh gen_best.py bestModel
python3 $1 $2
mkdir -p results/$2
cp $1 results/$2
python3 $1
mv -f experiment.txt results/$2
zip -r code.zip code
mv -f code.zip results/$2



run_experiment -b deeplab_arrayjob.sh -e results/$2/experiment.txt -m 32 \ --cpus-per-task=4 --gres=gpu:1 --mem=12000 --exclude=damnii[06]
