#USAGE: bash run_exp.sh <gen experiment file> <name of the experiment>
#EXAMPLE: bash run_exp.sh gen_best.py bestModel
python3 $1 $2
mkdir -p results/$2
cp $1 results/$2
python3 $1
mv -f experiment.txt results/$2
zip -r code.zip code
mv -f code.zip results/$2



#run_experiment -b deeplab_arrayjob.sh -e results/$2/experiment.txt \ --partition=PGR-Standard --cpus-per-task=8 --gres=gpu:4 --mem=20000 --exclude=damnii[06]
