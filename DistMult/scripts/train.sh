if [ ! -d "checkpoints" ]; then
mkdir checkpoints
fi
python train_distmult.py
python ../utils/gen_result.py