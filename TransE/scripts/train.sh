if [ ! -d "checkpoints" ]; then
mkdir checkpoints
fi
python train_transe.py
python ../utils/gen_result.py