if [ ! -d "checkpoints" ]; then
mkdir checkpoints
fi
python train_complex.py
python ../utils/gen_result.py