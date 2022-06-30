if [ ! -d "checkpoints" ]; then
mkdir checkpoints
fi
python train_transh.py
python ../utils/gen_result.py