if [ ! -d "checkpoints" ]; then
mkdir checkpoints
fi
python train_transd.py
python ../utils/gen_result.py