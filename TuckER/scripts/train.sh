python -u train_tucker.py \
--dataset OpenBG-IMG \
--num_iterations 500 \
--batch_size 200 \
--lr 0.0005 \
--dr 1.0 \
--edim 200 \
--rdim 200 \
--input_dropout 0.3 \
--hidden_dropout1 0.4 \
--hidden_dropout2 0.5 \
--label_smoothing 0.1

python ../utils/gen_result.py