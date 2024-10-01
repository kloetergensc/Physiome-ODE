if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/forecasting" ]; then
    mkdir ./logs/forecasting
fi


seq_len=24
model_name=DLinear
pred_len=12

root_path_name=./dataset/
data_path_name=INA01
model_id_name=INA01
data=ode
channels=29
channel_independence=1

python -u run_exp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data \
    --channel_independence $channel_independence \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $channels \
    --n_heads 4 \
    --head_dropout 0 \
    --des 'Exp' \
    --train_epochs 100 \
    --itr 1 --batch_size 128 >logs/forecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$channel_independence.log 