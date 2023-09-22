model_name=GPT4TS

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_mask_0.125 \
  --mask_rate 0.125 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --patch_size 1 \
  --stride 1 \
  --gpt_layer 6 \
  --d_model 768 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 3 \
  --train_epochs 30 \
  --learning_rate 0.002

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_mask_0.25 \
  --mask_rate 0.25 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 16 \
  --patch_size 1 \
  --stride 1 \
  --gpt_layer 6 \
  --d_model 768 \
  --des 'Exp' \
  --itr 3 \
  --train_epochs 30 \
  --learning_rate 0.002

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_mask_0.375 \
  --mask_rate 0.375 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 16 \
  --patch_size 1 \
  --stride 1 \
  --gpt_layer 6 \
  --d_model 768 \
  --des 'Exp' \
  --itr 3 \
  --train_epochs 30 \
  --learning_rate 0.002

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_mask_0.5 \
  --mask_rate 0.5 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 16 \
  --patch_size 1 \
  --stride 1 \
  --gpt_layer 6 \
  --d_model 768 \
  --des 'Exp' \
  --itr 3 \
  --train_epochs 30 \
  --learning_rate 0.002
