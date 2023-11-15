@echo off
call conda activate myenv
set CUDA_VISIBLE_DEVICES=0
set seq_len=168
set model=PatchTST

for %%a in (100) do (
    for %%b in (96) do (
        python main.py ^
            --root_path ./ ^
            --data_path ttf.data ^
            --model_id ETTh2_%model%_%gpt_layer%_%seq_len%_%%b_%%a ^
            --data ttf ^
            --seq_len %seq_len% ^
            --label_len 168 ^
            --pred_len %%b ^
            --batch_size 32 ^
            --decay_fac 0.5 ^
            --learning_rate 0.001 ^
            --train_epochs 10 ^
            --d_model 168 ^
            --n_heads 4 ^
            --d_ff 256 ^
            --dropout 1 ^
            --enc_in 7 ^
            --c_out 7 ^
            --freq 0 ^
            --patch_size 16 ^
            --stride 8 ^
            --percent %%a ^
            --gpt_layer 6 ^
            --itr 1 ^
            --model %model% ^
            --cos 1 ^
            --tmax 20 ^
            --pretrain 0 ^
            --is_gpt 0 ^
            --target y ^
	        --loss_func mse ^
            --num_classes 2
    )
)
