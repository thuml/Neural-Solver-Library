python run.py \
--gpu 6 \
--data_path /data/fno/ \
--loader ns \
--geotype structured_2D \
--task dynamic_autoregressive \
--teacher_forcing 0 \
--space_dim 2 \
--fun_dim 10 \
--out_dim 1 \
--model LSM \
--n_hidden 32 \
--n_heads 8 \
--n_layers 8 \
--slice_num 64 \
--unified_pos 1 \
--ref 8 \
--batch-size 4 \
--epochs 500 \
--eval 0 \
--save_name ns_LSM_wo_teacher_forcing