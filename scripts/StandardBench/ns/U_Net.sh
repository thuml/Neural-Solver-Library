python run.py \
--gpu 6 \
--data_path /data/fno \
--loader ns \
--geotype structured_2D \
--task dynamic_autoregressive \
--space_dim 2 \
--fun_dim 10 \
--out_dim 1 \
--model U_Net \
--n_hidden 256 \
--n_heads 8 \
--n_layers 8 \
--mlp_ratio 2 \
--slice_num 32 \
--unified_pos 1 \
--ref 8 \
--batch-size 2 \
--epochs 500 \
--eval 0 \
--save_name ns_U_Net