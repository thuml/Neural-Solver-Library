python run.py \
--gpu 2 \
--data_path /data/fno/pipe \
--loader pipe \
--geotype structured_2D \
--space_dim 2 \
--fun_dim 0 \
--out_dim 1 \
--model LSM \
--n_hidden 32 \
--n_heads 8 \
--n_layers 8 \
--slice_num 64 \
--unified_pos 0 \
--ref 8 \
--batch-size 4 \
--epochs 500 \
--eval 0 \
--normalize 1 \
--save_name pipe_LSM