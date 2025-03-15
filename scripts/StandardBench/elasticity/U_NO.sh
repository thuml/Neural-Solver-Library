python run.py \
--gpu 5 \
--data_path /data/fno/ \
--loader elas \
--geotype unstructured \
--scheduler CosineAnnealingLR \
--space_dim 2 \
--fun_dim 0 \
--out_dim 1 \
--normalize 0 \
--model U_NO \
--n_hidden 32 \
--n_heads 8 \
--n_layers 8 \
--mlp_ratio 2 \
--slice_num 64 \
--unified_pos 0 \
--ref 8 \
--batch-size 1 \
--epochs 500 \
--eval 0 \
--save_name elas_U_NO