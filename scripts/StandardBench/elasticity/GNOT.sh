python run.py \
--gpu 1 \
--data_path /data/fno/ \
--loader elas \
--geotype unstructured \
--scheduler CosineAnnealingLR \
--space_dim 2 \
--fun_dim 0 \
--out_dim 1 \
--normalize 1 \
--model GNOT \
--n_hidden 128 \
--n_heads 8 \
--n_layers 8 \
--mlp_ratio 2 \
--slice_num 64 \
--unified_pos 0 \
--ref 8 \
--batch-size 1 \
--epochs 500 \
--eval 0 \
--save_name elas_GNOT