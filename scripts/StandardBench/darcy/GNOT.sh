python run.py \
--gpu 6 \
--data_path /data/fno \
--loader darcy \
--geotype structured_2D \
--task steady \
--normalize 1 \
--derivloss 1 \
--downsamplex 5 \
--downsampley 5 \
--space_dim 2 \
--fun_dim 1 \
--out_dim 1 \
--model GNOT \
--n_hidden 128 \
--n_heads 8 \
--n_layers 8 \
--mlp_ratio 2 \
--slice_num 64 \
--unified_pos 1 \
--ref 8 \
--batch-size 4 \
--epochs 500 \
--eval 0 \
--save_name darcy_GNOT