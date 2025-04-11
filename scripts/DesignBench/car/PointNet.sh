python run.py \
--gpu 5 \
--data_path /data/PDE_data/mlcfd_data/ \
--loader car_design \
--geotype unstructured \
--task steady_design \
--space_dim 3 \
--fun_dim 7 \
--out_dim 4 \
--model PointNet \
--n_hidden 16 \
--n_heads 8 \
--n_layers 8 \
--mlp_ratio 2 \
--slice_num 32 \
--unified_pos 0 \
--ref 8 \
--batch-size 1 \
--epochs 200 \
--eval 0 \
--save_name car_design_PointNet