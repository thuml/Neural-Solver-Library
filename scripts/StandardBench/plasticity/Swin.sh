python run.py \
--gpu 5 \
--data_path /data/fno/ \
--loader plas \
--geotype structured_2D \
--task dynamic_conditional \
--ntrain 900 \
--ntest 80 \
--T_out 20 \
--time_input 1 \
--space_dim 2 \
--fun_dim 1 \
--out_dim 4 \
--model Swin_Transformer \
--n_hidden 128 \
--n_heads 8 \
--n_layers 8 \
--slice_num 64 \
--unified_pos 0 \
--ref 8 \
--batch-size 8 \
--epochs 500 \
--eval 0 \
--save_name plas_Swin_Transformer
