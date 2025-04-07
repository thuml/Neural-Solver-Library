python run.py \
--gpu 1 \
--data_path /data/PDEBench/2D/DarcyFlow/2D_DarcyFlow_beta1.0_Train.hdf5 \
--loader pdebench_steady_darcy \
--geotype structured_2D \
--task steady \
--scheduler StepLR \
--downsamplex 1 \
--downsampley 1 \
--space_dim 2 \
--fun_dim 1 \
--out_dim 1 \
--model FNO \
--n_hidden 64 \
--n_heads 8 \
--n_layers 8 \
--unified_pos 1 \
--ref 8 \
--batch-size 50 \
--epochs 500 \
--eval 0 \
--ntrain 8000 \
--save_name pdebench_darcy_FNO