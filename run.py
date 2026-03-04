import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import *

parser = argparse.ArgumentParser('Training Neural PDE Solvers')

## training
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epochs', type=int, default=500, help='maximum epochs')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
parser.add_argument('--pct_start', type=float, default=0.3, help='oncycle lr schedule')
parser.add_argument('--batch-size', type=int, default=8, help='batch size')
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None, help='make the training stable')
parser.add_argument('--derivloss', type=bool, default=False, help='adopt the spatial derivate as regularization')
parser.add_argument('--teacher_forcing', type=int, default=1,
                    help='adopt teacher forcing in autoregressive to speed up convergence')
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type, select from Adam, AdamW')
parser.add_argument('--scheduler', type=str, default='OneCycleLR',
                    help='learning rate scheduler, select from [OneCycleLR, CosineAnnealingLR, StepLR]')
parser.add_argument('--step_size', type=int, default=100, help='step size for StepLR scheduler')
parser.add_argument('--gamma', type=float, default=0.5, help='decay parameter for StepLR scheduler')

## data
parser.add_argument('--data_path', type=str, default='/data/fno/', help='data folder')
parser.add_argument('--loader', type=str, default='airfoil', help='type of data loader')
parser.add_argument('--train_ratio', type=float, default=0.8, help='training data ratio')
parser.add_argument('--ntrain', type=int, default=1000, help='training data numbers')
parser.add_argument('--ntest', type=int, default=200, help='test data numbers')
parser.add_argument('--normalize', type=bool, default=False, help='make normalization to output')
parser.add_argument('--norm_type', type=str, default='UnitTransformer',
                    help='dataset normalize type. select from [UnitTransformer, UnitGaussianNormalizer]')
parser.add_argument('--geotype', type=str, default='unstructured',
                    help='select from [unstructured, structured_1D, structured_2D, structured_3D]')
parser.add_argument('--time_input', type=bool, default=False, help='for conditional dynamic task')
parser.add_argument('--space_dim', type=int, default=2, help='position information dimension')
parser.add_argument('--fun_dim', type=int, default=0, help='input observation dimension')
parser.add_argument('--out_dim', type=int, default=1, help='output observation dimension')
parser.add_argument('--shapelist', type=list, default=None, help='for structured geometry')
parser.add_argument('--downsamplex', type=int, default=1, help='downsample rate in x-axis')
parser.add_argument('--downsampley', type=int, default=1, help='downsample rate in y-axis')
parser.add_argument('--downsamplez', type=int, default=1, help='downsample rate in z-axis')
parser.add_argument('--radius', type=float, default=0.2, help='for construct geometry')

## task
parser.add_argument('--task', type=str, default='steady',
                    help='select from [steady, dynamic_autoregressive, dynamic_conditional]')
parser.add_argument('--T_in', type=int, default=10, help='for input sequence')
parser.add_argument('--T_out', type=int, default=10, help='for output sequence')

## models
parser.add_argument('--model', type=str, default='Transolver')
parser.add_argument('--n_hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n_layers', type=int, default=3, help='layers')
parser.add_argument('--n_heads', type=int, default=4, help='number of heads')
parser.add_argument('--act', type=str, default='gelu')
parser.add_argument('--mlp_ratio', type=int, default=1, help='mlp ratio for feedforward layers')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--unified_pos', type=int, default=0, help='for unified position embedding')
parser.add_argument('--ref', type=int, default=8, help='number of reference points for unified pos embedding')

## model specific configuration
parser.add_argument('--slice_num', type=int, default=32, help='number of physical states for Transolver')
parser.add_argument('--modes', type=int, default=12, help='number of basis functions for LSM and FNO')
parser.add_argument('--psi_dim', type=int, default=8, help='number of psi_dim for ONO')
parser.add_argument('--attn_type', type=str, default='nystrom',help='attn_type for ONO, select from nystrom, linear, selfAttention')
parser.add_argument('--mwt_k', type=int, default=3,help='number of wavelet basis functions for MWT')
parser.add_argument('--standardize_query_pos', type=bool, default=False, help='standardize query pos in UPT')   
parser.add_argument('--num_output_tokens', type=int, default=1024, help='number of output tokens for perceiver in UPT')

## eval
parser.add_argument('--eval', type=int, default=0, help='evaluation or not')
parser.add_argument('--save_name', type=str, default='Transolver_check', help='name of folders')
parser.add_argument('--vis_num', type=int, default=10, help='number of visualization cases')
parser.add_argument('--vis_bound', type=int, nargs='+', default=None, help='size of region for visualization, in list')

args = parser.parse_args()
eval = args.eval
save_name = args.save_name
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def main():
    if args.task == 'steady':
        from exp.exp_steady import Exp_Steady
        exp = Exp_Steady(args)
    elif args.task == 'steady_design':
        from exp.exp_steady_design import Exp_Steady_Design
        exp = Exp_Steady_Design(args)
    elif args.task == 'dynamic_autoregressive':
        from exp.exp_dynamic_autoregressive import Exp_Dynamic_Autoregressive
        exp = Exp_Dynamic_Autoregressive(args)
    elif args.task == 'dynamic_conditional':
        from exp.exp_dynamic_conditional import Exp_Dynamic_Conditional
        exp = Exp_Dynamic_Conditional(args)
    else:
        raise NotImplementedError

    if eval:
        exp.test()
    else:
        exp.train()
        exp.test()


if __name__ == "__main__":
    main()
