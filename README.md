# <img src=".\pic\logo.png" alt="Logo" width="30" style="vertical-align: middle;">Neural-Solver-Library (NeuralSolver)

NeuralSolver is an open-source library for deep learning researchers, especially for neural PDE solvers.

:triangular_flag_on_post:**News** (2025.10) We propose [FlashBias (NeurIPS 2025)](https://github.com/thuml/FlashBias) to support fast computation of scientific models. With FlashBias, it is easy to introduce a learnable spatial distance bias into the Transformer for better geometry learning. Check the [Transformer_Spatial_Bias](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Transformer_Spatial_Bias.py) model for details. Welcome to try FlashBias and implement your own scientific bias.

:triangular_flag_on_post:**News** (2025.03) We release the NeuralSolver as a simple and neat code base for benchmarking neural PDE solvers, which is extended from our previous GitHub repository [Transolver](https://github.com/thuml/Transolver).

## Features

This library currently supports the following benchmarks:

- Six Standard Benchmarks from [[FNO]](https://arxiv.org/abs/2010.08895) and [[geo-FNO]](https://arxiv.org/abs/2207.05209)
- PDEBench [[NeurIPS 2022 Track Datasets and Benchmarks]](https://arxiv.org/abs/2210.07182) for benchmarking autoregressive tasks
- ShapeNet-Car from [[TOG 2018]](https://dl.acm.org/doi/abs/10.1145/3197517.3201325) for benchmarking industrial design tasks

<p align="center">
<img src=".\pic\task.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> Examples of supported PDE-solving tasks.
</p>

## Supported Neural Solvers

Here is the list of supported neural PDE solvers:

- [x] **Transolver** - Transolver: A Fast Transformer Solver for PDEs on General Geometries [[ICML 2024]](https://arxiv.org/abs/2402.02366) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Transolver.py)
- [x] **UPT** - Universal Physics Transformers: A Framework For Efficiently Scaling Neural Operators [[NeurIPS 2024]](https://arxiv.org/abs/2402.12365) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/UPT.py)
- [x] **ONO** - Improved Operator Learning by Orthogonal Attention [[ICML 2024]](https://arxiv.org/abs/2310.12487v3) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/ONO.py)
- [x] **Factformer** - Scalable Transformer for PDE Surrogate Modeling [[NeurIPS 2023]](https://arxiv.org/abs/2305.17560) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Factformer.py)
- [x] **U-NO** - U-NO: U-shaped Neural Operators [[TMLR 2023]](https://openreview.net/pdf?id=j3oQF9coJd) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/U_NO.py)
- [x] **LSM** - Solving High-Dimensional PDEs with Latent Spectral Models [[ICML 2023]](https://arxiv.org/pdf/2301.12664) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/LSM.py)
- [x] **GNOT** - GNOT: A General Neural Operator Transformer for Operator Learning [[ICML 2023]](https://arxiv.org/abs/2302.14376) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/GNOT.py)
- [x] **F-FNO** - Factorized Fourier Neural Operators [[ICLR 2023]](https://arxiv.org/abs/2111.13802) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/F_FNO.py)
- [x] **U-FNO** - An enhanced Fourier neural operator-based deep-learning model for multiphase flow [[Advances in Water Resources 2022]](https://www.sciencedirect.com/science/article/pii/S0309170822000562) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/U_FNO.py)
- [x] **Galerkin Transformer** - Choose a Transformer: Fourier or Galerkin [[NeurIPS 2021]](https://arxiv.org/abs/2105.14995) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Galerkin_Transformer.py)
- [x] **MWT** - Multiwavelet-based Operator Learning for Differential Equations [[NeurIPS 2021]](https://openreview.net/forum?id=LZDiWaC9CGL) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/MWT.py)
- [x] **FNO** - Fourier Neural Operator for Parametric Partial Differential Equations [[ICLR 2021]](https://arxiv.org/pdf/2010.08895) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/FNO.py)
- [x] **Transformer** - Attention Is All You Need [[NeurIPS 2017]](https://arxiv.org/pdf/1706.03762) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Transformer.py)

Some vision backbones can be good baselines for tasks in structured geometries:

- [x] **Swin Transformer** - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [[ICCV 2021]](https://arxiv.org/abs/2103.14030) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Swin_Transformer.py)
- [x] **U-Net** - U-Net: Convolutional Networks for Biomedical Image Segmentation [[MICCAI 2015]](https://arxiv.org/pdf/1505.04597) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/U_Net.py)

Some classical geometric deep  models are also included for design tasks:

- [x] **Graph-UNet** - Graph U-Nets [[ICML 2019]](https://arxiv.org/pdf/1905.05178) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Graph_UNet.py)
- [x] **GraphSAGE** - Inductive Representation Learning on Large Graphs [[NeurIPS 2017]](https://arxiv.org/pdf/1706.02216) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/GraphSAGE.py)
- [x] **PointNet** - PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation [[CVPR 2017]](https://arxiv.org/pdf/1612.00593) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/PointNet.py)

🌟 We have made a great effort to ensure good reproducibility, and are glad to claim that the official results of all the above methods can be completely reproduced (sometimes even better) by this library. 

## Usage

1. Install Python 3.8. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the data from our previous repository [here](https://github.com/thuml/Transolver).
3. Train and evaluate the model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as follows:

```bash
bash ./scripts/StandardBench/airfoil/Transolver.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transolver.py`.
- Include the newly added model in the `model_dict` of `./models/model_factory.py`.
- Create the corresponding scripts under the folder `./scripts`, where you can set hyperparameters following the provided scripts of other models.

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{wu2024Transolver,
  title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
  author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## Contact

If you have any questions or want to use the code, please contact our team or describe it in Issues.

Current maintenance team:

- Haixu Wu (Ph.D., wuhaixu98@gmail.com)
- Yuanxu Sun (Master student, sunyuanx22@mails.tsinghua.edu.cn)
- Hang Zhou (Ph.D. student, zhou-h23@mails.tsinghua.edu.cn)
- Yuezhou Ma (Ph.D. student, mayz24@mails.tsinghua.edu.cn)
- Haonan Shangguan (Ph.D. student, sghn25@mails.tsinghua.edu.cn)

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code base or datasets:

https://github.com/thuml/Transolver

https://github.com/thuml/Latent-Spectral-Models

https://github.com/neuraloperator/neuraloperator

https://github.com/neuraloperator/Geo-FNO
