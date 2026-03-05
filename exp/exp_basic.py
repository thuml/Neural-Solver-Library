import os
import torch
from models.model_factory import get_model
from data_provider.data_factory import get_data


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


class Exp_Basic(object):
    def __init__(self, args):
        self.dataset, self.train_loader, self.test_loader, args.shapelist = get_data(args)
        self.model = get_model(args).cuda()
        self.model.min_max_normalizer = getattr(self.dataset, "min_max_normalizer", None)
        self.args = args
        print(self.args)
        print(self.model)
        count_parameters(self.model)

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
