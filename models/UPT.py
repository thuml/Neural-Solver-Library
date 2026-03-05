import os
import sys
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from layers.Embedding import unified_pos_embedding
from layers.UPT_Blocks import RansPerceiver_Encoder, TransformerModel, RansPerceiver_Decoder


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'UPT'
        self.args = args

        self.input_shape = (None, self.args.space_dim + self.args.fun_dim)
        self.output_shape = (None, self.args.out_dim)

        if args.unified_pos and args.geotype != 'unstructured':  # only for structured mesh
            self.pos = unified_pos_embedding(args.shapelist, args.ref)
            self.input_shape = (None, self.args.fun_dim + self.args.ref ** len(args.shapelist))

        # mesh_encoder
        self.mesh_encoder = RansPerceiver_Encoder(num_output_tokens=args.num_output_tokens,
                                                 add_type_token=False,
                                                 init_weights='truncnormal',
                                                 dim=args.n_hidden,
                                                 num_attn_heads=args.n_heads,
                                                 input_shape=self.input_shape,
                                                 fun_dim=self.args.fun_dim,
                                                 args=args
                                                 )
        # latent
        self.latent = TransformerModel(init_weights='truncnormal',
                                        drop_path_rate=0.3,
                                        drop_path_decay=False,
                                        dim=args.n_hidden,
                                        num_attn_heads=args.n_heads,
                                        depth=args.n_layers,
                                        input_shape=self.mesh_encoder.output_shape,
                                        condition_dim=None
                                        )
        # decoder
        self.decoder = RansPerceiver_Decoder(init_weights='truncnormal',
                                            dim=args.n_hidden,
                                            num_attn_heads=args.n_heads,
                                            input_shape=self.latent.output_shape,
                                            ndim=self.input_shape[1],
                                            output_shape=self.output_shape,
                                            fun_dim=self.args.fun_dim,
                                            use_last_norm=True,
                                            args=args
                                            )
            
    def structured_geo(self, x, fx, T=None, geo=None):
        if self.args.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if fx is not None and fx.dim() == 2:
                fx = fx.unsqueeze(0)
        # encode
        embed = self.mesh_encoder(x=x, fx=fx, T=T)

        # propagate
        propagated = self.latent(embed)

        # decode
        x_hat = self.decoder(propagated, query_x=x, query_fx=fx, T=T)

        return x_hat

    def unstructured_geo(self, x, fx, T=None, geo=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if fx is not None and fx.dim() == 2:
                fx = fx.unsqueeze(0)
        # encode
        embed = self.mesh_encoder(x=x, fx=fx, T=T)

        # propagate
        propagated = self.latent(embed)

        # decode
        x_hat = self.decoder(propagated, query_x=x, query_fx=fx, T=T)

        return x_hat

    def forward(self, x, fx, T=None, geo=None):
        if self.min_max_normalizer is not None:
            x = self.min_max_normalizer.encode(x)
        if self.args.geotype == 'unstructured':
            return self.unstructured_geo(x, fx, T)
        else:
            return self.structured_geo(x, fx, T)