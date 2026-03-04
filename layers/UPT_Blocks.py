import torch
from torch import nn
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import PerceiverPoolingBlock, Mlp, PerceiverBlock, DitBlock, PrenormBlock
from functools import partial
from layers.Embedding import timestep_embedding

################################################################
# UPT Mesh Encoder
################################################################

class RansPerceiver_Encoder(nn.Module):
    def __init__(
            self,
            dim,
            num_attn_heads,
            num_output_tokens,
            add_type_token=False,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
            input_shape=None,
            fun_dim=0,
            args=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attn_heads = num_attn_heads
        self.num_output_tokens = num_output_tokens
        self.add_type_token = add_type_token
        self.input_shape = input_shape
        self.fun_dim = fun_dim
        self.args = args

        # set ndim
        _, ndim = self.input_shape
        ndim = ndim - self.fun_dim

        # pos_embed
        if self.fun_dim != 0:
            self.pos_embed = ContinuousSincosEmbed(dim=dim - self.fun_dim, ndim=ndim)
        else:
            self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)

        # perceiver
        self.mlp = Mlp(in_dim=dim, hidden_dim=dim * 4, init_weights=init_weights)
        self.block = PerceiverPoolingBlock(
            dim=dim,
            num_heads=num_attn_heads,
            num_query_tokens=num_output_tokens,
            perceiver_kwargs=dict(
                init_weights=init_weights,
                init_last_proj_zero=init_last_proj_zero,
            ),
        )

        if add_type_token:
            self.type_token = nn.Parameter(torch.empty(size=(1, 1, dim,)))
        else:
            self.type_token = None

        # output shape
        self.output_shape = (num_output_tokens, dim)

        if self.args.time_input:
            self.time_fc = nn.Sequential(nn.Linear(args.n_hidden, args.n_hidden), nn.SiLU(),
                                         nn.Linear(args.n_hidden, args.n_hidden))

    def forward(self, x, fx, T=None):

        x = self.pos_embed(x)

        if self.fun_dim != 0:
            x = torch.cat([x, fx], dim=-1)
        mask = None

        if T is not None:
            Time_emb = timestep_embedding(T, self.dim).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            x = x + Time_emb

        # perceiver
        x = self.mlp(x)
        x = self.block(kv=x, attn_mask=mask)

        if self.add_type_token:
            x = x + self.type_token

        return x
    
################################################################
# UPT Latent
################################################################

class TransformerModel(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            num_attn_heads,
            drop_path_rate=0.0,
            drop_path_decay=True,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
            input_shape=None,
            condition_dim=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero
        self.input_shape = input_shape
        self.condition_dim = condition_dim

        assert len(self.input_shape) == 2
        seqlen, input_dim = self.input_shape
        self.output_shape = (seqlen, dim)

        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights)

        # blocks
        if self.condition_dim is not None:
            block_ctor = partial(DitBlock, cond_dim=self.condition_dim)
        else:
            block_ctor = PrenormBlock
        if drop_path_decay:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth
        self.blocks = nn.ModuleList([
            block_ctor(
                dim=dim,
                num_heads=num_attn_heads,
                drop_path=dpr[i],
                init_weights=init_weights,
                init_last_proj_zero=init_last_proj_zero,
            )
            for i in range(self.depth)
        ])

    def forward(self, x, condition=None, static_tokens=None):
        assert x.ndim == 3

        # concat static tokens
        if static_tokens is not None:
            x = torch.cat([static_tokens, x], dim=1)

        # input projection
        x = self.input_proj(x)

        # apply blocks
        blk_kwargs = dict(cond=condition) if condition is not None else dict()
        for blk in self.blocks:
            x = blk(x, **blk_kwargs)

        # remove static tokens
        if static_tokens is not None:
            num_static_tokens = static_tokens.size(1)
            x = x[:, num_static_tokens:]

        return x
    
################################################################
# UPT Decoder
################################################################

class RansPerceiver_Decoder(nn.Module):
    def __init__(
            self,
            dim,
            num_attn_heads,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
            use_last_norm=False,
            input_shape=None,
            ndim=None,
            output_shape=None,
            fun_dim=0,
            args=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attn_heads = num_attn_heads
        self.use_last_norm = use_last_norm
        self.input_shape = input_shape
        self.ndim = ndim - fun_dim
        self.output_shape = output_shape
        self.fun_dim = fun_dim
        self.args = args

        # input projection
        _, input_dim = self.input_shape
        self.proj = LinearProjection(input_dim, dim, init_weights=init_weights)

        # query tokens (create them from a positional embedding)
        if self.fun_dim != 0:
            self.pos_embed = ContinuousSincosEmbed(dim=dim - self.fun_dim, ndim=self.ndim)
        else:
            self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=self.ndim)
        self.query_mlp = Mlp(in_dim=dim, hidden_dim=dim, init_weights=init_weights)

        # latent to pixels
        self.perceiver = PerceiverBlock(
            dim=dim,
            num_heads=num_attn_heads,
            init_last_proj_zero=init_last_proj_zero,
            init_weights=init_weights,
        )
        _, output_dim = self.output_shape
        self.norm = nn.LayerNorm(dim, eps=1e-6) if use_last_norm else nn.Identity()
        self.pred = LinearProjection(dim, output_dim, init_weights=init_weights)

        if self.args.time_input:
            self.time_fc = nn.Sequential(nn.Linear(args.n_hidden, args.n_hidden), nn.SiLU(),
                                         nn.Linear(args.n_hidden, args.n_hidden))

    def forward(self, x, query_x, query_fx, T=None):
        # input projection
        x = self.proj(x)

        query_x = self.pos_embed(query_x)

        # create query
        if self.fun_dim != 0:
            query_x = torch.cat([query_x, query_fx], dim=-1)
        query_x = self.query_mlp(query_x)

        if T is not None:
            Time_emb = timestep_embedding(T, self.dim).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            x = x + Time_emb

        # decode
        x = self.perceiver(q=query_x, kv=x)
        x = self.norm(x)
        x = self.pred(x)

        return x