
import logging
from abc import abstractmethod
from typing import Optional, override, Dict, Any, Union, Literal, Tuple, Sequence
import torch
from torch import Tensor
import torch.nn as nn
import math
from diffusers.models.attention import (
    BasicTransformerBlock, _chunked_feed_forward)
from timm.layers import DropPath
from einops import rearrange
import torch.nn.functional as F
import functools

_logger = logging.getLogger(__name__)

def _get_activation(
        activation: str, **kwargs):
    if activation == 'relu':
        return nn.ReLU(**kwargs)
    elif activation == 'gelu':
        return nn.GELU(**kwargs)
    elif activation == 'silu':
        return nn.SiLU(**kwargs)
    raise NotImplementedError(activation)

class MCDropout(nn.Dropout):
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, True, self.inplace)


class MLP(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: Optional[Union[int ,Sequence[int]]] = None,
            dropout: float = 0.1,
            mc_dropout = 0.0,
            final_mc_dropout: bool = True,
            activation: str = 'relu',
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = []
        elif not isinstance(hidden_channels, Sequence):
            hidden_channels = [hidden_channels]
        hidden_channels += [out_channels]

        norms = []
        acts = []
        projs = []
        dropouts = []
        mc_dropouts = []

        in_chans = in_channels
        for out_chans in hidden_channels:
            norms += [nn.LayerNorm(in_chans)]
            acts += [_get_activation(activation)]
            projs += [nn.Linear(in_chans, out_chans)]
            dropouts += [nn.Dropout(dropout) if dropout > 0 else nn.Identity()]
            mc_dropouts += [
                nn.Dropout(mc_dropout) if mc_dropout > 0
                else nn.Identity()
            ]

            in_chans = out_chans

        if not final_mc_dropout:
            mc_dropouts[-1] = nn.Identity()

        self.norms = nn.ModuleList(norms)
        self.acts = nn.ModuleList(acts)
        self.dropouts = nn.ModuleList(dropouts)
        self.projs = nn.ModuleList(projs)
        self.mc_dropouts = nn.ModuleList(mc_dropouts)

    def forward(self, x):
        for norm, act, proj, drop, mc_drop in zip(
                self.norms, self.acts, self.projs, self.dropouts, self.mc_dropouts):
            x = norm(x)
            x = act(x)
            x = drop(x)
            x = proj(x)
            x = mc_drop(x)
        return x


class NeuralFusor(nn.Module):
    @override
    def __init__(
            self,
            in_channels_a: int,
            in_channels_b: int,
            out_channels: int,
            hidden_channels: Optional[Union[int, Sequence[int]]] = None,
            dropout=0.1,
            mc_dropout=0.0,
            activation: str = 'relu',
    ) -> None:
        super().__init__()
        self.norm_a = nn.LayerNorm(in_channels_a)
        self.norm_b = nn.LayerNorm(in_channels_b)

        self.mlp = MLP(
            in_channels_a + in_channels_b,
            out_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            mc_dropout=mc_dropout,
            activation=activation,
        )

    def forward(
            self, feats_a: Tensor, feats_b: Tensor):
        feats_f = torch.cat(
            [self.norm_a(feats_a), self.norm_b(feats_b)],
            dim=-1)
        feats_f = self.mlp(feats_f)
        return feats_f


class FeatFusor(nn.Module):
    def __init__(
            self,
            in_channels_a: int,
            in_channels_b: int,
            out_channels: int,
            dropout=0.1,
            classifier_dropout=0.1,
            mc_dropout=0.0,
            activation: str = 'relu',
    ):
        super().__init__()
        hidden_state = round((in_channels_a + in_channels_b) / 2.)
        self.fusor = NeuralFusor(
            in_channels_a=in_channels_a,
            in_channels_b=in_channels_b,
            out_channels=hidden_state,
            dropout=dropout,
            mc_dropout=mc_dropout,
            activation=activation,
        )

        self.classifier = LinearClassifier(
            hidden_state, out_channels,
            dropout=classifier_dropout,
        )

    def forward(self, feats_a: Tensor, feats_b: Tensor) -> Tensor:
        feats_f = self.fusor(feats_a, feats_b)
        return self.classifier(feats_f)

    def to_contrastive(self, out_channels: int, mlp = True):
        self.classifier.to_contrastive(out_channels, mlp)


class FeatureTokenizer(nn.Module):
    @override
    def __init__(
            self,
            in_channels: int,
            dim: int,
            bias: bool = True,
    ) -> None:
        super().__init__()
        # (N, C)
        self.weight = nn.Parameter(
            torch.randn(in_channels, dim))
        # self.norm = nn.LayerNorm(1)
        self.norm = nn.Identity()
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(in_channels, dim))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            # self.norm_bias = nn.LayerNorm(dim)
            self.norm_bias = nn.Identity()
        else:
            self.bias = None

    def forward(self, x: Tensor):
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )
        x = self.norm(x[..., None]) * self.weight
        if self.bias is not None:
            x = x + self.norm_bias(self.bias)
        return x


class SpecialToken(nn.Module):
    def __init__(
            self,
            dim: int,
            learnable=True,
    ):
        super().__init__()
        self.dim = dim
        if learnable:
            self.token = nn.Parameter(torch.empty(1, 1, self.dim))
        else:
            self.register_buffer(
                'token', torch.empty(1, 1, self.dim))

        self._init_weights()

    def forward(
            self, batch_size: int, n_tokens: int):
        return self.token.repeat(batch_size, n_tokens, 1)

    @abstractmethod
    def _init_weights(self):
        raise NotImplementedError


class ClassToken(SpecialToken):
    def _init_weights(self):
        torch.nn.init.normal_(self.token)

class EmptyToken(SpecialToken):
    def _init_weights(self):
        torch.nn.init.zeros_(self.token)

class ConditionalPositionalEmbedding(nn.Module):
    def __init__(
            self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor):
        x = rearrange(x, 'b ... c -> b c ...')
        x = self.conv(x)
        x = rearrange(x, 'b c ... -> b ... c')
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length: int, dim: int):
        super().__init__()
        self.embedding = nn.Parameter(
            torch.zeros(1, sequence_length, dim))

    def forward(self, batch_size: int):
        return self.embedding.repeat(batch_size, 1, 1)

class GapLayer(nn.Module):
    @override
    def __init__(
            self,
            pool_type: Literal['mean', 'max', 'select'],
            select_seq_idx: Optional[int] = 0,
    ) -> None:
        super().__init__()
        if pool_type == 'select':
            assert select_seq_idx is not None

        self.pool_type = pool_type
        self.select_dim = select_seq_idx

        if self.pool_type == 'mean':
            self._gap_fn = self._avg_gap
        elif self.pool_type == 'max':
            self._gap_fn = self._max_gap
        elif self.pool_type == 'select':
            self._gap_fn = self._select_gap
        else:
            raise ValueError(f'Unsupported pool type: {self.pool_type}')

    def forward(self, x):
        return self._gap_fn(x)

    def _avg_gap(self, x):
        return x.mean(dim=1)

    def _max_gap(self, x):
        return x.max(dim=1)

    def _select_gap(self, x):
        return x[:, self.select_dim]


class TransformerLayer(BasicTransformerBlock):
    @override
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            drop_path=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "swiglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
            norm_eps: float = 1e-5,
            final_dropout: bool = False,
            attention_type: str = "default",
            positional_embeddings: Optional[str] = None,
            num_positional_embeddings: Optional[int] = None,
            ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
            ada_norm_bias: Optional[int] = None,
            ff_inner_dim: Optional[int] = None,
            ff_bias: bool = True,
            attention_out_bias: bool = True,
            selective_attn: bool = False,
            selective_attn_learn_tau: bool = False,
            selective_attn_init: Union[Literal['random'], int] = 'random',
            mc_dropout=0.0,
    ) -> None:
        super().__init__(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=attention_bias,
            only_cross_attention=only_cross_attention,
            double_self_attention=double_self_attention,
            upcast_attention=upcast_attention,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_type=norm_type,
            norm_eps=norm_eps,
            final_dropout=final_dropout,
            attention_type=attention_type,
            positional_embeddings=positional_embeddings,
            num_positional_embeddings=num_positional_embeddings,
            ada_norm_continous_conditioning_embedding_dim=ada_norm_continous_conditioning_embedding_dim,
            ada_norm_bias=ada_norm_bias,
            ff_inner_dim=ff_inner_dim,
            ff_bias=ff_bias,
            attention_out_bias=attention_out_bias,
        )
        assert norm_type == 'layer_norm', f'{norm_type=}'
        assert attention_type == 'default', f'{attention_type=}'
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0. else nn.Identity())
        self.selective_attn = selective_attn

        if self.selective_attn:
            raise NotImplementedError
            assert self.attn2 is not None, f'{self.attn2=}'
            # 0: no attention
            # 1: self-attention only
            # 2: cross-attention only
            # 3: both attentions
            self.choice_op = DifferentiableChoice(
                n_categories=4,
                temperature=0.07,
                learnable_temperature=selective_attn_learn_tau,
                init_policy=selective_attn_init,
            )
        else:
            self.choice_op = None

        self.mc_dropout = MCDropout(mc_dropout) if mc_dropout > 0 else nn.Identity()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        external_pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                _logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
        else:
            cross_attention_kwargs = {}
        # Notice that normalization is always applied before the real computation in the following blocks.

        self_attn_fun = functools.partial(
            self._calc_self_attn,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            external_pos_embed=external_pos_embed,
        )

        cross_attn_fun = functools.partial(
            self._calc_cross_attn,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            external_pos_embed=external_pos_embed,
        )

        if not self.selective_attn:
            # 0. Self-Attention
            hidden_states = self_attn_fun(hidden_states=hidden_states)
            # 3. Cross-Attention
            if self.attn2 is not None:
                hidden_states = cross_attn_fun(hidden_states=hidden_states)
        else:
            weights = self.choice_op() # (4, )
            if self.training:
                hidden_states_idt = hidden_states.clone()
                hidden_states_self = self_attn_fun(hidden_states=hidden_states.clone())
                hidden_states_cross = cross_attn_fun(hidden_states=hidden_states.clone())
                hidden_states_both = cross_attn_fun(hidden_states=hidden_states_self.clone())
                hidden_states = (
                        weights[0] * hidden_states_idt
                        + weights[1] * hidden_states_self
                        + weights[2] * hidden_states_cross
                        + weights[3] * hidden_states_both
                )
            else:
                op = weights.item() # (1,
                if op == 0:
                    hidden_states = hidden_states
                elif op == 1:
                    hidden_states = self_attn_fun(
                        hidden_states=hidden_states)
                elif op == 2:
                    hidden_states = cross_attn_fun(
                        hidden_states=hidden_states)
                elif op == 3:
                    hidden_states = cross_attn_fun(
                        hidden_states=self_attn_fun(
                            hidden_states=hidden_states)
                    )
                else:
                    raise ValueError(f'{op=}')

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        hidden_states = self.drop_path(ff_output) + hidden_states
        hidden_states = self.mc_dropout(hidden_states)
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

    def _calc_self_attn(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            external_pos_embed: Optional[torch.Tensor] = None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        if external_pos_embed is not None:
            # For LaPE: Layer-adaptive Position Embedding for Vision Transformers with Independent Layer Normalization
            # (B, L, C) + (1 or B, L, C)
            norm_hidden_states = norm_hidden_states + external_pos_embed

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        hidden_states = self.drop_path(attn_output) + hidden_states
        hidden_states = self.mc_dropout(hidden_states)

        return hidden_states

    def _calc_cross_attn(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            external_pos_embed: Optional[torch.Tensor] = None,
    ):
        norm_hidden_states = self.norm2(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)
        if external_pos_embed is not None:
            # For LaPE: Layer-adaptive Position Embedding for Vision Transformers with Independent Layer Normalization
            # (B, L, C) + (1 or B, L, C)
            norm_hidden_states = norm_hidden_states + external_pos_embed

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = self.drop_path(attn_output) + hidden_states
        hidden_states = self.mc_dropout(hidden_states)

        return hidden_states


class LPNorm(nn.Module):
    def __init__(self, dim=1, p=2, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.p = p
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, p=self.p, dim=self.dim, eps=self.eps)


class LinearClassifier(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(in_channels)
        self.dropout = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        self.proj = nn.Linear(in_channels, out_channels)
        self.is_contrastive = False

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x

    def to_contrastive(self, out_channels: int, mlp = True):
        proj = []
        if mlp:
            proj.append(nn.Linear(self.in_channels, self.in_channels))
            proj.append(nn.Linear(self.in_channels, out_channels))
        else:
            proj.append(nn.Linear(self.in_channels, out_channels))
        proj.append(LPNorm(dim=1, p=2))
        proj = nn.Sequential(*proj)
        self.proj = proj
        self.out_channels = out_channels
        self.is_contrastive = True
        return self


class TNF(nn.Module):
    @override
    def __init__(
            self,
            in_channels_a: int,
            in_channels_b: int,
            out_channels: int,
            dropout: float = 0.1,
            classifier_dropout: float = 0.1,
            mc_dropout: float = 0.0,
            predict_std: bool = False,
            fusion_model: Literal['mlp', 'dca'] = 'mlp',
            fusion_model_config: Optional[Dict[str, Any]] = None,
            use_token_addition=False,
            learnable_empty_token=False,
    ) -> None:
        super().__init__()

        self.fusion_model_name = fusion_model

        if fusion_model_config is None:
            fusion_model_config = {}

        if predict_std:
            _logger.info(f'Using Variance Prediction.')
            out_channels = out_channels * 2
        self.predict_std = predict_std
        self.using_mc_dropout = False
        if mc_dropout > 0:
            _logger.info(f'Using MC Dropout: {mc_dropout=}')
            self.using_mc_dropout = True

        self.empty_token_a = EmptyToken(
            in_channels_a, learnable=learnable_empty_token)
        self.empty_token_b = EmptyToken(
            in_channels_b, learnable=learnable_empty_token)
        self.use_token_addition = use_token_addition

        if self.fusion_model_name == 'mlp':
            self.fusor = FeatFusor(
                in_channels_a,
                in_channels_b,
                out_channels,
                dropout=dropout,
                classifier_dropout=classifier_dropout,
                mc_dropout=mc_dropout,
                **fusion_model_config,
            )
        else:
            raise NotImplementedError(self.fusion_model_name)

    def forward(
            self,
            feats_a: Tensor,
            feats_b: Optional[Tensor] = None,
            action: Optional[Literal['a', 'b', 'fusion']] = 'fusion',
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        B = len(feats_a)
        if action == 'fusion':
            assert feats_b is not None, f'{feats_b=}'
            if self.use_token_addition:
                empty_a = self.empty_token_a(B, 1).squeeze(1)
                empty_b = self.empty_token_b(B, 1).squeeze(1)
                return self.fusor(empty_a + feats_a, empty_b + feats_b)
            else:
                return self.fusor(feats_a, feats_b)
        elif action == 'a':
            assert feats_b is None, f'{type(feats_b)=}'
            empty_b = self.empty_token_b(B, 1).squeeze(1)
            if self.use_token_addition:
                empty_a = self.empty_token_a(B, 1).squeeze(1)
                return self.fusor(empty_a + feats_a, empty_b)
            else:
                return self.fusor(feats_a, empty_b)
        elif action == 'b':
            empty_a = self.empty_token_a(B, 1).squeeze(1)
            if self.use_token_addition:
                empty_b = self.empty_token_b(B, 1).squeeze(1)
                return self.fusor(empty_a, empty_b + feats_a)
            else:
                return self.fusor(empty_a, feats_a)
        raise NotImplementedError(action)

    def to_contrastive(self, out_channels: int, mlp = True):
        self.fusor.to_contrastive(out_channels, mlp)
        return self

    def load_contrastive_state_dict(
            self, contrastive_state_dict: Dict[str, Any]):
        if self.fusion_model_name == 'mlp':
            ignore_keys = [
                'fusor.classifier.proj.weight',
                'fusor.classifier.proj.bias',
            ]
        elif self.fusion_model_name == 'dca':
            ignore_keys = [
                'fusor.classifier_a.proj.weight',
                'fusor.classifier_a.proj.bias',
                'fusor.classifier_b.proj.weight',
                'fusor.classifier_b.proj.bias',
            ]
        else:
            raise NotImplementedError(self.fusion_model_name)
        ignore_keys = set(ignore_keys)

        state_dict = self.state_dict()
        for k in state_dict:
            if k in ignore_keys:
                continue
            state_dict[k] = contrastive_state_dict[k]
        self.load_state_dict(state_dict)

if __name__ == '__main__':
    with torch.no_grad():
        feats_a_channels = 2048
        feats_b_channels = 192
        out_channels = 3
        dim = 192
        n_layers = 3
        n_compress_feats = 192
        batch_size = 2

        feats_a = torch.randn(batch_size, feats_a_channels)
        feats_b = torch.rand(batch_size, feats_b_channels)

        model = TNF(
            in_channels_a=feats_a_channels,
            in_channels_b=feats_b_channels,
            out_channels=out_channels,
            dropout=0.5,
            mc_dropout=0.5,
            predict_std=False,
            fusion_model='mlp',
        )

        ya = model(feats_a, action='a')
        yb = model(feats_b, action='b')
        yf = model(feats_a, feats_b, action='fusion')
        print(ya.shape, yb.shape, yf.shape)

        # For contrastive learning
        model = model.to_contrastive(out_channels=128, mlp=True)
        ya = model(feats_a, action='a')
        yb = model(feats_b, action='b')
        yf = model(feats_a, feats_b, action='fusion')
        print(ya.shape, yb.shape, yf.shape)
