import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from rotary_embedding_torch import RotaryEmbedding


@dataclass
class ModelConfig:
    model_type:str= 'nGPT0.5B'
    vocab_size: int = 50304
    block_size: int = 1024
    n_embd: int = None
    n_heads: int = None
    n_layers: int = None
    n_factor: int = 4
    scale: float = block_size ** (-0.5)
    bias: bool = False
    dropout: float = 0.1

@dataclass
class OptimizerConfig:
    weight_decay:float=None
    learning_rate: float=None
    betas: tuple=None
# create functions
def norm(x):
    out = x / x.norm(p=2, dim=-1, keepdim=True)
    return out


def top_p_sampling(x, top_p):
    probs_sort, probs_indices = torch.sort(x, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_indices, -1, next_token)
    return next_token


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Wu = nn.Linear(
            config.n_embd, config.n_factor * config.n_embd, bias=config.bias
        )
        self.Wv = nn.Linear(
            config.n_embd, config.n_factor * config.n_embd, bias=config.bias
        )
        self.mlp_proj = nn.Linear(
            config.n_factor * config.n_embd, config.n_embd, bias=config.bias
        )
        self.scale = config.n_embd**0.5
        self.scale_uv = 1
        self.s_u_init = 1
        self.s_u = nn.Parameter(
            self.s_u_init
            * torch.ones(config.n_factor * config.n_embd, dtype=torch.float32)
        )
        self.s_v_init = 1
        self.s_v = nn.Parameter(
            self.s_v_init
            * torch.ones(config.n_factor * config.n_embd, dtype=torch.float32)
        )

    def forward(self, x):
        scale_u = self.s_u * (self.s_u_init / self.s_uv)
        u = norm(self.Wu(x)) * scale_u
        scale_v = self.s_v * (self.s_v_init / self.s_uv) * self.scale
        v = norm(self.Wv(x)) * scale_v
        return self.mlp_proj(F.silu(v) * u)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_heads
        self.to_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.to_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.to_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.head_dim = config.n_embd // self.heads
        self.scaling = (self.head_dim) ** 0.5
        self.rotary_embedding = RotaryEmbedding(self.head_dim)

        # S_qk_ Sclae value
        self.S_qk_int = 1
        self.S_qk_scale = config.scale
        self.S_qk = nn.Parameter(
            self.S_qk_int * torch.ones(config.n_embd, dtype=torch.float32)
        )
        # check flash attention
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "mask",
                torch.tril(
                    torch.ones(config.block_size, config.block_size).view(
                        1, 1, config.block_size, config.block_size
                    )
                ),
            )

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # scale Sz
        self.s_z_init = 1
        self.s_z = nn.Parameter(
            self.s_z_init * (torch.ones(config.n_embd, dtype=torch.float32))
        )

    def forward(self, x):
        B, T, C = x.size()
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)

        # rotary_embedding
        q = self.rotary_embedding.rotate_queries_or_keys(q)
        k = self.rotary_embedding.rotate_queries_or_keys(k)

        # normalize q
        sqk = (self.S_qk * (self.S_qk_int / self.S_qk_scale)).view(
            1, self.heads, 1, C // self.heads
        )
        q = norm(q) * sqk
        k = norm(k) * sqk
        k = k * self.scaling
        if self.flash:
            attn = torch.nn.functional.scaled_dot_product_attention(
                q.to(dtype=torch.bfloat16),
                k.to(dtype=torch.bfloat16),
                v.to(dtype=torch.bfloat16),
                attn_mask=None,
                dropout_p=0,
                is_causal=True,
            )
        else:
            attn = torch.matmul(q, k.transpose(-1, -2))
            attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = attn @ v
        y = attn.to(dtype=q.dtype)
        y = y.continuous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.FFN = FFN(config)
        # alpha params
        self.alpha_a_init = 0.05
        self.alpha_a = nn.Parameter(self.alpha_a_init * torch.ones(config.n_embd))
        self.alpha_a_scale = config.scale

        self.alpha_m_init = 0.05
        self.alpha_m = nn.Parameter(self.alpha_m_init * torch.ones(config.n_embd))
        self.alpha_m_scale = config.scale

    def forward(self, x):
        x = norm(x)
        alpha_a_scale = self.alpha_a * (self.alpha_a_init / self.alpha_a_scale)

        x = torch.abs(alpha_a_scale) * (norm(self.attention(x)) - x)
        x = norm(x)
        alpha_m_scale = self.alpha_m * (self.alpha_m_init / self.alpha_m_scale)
        x = torch.abs(alpha_m_scale) * (norm(self.FFN(x)) - x)
        x = norm(x)
        return x


class nGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        self.layers = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                logits=nn.Linear(config.n_embd, config.vocab_size),
            )
        )

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=config.scale / math.sqrt(2 * config.n_layers)
                )

        # logits scale
        self.s_z_init = 1
        self.s_z_scale = config.scale
        self.s_z = nn.Parameter(self.s_z_init * torch.ones(config.vocab_size))
    def _set_model_config(self,config):
        type_given=config.model_type is not None
        params_given=all([config.n_layers is not None,config.n_heads is not None,config.n_embd is not None])
        if type_given and not params_given:
            config.__dict__.update({
                'mininGPT'  : dict(n_layers=12,n_heads=8,n_embd=768),
                'nGPT0.5B' : dict(n_layers=24,n_heads=16,n_embd=1024), #500M
                'nGPT1B'    : dict(n_layers=36,n_heads=20,n_embd=1280) #1B
            }[config.model_type])
        
    def forward(self, x, target=None):
        x = self.layers.wte(x)
        for block in self.layers.h:
            x = block(x)
        s_z = self.s_z * (self.s_z_init / self.s_z_scale)
        if target is not None:
            logits = s_z * self.layers.logits(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1), target.view(-1)), ignore_index=-1
            )
        else:
            logits = s_z * self.layers.logits(x[:, [-1], :])
            loss = None

        return logits, loss

    # calculate model params
    def get_total_params(self):
        n_params = sum(p.numel() for p in self.parameters())

        return n_params

    # initialize weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.scale)

    @torch.no_grad()
    def generate(self, txt, max_tokens=20, top_p=0.9, temperature=1):
        for i in range(max_tokens):
            txt_cond = txt if len(txt) < self.block_size else txt[:, -self.block_size]
            logits = self(txt_cond) / temperature
            probs = F.softmax(logits, dim=-1)
            idx = top_p_sampling(probs, top_p)
            txt = idx.cat([txt, idx])
        return txt

# applying weight decay
def configure_optimizers(model,config: OptimizerConfig ):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # weights with the dimention is greater than 2 will be applied weight decay, otherwise weights will dimention is smaller than 1 like bias do not apply
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"param": decay_params, "weight_decay": config.weight_decay},
        {"param": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"total numbers of decay params{num_decay_params}")
    print(f"total numbers of nodecay params {num_nodecay_params}")
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = False
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.adamw(
        optim_groups, lr=config.learning_rate, betas=config.betas, **extra_args
    )
    print(f"using AdamW: {use_fused}")
    return optimizer


model = nGPT(ModelConfig).to("cuda")
print(model.get_total_params())
