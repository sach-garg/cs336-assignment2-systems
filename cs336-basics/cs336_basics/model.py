from __future__ import annotations

import json
import logging
import math
import os
import warnings

import einx
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.nn_utils import softmax

logger = logging.getLogger(__name__)


# class Linear(nn.Module):
#     def __init__(self, d_in: int, d_out: int):
#         """A linear layer initialized with truncated normal fan-in fan-out.

#         Args:
#             d_in: int
#                 The number of input features.
#             d_out: int
#                 The number of output features.
#         """

#         super().__init__()
#         std = math.sqrt(2 / (d_in + d_out))
#         self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
#             nn.init.trunc_normal_(torch.empty(d_out, d_in), std=std, a=-3 * std, b=3 * std), requires_grad=True
#         )

#     def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
#         return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

#     def extra_repr(self):
#         return f"d_out={self.weight.shape[0]}, d_in={self.weight.shape[1]}"

""" In my Linear, I changed name of variables (from A1) and added extra_repr function"""


class Linear(nn.Module):
    def __init__(self, d_in:int, d_out:int, device=None, dtype = None):
        super().__init__()
        self.device = device if device is not None else "cpu"
        self.weight = nn.Parameter(torch.empty(d_out,d_in,device=self.device,dtype=dtype)) ### always define in (d_out, d_in format)
        self.sigma = math.sqrt(2/(d_in+d_out))
        nn.init.trunc_normal_(self.weight, mean=0., std=self.sigma, a=-3*self.sigma, b=3*self.sigma)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return x@self.weight.transpose(0,1)

    def extra_repr(self):
            return f"d_out={self.weight.shape[0]}, d_in={self.weight.shape[1]}"

    

# class Embedding(nn.Module):
#     def __init__(self, vocab_size: int, d_model: int):
#         super().__init__()
#         std = 1.0
#         self.weight = nn.Parameter(
#             nn.init.trunc_normal_(torch.empty(vocab_size, d_model), std=std, a=-3 * std, b=3 * std), requires_grad=True
#         )

#     def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
#         return self.weight[token_ids, :]

#     def extra_repr(self):
#         return f"vocab_size={self.weight.shape[0]}, d={self.weight.shape[1]}"

""" In my Embedding, change the name from self.token_enbedding to self.weight and add extra_repr"""

class Embedding(nn.Module):
    def __init__(self,vocab_size:int,d_model:int,device=None,dtype=None):
        super().__init__()
        self.device = device if device is not None else "cpu"
        self.weight = nn.Parameter(torch.empty(vocab_size,d_model,device = self.device, dtype = dtype))
        self.sigma = 1
        nn.init.trunc_normal_(self.weight, mean=0., std=self.sigma, a=-3*self.sigma, b=3*self.sigma)
    
    def forward(self, token_ids:torch.Tensor):
        return self.weight[token_ids] ## (B,T) -> (B,T,C)

    def extra_repr(self):
            return f"vocab_size={self.weight.shape[0]}, d={self.weight.shape[1]}"




# class RMSNorm(nn.Module):
#     """
#     This module implements root mean square layer normalization, as
#     described in Eq. 4 of https://arxiv.org/abs/1910.07467

#     Args:
#         hidden_size: int
#             Dimensionality of the input to normalize.
#         eps: float, default is 1e-5
#             A value added to the denominator for numerical stability.

#     Returns:
#         FloatTensor of same shape as input.
#     """

#     def __init__(
#         self,
#         hidden_size: int,
#         eps: float = 1e-5,
#         device=None,
#     ):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
#         self.eps = eps

#     def forward(self, x):
#         """
#         Args:
#             x: FloatTensor of shape `(batch_size, *)`.
#                 The input to apply root mean square layer normalization on.

#         Returns:
#             FloatTensor of same shape as input
#         """
#         # NOTE: in practice, many implementations will
#         # manually upcast the input to fp32 here to prevent overflow when you
#         # square the input.
#         # https://github.com/pytorch/pytorch/issues/66707
#         in_dtype = x.dtype

#         x = x.to(torch.float32)
#         rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#         x = x * rms

#         return (self.weight * x).to(in_dtype)

    # def extra_repr(self):
    #     return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"

""" From my RMS Norm (A1), the change is: d_model -> hidden_size, gamma -> weight. Added extra_repr"""

class RMSNorm(nn.Module): ## cheaper than layer_norm because no mean calculation and subtraction, and also bias parameter. Usually recentering done by layer normalization is not required. 
    ### RMS Normalization is done to control the norm of x such that it doesn't saturate the activations.
    ### If gamma is all ones, then RMS norm makes the norm of x around \sqrt{d}. Note that if x was standard normally distributed, then \E\|\x\|^2 =d
    ### Also layer-norm without \beta parameter, is just RMSNorm on (\x-\mu)
    ## Layernorm(\x,\gamma, \beta=0) = RMS Norm(x-\E\x, \gamma) where \E\x in mean of all entries in \x
    def __init__(self,hidden_size:int,eps: float = 1e-5, device=None,dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size,device=device,dtype=dtype)) ### not torch.empty or torch.zeros.
        self.eps = eps
        self.d_model = hidden_size
    
    def forward(self,x:torch.Tensor):  ### x is [B,T,C]
        in_dtype = x.dtype
        x = x.to(torch.float32)
        fac = torch.rsqrt(torch.mean(x**2,dim=-1,keepdim=True)+ self.eps) #### keeping the last dimension here for broadcasting, fac becomes [B,T,1]
        return (x*fac * self.weight).to(in_dtype)
    def extra_repr(self):
            return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"




# class RotaryEmbedding(nn.Module):
#     def __init__(self, context_length: int, dim: int, theta: float = 10000.0):
#         super().__init__()
#         self.register_buffer(
#             "_freq_cis_cache", RotaryEmbedding._init_cache(context_length, dim, theta), persistent=False
#         )
#         self._freq_cis_cache: Float[Tensor, "2 context_length half_dim"]

#     @staticmethod
#     def _init_cache(context_length: int, dim: int, theta: float) -> Float[Tensor, " 2 context_length half_dim"]:
#         assert dim % 2 == 0

#         d = torch.arange(0, dim, 2) / dim
#         freqs = torch.tensor(theta) ** -d
#         t = torch.arange(context_length)

#         freqs = einsum(t, freqs, "t, f -> t f")

#         cos, sin = torch.cos(freqs), torch.sin(freqs)
#         return torch.stack((cos, sin))

#     def forward(
#         self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"] | None
#     ) -> Float[Tensor, " ... seq d"]:
#         x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2).unbind(0)

#         # Standard
#         # cos, sin = self._freq_cis_cache[:, pos_ids, :]

#         # einx
#         if pos_ids is not None:
#             cos, sin = einx.get_at("cos_sin [pos] half_dim, ... -> cos_sin ... half_dim", self._freq_cis_cache, pos_ids)
#         else:
#             seq_len = x.size(-2)
#             cos, sin = self._freq_cis_cache[:, :seq_len, :].unbind(0)

#         # 2D rotation matrix applied to pairs in x
#         x1_rot = cos * x1 - sin * x2
#         x2_rot = sin * x1 + cos * x2
#         # result = einx.id("... x_half, ... x_half -> ... (x_half (1 + 1))", x1_rot, x2_rot).contiguous()
#         result = torch.concat((x1_rot, x2_rot), dim=-1)
#         return result

#     def extra_repr(self):
#         return f"context_length={self._freq_cis_cache.shape[0]}, dim/2={self._freq_cis_cache.shape[1]}"


""" Changes in RotaryEmbedding: RoPE - > RotaryEmbedding, T -> context_length, dh ->dim, base->theta.
In forward: Q->x, TP -> pos_ids, changed logic of pos_ids not None to align with the pos_ids=None code. Added extra_repr method"""

class RotaryEmbedding(nn.Module):
    def __init__(self, context_length:int, dim:int, theta: float =10000, device=None):
        super().__init__()
        if dim%2!=0:
            raise ValueError("RoPE is defined only for even dimensions")
        self.device = device if device is not None else "cpu"
        self.theta = theta
        theta_dim = torch.tensor([1/theta**(2*(k-1)/dim) for k in range(1,int(dim/2)+1)],device=self.device) #<- [D/2]
        pos = torch.arange(context_length,device=self.device) #<-[T]
        theta_pos_dim = torch.outer(pos,theta_dim) #<-[T,D/2]
        cos = theta_pos_dim.cos()
        sin = theta_pos_dim.sin()
        
        self.register_buffer('cos',cos) # [T,D/2]
        self.register_buffer('sin',sin)  # [T,D/2]
                
        self.register_buffer('theta_dim',theta_dim)


    def forward(self,x:torch.Tensor,pos_ids:torch.Tensor | None=None): ### Parameter could be K also Q:Query, K: Key
        
        T = x.shape[-2]
        x_even = x[...,0::2] ### a view so no memory cost
        x_odd = x[...,1::2] ### a view so no memory cost
        out = torch.empty_like(x)

      
        if pos_ids is not None: ## for tensors just writing if TP doesn't work
            theta_pos_dim1 = pos_ids.unsqueeze(-1) * self.theta_dim ### [... T] -->(unsqueeze) [... T,1] ---> (*) [... T, dh/2]
            cos1 =  theta_pos_dim1.cos()
            sin1 = theta_pos_dim1.sin()
            out[..., 0::2] = x_even*cos1 - x_odd*sin1
            out[...,1::2] = x_even*sin1 + x_odd*cos1
            return out
      

        out[..., 0::2] = x_even*self.cos[:T] - x_odd*self.sin[:T]
        out[...,1::2] = x_even*self.sin[:T] + x_odd*self.cos[:T]
        return out
    
    def extra_repr(self):
            return f"context_length={self.cos.shape[0]}, dim/2={self.cos.shape[1]}"




# class BasicsTransformerLM(nn.Module):
#     """A Transformer language model.

#     Args:
#         vocab_size: int
#             The number of unique items in the output vocabulary to be predicted.
#         context_length: int,
#             The maximum number of tokens to process at once.
#         d_model: int
#             The dimensionality of the model embeddings and sublayer outputs.
#         num_layers: int
#             The number of Transformer layers to use.
#         num_heads: int
#             Number of heads to use in multi-headed attention. `d_model` must be
#             evenly divisible by `num_heads`.
#         d_ff: int
#             Dimensionality of the feed-forward inner layer (section 3.3).

#     Returns:
#         FloatTensor of shape (batch size, sequence_length, vocab_size) with the
#         predicted unnormalized next-word distribution for each token.
#     """

#     def __init__(
#         self,
#         vocab_size: int,
#         context_length: int,
#         d_model: int,
#         num_layers: int,
#         num_heads: int,
#         d_ff: int,
#         rope_theta: float | None = 10_000.0,
#     ):
#         # Store the model configuration for serialization / deserialization
#         self.config = {
#             k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
#         }
#         super().__init__()
#         self.context_length = context_length
#         self.d_model = d_model
#         self.token_embeddings = Embedding(vocab_size, d_model)
#         d_head = d_model // num_heads
#         self.positional_encoder = (
#             RotaryEmbedding(context_length, d_head, rope_theta) if rope_theta is not None else None
#         )

#         self.layers = nn.ModuleList(
#             [
#                 TransformerBlock(
#                     d_model=d_model,
#                     num_heads=num_heads,
#                     d_ff=d_ff,
#                     positional_encoder=self.positional_encoder,
#                 )
#                 for _ in range(num_layers)
#             ]
#         )
#         self.ln_final = RMSNorm(d_model)
#         self.lm_head = Linear(d_model, vocab_size)
#         # Tie the weights, since the paper mentions that "we share the same weight
#         # matrix between the two embedding layers and the pre-softmax linear transformation"
#         # self.lm_head.weight = self.token_embeddings.weight
#         # report number of parameters
#         logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")

#     def get_num_params(self) -> int:
#         """
#         Return the number of parameters in the model.
#         For non-embedding count (default), the position embeddings get subtracted.
#         The token embeddings would too, except due to the parameter sharing these
#         params are actually used as weights in the final layer, so we include them.
#         """
#         n_params = sum(p.numel() for p in self.parameters())
#         return n_params

#     def forward(self, x: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length vocab_size"]:
#         """
#         Args:
#             x: Input IDs for language modeling.

#         Returns: A FloatTensor of shape
#             (batch size, sequence_length, vocab_size) with the predicted unnormalized next-word
#             distribution for each token.
#         """
#         _, sequence_length = x.size()
#         # (batch size, sequence_length, d_model)
#         # NOTE: paper mentions "In the embedding layers, we multiply those
#         # weights by sqrt(d_model)", but we aren't doing that here.
#         embedded_tokens = self.token_embeddings(x)

#         # (batch size, sequence_length, d_model)
#         # x = self.positional_encoder(embedded_tokens, positions)
#         x = embedded_tokens

#         for layer in self.layers:
#             # (batch size, sequence_length, d_model)
#             x = layer(x)
#         # (batch size, sequence_length, d_model)
#         x = self.ln_final(x)
#         # (batch size, sequence_length, vocab_size)
#         logits = self.lm_head(x)
#         return logits

#     @torch.no_grad()
#     def generate(
#         self,
#         x: torch.Tensor,
#         max_new_tokens: int,
#         temperature: float = 1.0,
#         top_k: int | None = None,
#         eos_token_id: int | None = None,
#     ):
#         """
#         Args:
#             x: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
#                 Input IDs to condition on when generating.
#             max_new_tokens: int
#                 Maximum number of tokens to generate.
#             temperature: float
#                 Temperature to use during generation.
#             top_k: int
#                 If provided, only sample from the `top_k` vocab items (by probability).
#             eos_token_id: int
#                 If provided, stop generation when we generate this ID.

#         Returns: A LongTensor of shape (max_new_tokens,) with the generated model output.
#         """
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#         original_sequence_length = x.size(-1)
#         for _ in range(max_new_tokens):
#             # Take the last `context_length` tokens if the input is
#             # beyond the model's context length
#             x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
#             # Get the logits from the model
#             logits = self.forward(x)
#             # Take the logits for the next token
#             next_token_logits = logits[:, -1]
#             # apply temperature scaling
#             temperature_scaled_next_token_logits = next_token_logits / temperature
#             # If top-k is provided, take the tokens with the highest score
#             if top_k:
#                 topk_values, _ = torch.topk(
#                     temperature_scaled_next_token_logits,
#                     min(top_k, temperature_scaled_next_token_logits.size(-1)),
#                 )
#                 # Get the score of the kth item that we kept---items with lower scores should be masked.
#                 threshold = topk_values[:, -1]
#                 topk_mask = temperature_scaled_next_token_logits < threshold
#                 temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))
#             next_token_probabilities = softmax(temperature_scaled_next_token_logits, dim=-1)
#             next_token_id = torch.multinomial(next_token_probabilities, 1)
#             # End generation if we see the EOS token ID
#             if eos_token_id is not None and next_token_id.item() == eos_token_id:
#                 break
#             x = torch.cat((x, next_token_id), dim=-1)
#         new_token_ids = x[:, original_sequence_length:]
#         return new_token_ids

#     @classmethod
#     def from_pretrained(cls, pretrained_model_path: str):
#         config_path = os.path.join(pretrained_model_path, "model_config.json")
#         with open(config_path) as f:
#             config = json.load(f)
#         model = cls(**config)
#         weights_path = os.path.join(pretrained_model_path, "model.pt")
#         state_dict = torch.load(weights_path)

#         # Remove _orig_mod. prefix that comes from serializing a compiled model
#         unwanted_prefix = "_orig_mod."
#         for k, _ in list(state_dict.items()):
#             if k.startswith(unwanted_prefix):
#                 state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
#         model.load_state_dict(state_dict)
#         return model


""" Only name changes from my A1 implemnentaion. Also, one change is removing the caching of causal mask.
May be assignment-2 does not use caching and expects mask matrix to be created inside CausalMultihead attention class"""

class BasicsTransformerLM(nn.Module):
    def __init__(self,vocab_size: int,context_length: int,d_model: int,num_layers: int,num_heads: int,d_ff: int,
        rope_theta: float | None = 10000.0,device=None,dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.context_length = context_length
        self.d_model = d_model
        self.device = device if device is not None else "cpu"
        self.dtype = dtype

        self.token_embeddings = Embedding(vocab_size, d_model, device=self.device, dtype=self.dtype)

        d_head = d_model // num_heads
        self.positional_encoder = (RotaryEmbedding(context_length, d_head, rope_theta, device=self.device)
            if rope_theta is not None else None)

        self.register_buffer('causal_mask',torch.tril(torch.ones(self.context_length,self.context_length,device = self.device,dtype=torch.bool)))
                

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model=d_model,num_heads=num_heads,d_ff=d_ff,positional_encoder=self.positional_encoder,
                    device=self.device, dtype=self.dtype) for _ in range(num_layers)])

       

        self.ln_final = RMSNorm(hidden_size=d_model, device=self.device, dtype=self.dtype)
        self.lm_head = Linear(d_model, vocab_size, device=self.device, dtype=self.dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(f"Sequence length exceeds context length {self.context_length}")

        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x,mask=self.causal_mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
 


# class TransformerBlock(nn.Module):
#     """A single Transformer layer.

#     This implements a single layer of the Transformer, as described in section 3.1
#     of the paper.

#     Args:
#         d_model: int
#             The dimensionality of the model embeddings and sublayer outputs.
#         num_heads: int
#             Number of heads to use in multi-headed attention. `d_model` must be
#             evenly divisible by `num_heads`.
#         d_ff: int
#             Dimensionality of the feed-forward inner layer (section 3.3).

#     Returns:
#         FloatTensor of shape `(batch_size, sequence_length, d_model)`.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         num_heads: int,
#         d_ff: int,
#         positional_encoder: RotaryEmbedding | None,
#     ):
#         super().__init__()
#         self.attn = CausalMultiHeadSelfAttention(
#             d_model=d_model,
#             num_heads=num_heads,
#             positional_encoder=positional_encoder,
#         )
#         self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
#         self.ln1 = RMSNorm(d_model)
#         self.ln2 = RMSNorm(d_model)

#     def forward(self, x: torch.Tensor):
#         """
#         Args:
#             x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
#                 The input to process with the Transformer block.

#         Returns:
#             FloatTensor of shape `(batch_size, sequence_length, d_model)`.
#         """
#         # NOTE: this is a pre-norm Transformer, and differs from the original
#         # description in the paper.
#         # Apply the multi-head self-attention sublayer
#         x_attn = self.attn(self.ln1(x))
#         attn_sublayer_output = x + x_attn

#         # Apply the feed-forward sublayer
#         x_ffn = self.ffn(self.ln2(attn_sublayer_output))
#         ffn_sublayer_output = attn_sublayer_output + x_ffn
#         return ffn_sublayer_output

""" Again lot of changes here due to changes in causal multihead attention. But logic is completely mine"""

class TransformerBlock(nn.Module):
    def __init__(self,d_model: int, num_heads: int, d_ff: int,positional_encoder: RotaryEmbedding | None = None,
                 device=None,dtype=None,):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.device = device if device is not None else "cpu"
        self.dtype = dtype

        self.ln1 = RMSNorm(hidden_size=d_model,device=self.device,dtype=self.dtype)

        self.attn = CausalMultiHeadSelfAttention(d_model=d_model,num_heads=num_heads,positional_encoder=positional_encoder,
            device=self.device,dtype=self.dtype)

        self.ln2 = RMSNorm(hidden_size=d_model,device=self.device,dtype=self.dtype)

        self.ffn = SwiGLU(d_model=d_model,d_ff=d_ff,device=self.device,dtype=self.dtype)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None,mask:torch.Tensor | None=None):
        y = self.ln1(x)
        x = x + self.attn(y,token_positions,mask)
        y = self.ln2(x)
        x = x + self.ffn(y)
        return x

       




# class SwiGLU(nn.Module):
#     def __init__(self, d_model: int, d_ff: int):
#         super().__init__()
#         self.w1 = Linear(d_model, d_ff)
#         self.w2 = Linear(d_ff, d_model)
#         self.w3 = Linear(d_model, d_ff)

#     def forward(self, x):
#         return self.w2(silu(self.w1(x)) * self.w3(x))

# def silu(x: torch.Tensor):
#     return x * torch.sigmoid(x)


"""SwiGLU changes: W1 -> w1, W2 -> w2 , W3 -> w3, Also their implementation had silu, so moved it outside from my implementation.""" 
"""Also their w2 is our W3 in original implementation and their w3 our W2. So defining as per their logic"""

def silu(x:torch.Tensor): ### x-> [B,T,C]
    return x*torch.sigmoid(x) #### [B,T,C]



class SwiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff:int,device=None,dtype=None):
        super().__init__()
        self.device = device if device is not None else "cpu"
        self.w1 = Linear(d_model,d_ff,device=self.device,dtype=dtype)
        self.w3 = Linear(d_model,d_ff,device=self.device,dtype=dtype)
        self.w2 = Linear(d_ff,d_model,device=self.device,dtype=dtype)
        

    def forward(self,x:torch.Tensor): ### x-> [B,T,C]
        return self.w2(silu(self.w1(x)) * self.w3(x))


# def scaled_dot_product_attention(
#     Q: Float[Tensor, " ... queries d_k"],
#     K: Float[Tensor, " ... keys    d_k"],
#     V: Float[Tensor, " ... keys    d_v"],
#     mask: Bool[Tensor, " ... queries keys"] | None = None,
# ) -> Float[Tensor, " ... queries d_v"]:
#     """Scaled dot-product attention.

#     This function implements Eq. 1 of the Transformer paper.

#     Args:
#         Q: Tensor of queries, may have any number of leading dimensions.
#         K: Tensor of keys, sharing leading dimensions with Q.
#         V: Tensor of values, sharding leading dimensions with Q and K.
#         mask: An (optional) mask of shape (..., seq_len, seq_len).
#             Attention scores for positions with a mask value of `False` should
#             be masked out, i.e., not affect the softmaxed attention probabilities.

#     Returns:
#         torch.FloatTensor of shape (..., seq_len, value_dimension)
#         with the output of running your scaled dot product attention
#         implementation with the provided key, query, and value tensors.
#     """

#     d_k = K.shape[-1]
#     attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

#     if mask is not None:
#         attention_scores = torch.where(mask, attention_scores, float("-inf"))

#     attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

#     return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

""" No change in scaled_dot_product_attention. Only Softmax -> softmax"""


def scaled_dot_product_attention(Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask=None):
    Tq,Cq = Q.shape[-2], Q.shape[-1]
    Tk,Ck = K.shape[-2], K.shape[-1] #### If not self-attention then Tq and Tk could be different, but Cq,Ck would be same

    Tv, Cv = V.shape[-2], V.shape[-1] ### Tv would be same as Tk

    ## Pytorch matmul
    attention_matrix = Q@K.transpose(-1,-2)/math.sqrt(Cq)

    if mask is not None:
        masked_attention = attention_matrix.masked_fill(~mask, float("-inf"))
    else:
        masked_attention = attention_matrix
    attention_scores = softmax(masked_attention,dim=-1)
    return attention_scores@V





# class CausalMultiHeadSelfAttention(nn.Module):
#     """Multi-Head Self-Attention

#     This function implements section 3.2.2 of the Transformer paper. In particular,
#     given an input tensor of shape `(batch_size, sequence_length, d_model)`, we project
#     it to create queries, keys, and values, and then perform causal multi-headed attention with
#     those queries, keys, and values.

#     Args:
#         d_model: int
#             The dimensionality of the model embeddings and sublayer outputs.
#         num_heads: int
#             Number of heads to use in multi-headed attention. `d_model` must be
#             evenly divisible by `num_heads`.

#     Returns:
#         Tensor of shape `(batch_size, sequence_length, d_model)`.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         num_heads: int,
#         positional_encoder: RotaryEmbedding | None = None,
#     ):
#         super().__init__()
#         if positional_encoder is None:
#             warnings.warn("No positional encoder provided", stacklevel=2)
#         assert d_model % num_heads == 0
#         self.d_model = d_model
#         self.num_heads = num_heads

#         self.d_k = d_model // num_heads
#         self.d_v = self.d_k

#         self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
#         self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
#         self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)

#         self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)

#         self.positional_encoder: RotaryEmbedding | None = positional_encoder  # RoPE

#     def forward(
#         self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None
#     ) -> Float[Tensor, " ... seq d_v"]:
#         """
#         Args:
#             x: The input to perform multi-headed self-attention on.
#             positional_ids: The positional indices along the sequence dimension of the input embeddings.

#         Returns:
#             Self-attention outputs.
#         """
#         *batch_dims, sequence_length, d_model = x.size()
#         assert d_model == self.d_model

#         Q = self.q_proj(x)
#         K = self.k_proj(x)
#         V = self.v_proj(x)

#         # Take apart each head from the embedding dimension of Q, K, V to shape (..., num_heads, seq_len, d_k).
#         Q, K, V = (
#             rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
#             for X in (Q, K, V)
#         )  # fmt: skip

#         if self.positional_encoder is not None:  # RoPE is enabled
#             if token_positions is not None:  # We got explicit position ids
#                 # Duplicate token positions for each head
#                 token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

#             Q = self.positional_encoder(Q, token_positions)
#             K = self.positional_encoder(K, token_positions)

#         # Construct causal mask
#         iota = torch.arange(sequence_length, device=x.device)
#         qi = rearrange(iota, "query -> query 1")
#         kj = rearrange(iota, "key   -> 1   key")
#         causal_mask = qi >= kj  # (query, key)
#         causal_mask = causal_mask.__getitem__((None,) * len(batch_dims) + (...,))  # Add appropriate leading dimensions

#         # Shape: (..., num_heads, sequence_length, d_k)
#         attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)

#         # Concatenate the attention output from all heads.
#         # (..., sequence_length, num_heads * d_v).
#         attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()

#         # Apply the output projection
#         output = self.output_proj(attn_output)
#         return output

""" Lot of changes in my implementation of class CausalMultiHeadSelfAttention. Firstly I had a function, which was called
transformer block, but now we have a class. This means the Transformer Block will change a lot.
But I have kept the implementation logic same as mine."""

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.positional_encoder = positional_encoder
        self.device = device if device is not None else "cpu"

        self.q_proj = Linear(d_model, d_model, device=self.device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=self.device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=self.device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=self.device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None,mask: None=None) -> torch.Tensor:
        C = x.shape[-1]
        T = x.shape[-2]

        Qproj = self.q_proj.weight
        Kproj = self.k_proj.weight
        Vproj = self.v_proj.weight
        Wproj = self.output_proj.weight

        Qproj = Qproj.view(self.num_heads, self.d_model // self.num_heads, C)
        Kproj = Kproj.view(self.num_heads, self.d_model // self.num_heads, C)
        Vproj = Vproj.view(self.num_heads, self.d_model // self.num_heads, C)

        x_view = x.unsqueeze(-3)

        Q = x_view @ Qproj.transpose(-1, -2)
        K = x_view @ Kproj.transpose(-1, -2)
        V = x_view @ Vproj.transpose(-1, -2)

        if self.positional_encoder is not None:
            if token_positions is not None:
                token_positions = token_positions.unsqueeze(-2)  # [..., 1, seq]
            Q = self.positional_encoder(Q, token_positions)
            K = self.positional_encoder(K, token_positions)

        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        else:
            mask = mask[:T,:T]
        attention = scaled_dot_product_attention(Q, K, V, mask)

        *batch_dims, h, T, d_head = attention.shape
        attn_out = attention.transpose(-3, -2).reshape(*batch_dims, T, h * d_head)

        return attn_out @ Wproj.transpose(-2, -1)


