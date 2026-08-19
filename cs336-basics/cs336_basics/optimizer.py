from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch


# def get_cosine_lr(
#     it: int,
#     max_learning_rate: float,
#     min_learning_rate: float,
#     warmup_iters: int,
#     cosine_cycle_iters: int,
# ):
#     """Cosine with warmup learning rate scheduler."""
#     # First, we linearly warmup for warmup_iters steps.
#     if it < warmup_iters:
#         return max_learning_rate * it / warmup_iters
#     # Then, if it > cosine_cycle_iters, we return min learning rate.
#     if it > cosine_cycle_iters:
#         return min_learning_rate
#     # Else, we use cosine decay down to min learning rate.
#     decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
#     return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


# class AdamW(torch.optim.Optimizer):
#     def __init__(
#         self,
#         params: Iterable[torch.nn.parameter.Parameter],
#         lr: float = 1e-3,
#         betas: tuple[float, float] = (0.9, 0.999),
#         eps: float = 1e-8,
#         weight_decay: float = 0.01,
#     ):
#         if not 0.0 <= lr:
#             raise ValueError(f"Invalid learning rate: {lr}")
#         if not 0.0 <= eps:
#             raise ValueError(f"Invalid epsilon value: {eps}")
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
#         super().__init__(params, defaults)

#     def step(self, closure: Callable | None = None):
#         loss = None
#         if closure is not None:
#             loss = closure()
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue

#                 grad = p.grad.data
#                 if grad.is_sparse:
#                     raise RuntimeError("Adam does not support sparse gradients")

#                 state = self.state[p]
#                 alpha = group["lr"]
#                 beta_1, beta_2 = group["betas"]
#                 eps = group["eps"]
#                 t = state.get("t", 1)

#                 # Apply weight decay
#                 alpha_t = alpha * (math.sqrt(1 - (beta_2**t)) / (1 - (beta_1**t)))
#                 p.data -= alpha * group["weight_decay"] * p.data

#                 prev_m_t = state.get("m", torch.zeros_like(grad))
#                 prev_v_t = state.get("v", torch.zeros_like(grad))

#                 m_t = beta_1 * prev_m_t + ((1 - beta_1) * grad)
#                 v_t = beta_2 * prev_v_t + ((1 - beta_2) * torch.square(grad))

#                 # Apply adjusted gradient step
#                 p.data -= alpha_t * m_t / (torch.sqrt(v_t) + eps)

#                 state["m"] = m_t
#                 state["v"] = v_t
#                 state["t"] = t + 1
#         return loss

#### Renamed my cosine_annealing from assignment1 to get_cosine_lr and changed order of paramaters

def get_cosine_lr(it:int, 
                     max_learning_rate: float,
                     min_learning_rate:float,
                     warmup_iters,
                     cosine_cycle_iters) -> float:
    if it < warmup_iters:
        return it*max_learning_rate/warmup_iters
    elif it > cosine_cycle_iters:
        return min_learning_rate

    else:
        temp = (it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi
        return min_learning_rate + 0.5 *(1 + math.cos(temp) ) *(max_learning_rate - min_learning_rate)

    
class AdamW(torch.optim.Optimizer):
    def __init__(self,params:Iterable[torch.nn.parameter.Parameter],lr:float=1e-3,weight_decay:float=0.0,betas: tuple[float, float] = (0.9, 0.999),eps:float=1e-8):
        defaults = {"lr":lr,"betas":betas,"eps":eps,"weight_decay":weight_decay}
        super().__init__(params,defaults)
    
    @torch.no_grad()
    def step(self,closure: Callable | None=None):
        loss = None
        if closure is not None:  ### This if statement gets used only for LBFGS type method, but due to syntax and design requirement we need it
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            lr = group["lr"]
            eps = group["eps"]
            lamda = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad
                t = state.get("t",1)
                m = beta1*state.get("momentum",torch.zeros_like(grad)) + (1-beta1)*grad
                v = beta2*state.get("v",torch.zeros_like(grad)) + (1-beta2)*(grad**2)
                update = m/(torch.sqrt(v)+eps)
                p-=(math.sqrt(1-beta2**t))/(1-beta1**t) *lr*update 
                p-=lr*lamda*p   
                state["momentum"] = m
                state["v"] = v
                state["t"] = t+1
        return loss
    

