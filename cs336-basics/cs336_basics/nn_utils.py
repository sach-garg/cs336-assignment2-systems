import torch
from typing import List


# def softmax(x, dim=-1):
#     rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
#     exponentiated_rescaled_input = torch.exp(rescaled_input)
#     return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)


""" Only change from A1 is the dim=-1 in the function call """


def softmax(logits:torch.Tensor,dim: int =-1):

    max_logits = torch.max(logits,dim=dim,keepdim=True).values ### max returns tuple (values,indices)
    stable_logits = torch.exp(logits - max_logits)
    normalizer = torch.sum(stable_logits,dim=dim,keepdim=True)
    return stable_logits/normalizer



# def log_softmax(x, dim=-1):
#     x_max = torch.max(x, dim=dim, keepdim=True)[0]
#     x = x - x_max
#     return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))



# def cross_entropy(inputs, targets):
#     negative_log_softmax_logits = -log_softmax(inputs)
#     return torch.mean(torch.gather(negative_log_softmax_logits, -1, targets.unsqueeze(-1)))

""" Change in cross_entropy: CrossEntropy -> cross_entropy"""


def cross_entropy(logits:torch.Tensor,y:torch.Tensor): ### logits <-- [B,vocab_size] ### y <- [B]
    B = logits.shape[0]
    max_logits = torch.max(logits,dim=1,keepdim=True).values #### <--[B,1]
    shifted_logits = logits-max_logits #### ---> [B,V]
    normalizer = torch.sum(torch.exp(shifted_logits),dim=1) #### --> [B]
    return -torch.mean(shifted_logits[torch.arange(B,device=logits.device), y]- torch.log(normalizer))


# def clip_gradient(parameters, max_norm):
#     grads = [p.grad for p in parameters if p.grad is not None]
#     norm = torch.tensor(0.0, device=grads[0].device)

#     for g in grads:
#         norm += (g**2).sum()

#     norm = torch.sqrt(norm)
#     clip_coef = min(1, max_norm / (norm + 1e-6))
#     for g in grads:
#         g *= clip_coef

""" Changes in clip_gradient: params -> parameters, M->max_norm"""

def clip_gradient(parameters: List[torch.Tensor],max_norm:float) -> None:
    device = parameters[0].device
    dtype = parameters[0].dtype
    temp = torch.sqrt(torch.sum(torch.tensor([torch.linalg.norm(p.grad)**2 for p in parameters if p.grad is not None],device = device, dtype = dtype)))
    if temp <=   max_norm:
        return
    scale = max_norm/(temp + 1e-6)
    for p in parameters:
        if p.grad is not None:
            p.grad*=scale
    return
    
