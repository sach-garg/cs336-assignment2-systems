from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch
import torch.cuda.nvtx as nvtx


from cs336_systems.configurations import BenchmarkConfig

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--measure_iters", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--mode", type=str, choices=["F", "FB", "FBO"])
    parser.add_argument("--device", type=str)
    parser.add_argument("--out_dir", type=str)
    return parser.parse_args()


def BuildConfig():
    config = BenchmarkConfig()
    args = parse_args()
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    return config


def main():
    
    config = BuildConfig()


    if config.mode not in ["F", "FB", "FBO"]:
        raise ValueError("mode must be one of 'F', 'FB', or 'FBO'")

    device = config.device if config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.randint(high=config.vocab_size, size=(config.batch_size, config.context_length+1), 
                           dtype=torch.long, device =device)

    x = data[:, :-1]
    y = data[:,1:]


    model = BasicsTransformerLM(config.vocab_size, config.context_length,config.d_model,
                              config.num_layers,config.num_heads,config.d_ff,
                              config.rope_theta,device=device,dtype=torch.float32).to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.lr_max, weight_decay=config.weight_decay,
                       betas=(config.beta1, config.beta2),eps=config.eps)

    for _ in range(config.warmup):
        logits = model(x)
        if config.mode == "F":
            continue
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)),y.reshape(-1)) ### -> [B*T,V], [B*T]
        optimizer.zero_grad()
        loss.backward()
        if config.mode == "FB":
            continue
        optimizer.step()


    for _ in range(1): ## Profiling only 1 iteration
        with nvtx.range("Forward"):
            logits = model(x)
        if config.mode=="F":
            continue
        with nvtx.range("Loss"):
            loss = cross_entropy(logits.reshape(-1, logits.size(-1)),y.reshape(-1)) ### -> [B*T,V], [B*T]
        optimizer.zero_grad()
        with nvtx.range("Backward"):
            loss.backward()
        if config.mode == "FB":
            continue
        with nvtx.range("Optimizer"):
            optimizer.step()
       

if __name__ == "__main__":
    main()



