from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from pathlib import Path
import torch
from timeit import default_timer
import pandas as pd


from cs336_systems.configurations import BenchmarkConfig


def append_benchmark_result(config, device, times):
    out_path = Path(config.out_dir) / "benchmark_history.pkl"
    new_row = pd.DataFrame(
        [{"d_model": config.d_model, "d_ff": config.d_ff,
            "num_layers": config.num_layers,"num_heads": config.num_heads,
            "mode": config.mode, "warm_up": config.warmup, "device": device,
              "Avg_time": pd.Series(times).mean(), "Std_time": pd.Series(times).std(), "times": times}])

    if out_path.exists():
        old_df = pd.read_pickle(out_path)
        df = pd.concat([old_df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_pickle(out_path)
    return df

def main():
    
    config = BenchmarkConfig()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    for it in range(config.warmup):
        logits = model(x)
        if config.mode == "F":
            continue
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)),y.reshape(-1)) ### -> [B*T,V], [B*T]
        optimizer.zero_grad()
        loss.backward()
        if config.mode == "FB":
            continue
        optimizer.step()

    times =[]
    if "cuda" in device:
        torch.cuda.synchronize()

    

    for it in range(config.measure_iters):
        if "cuda" in device:
            torch.cuda.synchronize()
        start = default_timer()

        logits = model(x)

        if config.mode=="F":
            if "cuda" in device:
                torch.cuda.synchronize()
            end = default_timer()
            times.append(end-start)
            continue
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)),y.reshape(-1)) ### -> [B*T,V], [B*T]
        optimizer.zero_grad()
        loss.backward()
        if config.mode == "FB":
            if "cuda" in device:
                torch.cuda.synchronize()
            end = default_timer()
            times.append(end-start)
            continue
        optimizer.step()
       
        if config.mode == "FBO":
            if "cuda" in device:
                torch.cuda.synchronize()
            end = default_timer()
            times.append(end-start)

    df = append_benchmark_result(config, device, times)
    print(df.tail())
    df.to_csv(Path(config.out_dir) / "benchmark_history.csv", index=False)
    print(f"mean time = {pd.Series(times).mean():.6f} s")
    print(f"std time = {pd.Series(times).std():.6f} s")


if __name__ == "__main__":
    main()



