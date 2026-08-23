from cs336_basics.model import scaled_dot_product_attention
import torch
from timeit import default_timer
import pandas as pd


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 8
    warmup=2
    profile_iterations=100
    
    stats={}
    for d_model in [16,32,64,128]:
        for T in [256, 1024, 4096, 8192, 16384]:
            forward_times =[]
            backward_times =[]
            memory_before_backward=[]
            attn=None
            try:
                Q = torch.randn(B,T,d_model,device=device, dtype=torch.float32,requires_grad=True)
                K = torch.randn(B,T,d_model,device=device, dtype=torch.float32,requires_grad=True)
                V = torch.randn(B,T,d_model,device=device, dtype=torch.float32,requires_grad=True)
                mask = torch.tril(torch.ones(T,T,device=device,dtype=torch.bool))
                for _ in range(warmup):
                    attn = scaled_dot_product_attention(Q,K,V,mask,use_nvtx=False)
                    if Q.grad is not None: Q.grad = None
                    if K.grad is not None: K.grad = None
                    if V.grad is not None: V.grad = None
                    attn.sum().backward()

                torch.cuda.synchronize()

                for _ in range(profile_iterations):
                    start = default_timer()
                    attn = scaled_dot_product_attention(Q,K,V,mask,use_nvtx=False)
                    torch.cuda.synchronize()
                    end = default_timer()
                    forward_times.append(end-start)

                    if Q.grad is not None: Q.grad = None
                    if K.grad is not None: K.grad = None
                    if V.grad is not None: V.grad = None
                    m = torch.cuda.memory_allocated()
                    memory_before_backward.append(m)
   
                    start = default_timer()
                    attn.sum().backward()
                    torch.cuda.synchronize()
                    end=default_timer()
                    backward_times.append(end-start)
                stats[(d_model, T)] = {
                                        "forward_mean": sum(forward_times) / len(forward_times),
                                        "forward_std": pd.Series(forward_times).std(),
                                        "forward_times":forward_times,
                                        "backward_mean": sum(backward_times) / len(backward_times),
                                        "backward_std": pd.Series(backward_times).std(),
                                        "backward_times":backward_times,
                                        "memory_before_backward":memory_before_backward,
                                        "memory_before_backward_mean": sum(memory_before_backward) / len(memory_before_backward),
                                        "memory_before_backward_std": pd.Series(memory_before_backward).std()}
            
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                stats[(d_model, T)] = {"status":"OOM"}

            del Q, K, V,attn,mask
            torch.cuda.empty_cache()
        stats_df = pd.DataFrame.from_dict(stats, orient="index")

    stats_df.index = pd.MultiIndex.from_tuples(stats_df.index, names=["d_model", "T"])
    stats_df = stats_df.reset_index()
    stats_df.to_pickle("pytorch_attention_benchmark.pkl")
    stats_df.to_csv("pytorch_attention_benchmark.csv", index=False)

    print(stats_df)
    return

if __name__ =="main":
    main()





