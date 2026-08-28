import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from timeit import default_timer
import pandas as pd
import pickle

### For CPU
def setup_CPU(rank,world_size):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "29501"

  dist.init_process_group("gloo",rank=rank,world_size=world_size) ## "gloo" for CPU


### For single node, multi GPU training
def setup(rank,world_size):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "29500"
  torch.cuda.set_device(rank)
  dist.init_process_group("nccl",rank=rank,world_size=world_size) ## "nccl" for GPUs

def sum_across_devices(rank,world_size,m,result_queue):
  setup(rank,world_size)
  data = torch.zeros(m,dtype=torch.float32).to("cuda") 
  #data = torch.randn(m,dtype=torch.float32) ## CPU

  ### warm_up
  for _ in range(5):
    dist.all_reduce(data,async_op=False)


  ###
  torch.cuda.synchronize() ## finish warmup before procedding ahead
  dist.barrier() ## wait for all other GPUs to arrive here
  torch.cuda.synchronize() ## finish GPU work used by the barrier
  start = default_timer()
  dist.all_reduce(data,async_op=False)
  torch.cuda.synchronize() ### finish all reduce work
  end=default_timer()
  ## Define mytime tensor, as all_gather only communicates pytorch tensors,
  ## That means, a python float like end-start can not be communicated directly via all_gather
 
  mytime = torch.tensor([end-start],dtype=torch.float32,device=data.device) 

  ## Also all_gather requires preallocated output space (list) where received data will be written
  all_times = [torch.empty_like(mytime) for _ in range(world_size)]
  dist.all_gather(all_times,mytime)
  torch.cuda.synchronize() ### finish all gather
  if rank==0:
    times =[t.item() for t in all_times]
    result_queue.put(times)
  
  dist.destroy_process_group()

def main():
  tensor_sizes = [1,10,100,1000] ## tensor sizes in MB 
  stats=[]
  spawn_context = mp.get_context("spawn") ## jargon, needed because processes can't return values to their parent directly
  for world_size in [2,4,6]:
  
    for i,s in enumerate(tensor_sizes):
      print(world_size,s)
   
      m = s*(1024**2)//4 ## we will create fp32 tensors
      result_queue = spawn_context.SimpleQueue() ## jargon
      mp.spawn(fn=sum_across_devices,args=(world_size,m,result_queue),nprocs=world_size,join=True)
      times = result_queue.get()
      stats.append({"world_size":world_size, "tensor_size": tensor_sizes[i], "times":times,"avg_time": pd.Series(times).mean(), "std_time": pd.Series(times).std()})
      print(f"world_size: {world_size}, Tensor_size:{tensor_sizes[i]}, Avgtime:{pd.Series(times).mean()}")
  stats_df = pd.DataFrame(stats)
  stats_df.to_csv("allreduce_benchmarking.csv",index=False)
  return

if __name__ == "__main__":
  main()
  
      


  


    
