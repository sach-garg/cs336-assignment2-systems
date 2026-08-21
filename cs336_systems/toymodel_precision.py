import torch
import torch.nn as nn

# Case 1: accumulate in fp32 using fp32 increments
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print("fp32 accumulator, fp32 addends:", s)

# Case 2: accumulate in fp16 using fp16 increments
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print("fp16 accumulator, fp16 addends:", s)

# Case 3: fp32 accumulator, but adding fp16 values directly
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print("fp32 accumulator, fp16 addends:", s)

# Case 4: fp32 accumulator, fp16 value explicitly cast to fp32 before adding
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print("fp32 accumulator, fp16 addends cast to fp32:", s)



class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print("fc1 output dtype:", x.dtype)
        x = self.relu(x)
        print("relu output dtype:", x.dtype)
        x = self.ln(x)
        print("ln output dtype:", x.dtype)
        x = self.fc2(x)
        print("logits dtype:", x.dtype)
        return x

model = ToyModel(in_features =400, out_features = 400)
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(2,400,device=device)
y = torch.randn(2, 400, device=device)
model = model.to(device)
#dtype = torch.float16
dytpe = torch.bfloat16

print("parameter dtype outside autocontext:", model.fc1.weight.dtype)

with torch.autocast(device_type="cuda",dtype=dtype):
    logits = model(x)
    print("parameter dtype inside autocontext:", model.fc1.weight.dtype)
    loss = ((logits - y) ** 2).mean()
    print("loss dtype:", loss.dtype)
    loss.backward()
    print("gradient dtype inside autocontext",model.fc1.weight.grad.dtype)

