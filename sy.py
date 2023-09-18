import torch

test_conv = torch.nn.Conv2d(16, 33, 3, stride=2)
print(test_conv.weight.shape)

in_dim =320
lora_dim = 64
kernel_size = 3
out_dim = 320
lora_down = torch.nn.Conv2d(in_dim, lora_dim, kernel_size, bias=False)
lora_up = torch.nn.Conv2d(lora_dim, out_dim, (1, 1), (1, 1), bias=False)
in_rank, in_size, kernel_size, k_ = lora_down.shape
"""
def merge_conv(lora_down, lora_up):
    in_rank, in_size, kernel_size, k_ = lora_down.weight.shape
    out_size, out_rank, _, _ = lora_up.weight.shape
    assert in_rank == out_rank and kernel_size == k_, f"rank {in_rank} {out_rank} or kernel {kernel_size} {k_} mismatch"


    merged = lora_up.reshape(out_size, -1) @ lora_down.reshape(in_rank, -1)
    weight = merged.reshape(out_size, in_size, kernel_size, kernel_size)
    del lora_up, lora_down
    return weight
merged_weight = merge_conv(lora_down, lora_up)
print(merged_weight)
"""
up_weight = lora_up.weight
#down_weight = lora.lora_down.weight