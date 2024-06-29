import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
from transformers import AutoTokenizer


class KVCacheModule(nn.Module):
    def __init__(self):
        super(KVCacheModule, self).__init__()

    def forward(self, cache_id, batch_id, key, past):
        seq_length = key.shape[-2]
        kv_indices = torch.arange(seq_length).unsqueeze(0) + cache_id
        past = past.transpose(1, 2)
        past[batch_id, kv_indices] = key.transpose(1, 2)
        return past

    def without_transpose(self, cache_id, batch_id, key, past):
        # Calculate the sequence length
        seq_length = key.shape[-2]

        # Calculate the indices for KV (Key-Value) cache
        kv_indices = torch.arange(seq_length).unsqueeze(0) + cache_id

        # Reshape past tensor to [batch_size, seq_length, -1, 4] and key tensor to [batch_size, seq_length, 4, -1]
        past_reshaped = past.view(past.size(0), past.size(2), past.size(1), -1)
        key_reshaped = key.view(key.size(0), seq_length, -1, key.size(3))

        # Update past tensor with key values at appropriate indices
        # past_reshaped.scatter_(dim=-1, index=kv_indices.unsqueeze(1).expand(-1, seq_length, -1, -1), src=key_reshaped)
        past_reshaped[batch_id, kv_indices] = key_reshaped

        # Reshape past tensor back to its original shape
        past = past_reshaped.view_as(past)

        return past


# Create an instance of the model
model = KVCacheModule()

# Sample inputs
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer(
    [
        "Write a sample code for preprocessing strings in python",
        "Write a sample code for preprocessing strings in c++ and explain how it works in detail",
        "Write a sample code for preprocessing strings in rust and csharp",
    ]
)

batch_size = len(inputs.input_ids)

prompt_len_1 = len(inputs.input_ids[0])
prompt_len_2 = len(inputs.input_ids[1])
# Initialize cache_id tensor
cache_id = torch.zeros(len(inputs.input_ids), 1, dtype=torch.long)

for i, input_id in enumerate(inputs.input_ids):
    cache_id[i] = len(input_id)

batch_id = torch.arange(batch_size).view(-1, 1)
key = torch.randn((batch_size, 1, 2, 4), dtype=torch.float32)
past = torch.randn((batch_size, 1, 128, 4), dtype=torch.float32)

# Run forward pass
output_pytorch = model(cache_id, batch_id, key, past)
output_without_transpose = model.without_transpose(cache_id, batch_id, key, past)

# Export the model to ONNX
input_names = ["cache_id", "batch_id", "key", "past"]
output_names = ["output"]
dummy_input = (cache_id, batch_id, key, past)
onnx_path = "kv_cache_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        "cache_id": {0: "batch_size"},
        "batch_id": {0: "batch_size"},
        "key": {0: "batch_size"},
        "past": {0: "batch_size"},
    },
)

# Load the exported ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# Run inference with ONNX Runtime
ort_session = onnxruntime.InferenceSession(onnx_path)
output_onnxrt = ort_session.run(
    None,
    {
        "cache_id": cache_id.numpy(),
        "batch_id": batch_id.numpy(),
        "key": key.numpy(),
        "past": past.numpy(),
    },
)

# Compare outputs using numpy.allclose
if np.allclose(output_pytorch.numpy(), output_onnxrt[0]):
    print("Outputs are similar between PyTorch and ONNX Runtime.")
    # print("Pytorch output {} and Onnxrt output {}".format(output_pytorch, output_onnxrt))
else:
    print("Outputs differ between PyTorch and ONNX Runtime.")
