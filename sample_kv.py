import torch

# Define small tensors for illustration
batch_size = 2
seq_length = 2  # Fixed sequence length for key

# Create sample tensors
cache_index = torch.tensor([[2], [3]])  # Example cache indices
batch_index = torch.tensor([[0], [1]])  # Example batch indices
key = torch.tensor(
    [[[1, 2], [3, 4]], [[7, 8], [9, 10]]], dtype=torch.float32
)  # Example key tensor with fixed seq_length
past = torch.tensor(
    [
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.10], [0.11, 0.12]],
        [
            [0.13, 0.14],
            [0.15, 0.16],
            [0.17, 0.18],
            [0.19, 0.20],
            [0.21, 0.22],
            [0.23, 0.24],
        ],
    ],
    dtype=torch.float32,
)  # Example key tensor with fixed seq_length

key = key[:, None, :, :]
past = past[:, None, :, :]

# Calculate kv_indices
kv_indices = torch.arange(seq_length).unsqueeze(0) + cache_index

# Transpose past tensor to align dimensions
past = past.transpose(1, 2)

# Update past tensor with key values at appropriate indices
past[batch_index, kv_indices] = key.transpose(1, 2)

# Transpose past tensor back to its original shape
past = past.transpose(1, 2)

# Print tensors for visualization
print("cache_id:\n", cache_index)
print("batch_id:\n", batch_index)
print("key:\n", key)
print("past:\n", past)
