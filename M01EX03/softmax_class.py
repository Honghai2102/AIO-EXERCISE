import torch

# Data
data = torch.tensor([1, 2, 3])

# Custom Softmax class


class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        c = torch.max(x)
        x = x - c
        x = torch.exp(x)
        sum_exp = torch.sum(x, dim=0, keepdim=True)
        return x / sum_exp

# Softmax_stable


class SoftmaxStable(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        c = torch.max(x)
        x = x - c
        x = torch.exp(x)
        sum_exp = torch.sum(x, dim=0, keepdim=True)
        return x / sum_exp


if __name__ == "__main__":
    # Softmax instance
    softmax = Softmax()
    # Calculate softmax
    output = softmax(data)
    print(output)

    # Softmax_stable instance
    softmax_stable = SoftmaxStable()
    # Calculate softmax_stable
    output_stable = softmax_stable(data)
    print(output_stable)
