import torch
from torch import nn
class GatedMultimodalLayer(nn.Module):

    def __init__(self, size_in1, size_in2, size_out):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden_sigmoid = nn.Linear(size_in1 + size_in2, size_out, bias=False)
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden2(x2))
        x = torch.cat((x1, x2), dim=1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))

        return z * h1 + (1 - z) * h2
