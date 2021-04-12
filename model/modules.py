import torch
from torch.nn import Module, init
from torch.nn import Linear, Dropout, ReLU, Sigmoid


class PreNet(Module):
    def __init__(self, in_dims, out_dims_1, out_dims_2, dropout_rate):
        super(PreNet, self).__init__()

        self.relu = ReLU()
        self.drop = Dropout(dropout_rate)

        self.fc1 = Linear(in_dims, out_dims_1)
        self.fc2 = Linear(out_dims_1, out_dims_2)

    def froward(self, inputs):
        fc1_outputs = self.fc1(inputs)
        relu1_outputs = self.relu(fc1_outputs)
        layer_1_outputs = self.drop(relu1_outputs)

        fc2_outputs = self.fc2(layer_1_outputs)
        relu2_outputs = self.relu(fc2_outputs)
        layer_2_outputs = self.drop(relu2_outputs)

        return layer_2_outputs


class HighwayNet(Module):
    def __init__(self, in_dims, out_dims):
        super(HighwayNet, self).__init__()

        self.fc1 = Linear(in_dims, out_dims)
        self.fc2 = Linear(in_dims, out_dims)
        init.constant_(self.fc2.bias, -1.0)

        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def froward(self, inputs):
        h = self.fc1(inputs)
        H = self.relu(h)

        t = self.fc2(inputs)
        T = self.sigmoid(t)

        C = 1.0 - T

        outputs = H * T + inputs * C

        return outputs
