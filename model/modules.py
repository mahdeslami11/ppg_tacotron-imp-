import torch
import torch.nn.functional as Func
from torch.nn import Module
from torch.nn import Linear, Conv1d, MaxPool1d, Dropout, BatchNorm1d, ReLU, Sigmoid


class PreNet(Module):
    def __init__(self, in_dims, out_dims_1, out_dims_2, dropout_rate):
        super(PreNet, self).__init__()

        self.relu = ReLU()
        self.drop = Dropout(dropout_rate)

        self.fc1 = Linear(in_dims, out_dims_1)
        self.fc2 = Linear(out_dims_1, out_dims_2)

    def forward(self, inputs):
        # inputs : (N, L_in, in_dims)
        # fc1_outputs : (N, L_in, out_dims_1)
        fc1_outputs = self.fc1(inputs)
        relu1_outputs = self.relu(fc1_outputs)
        layer_1_outputs = self.drop(relu1_outputs)

        # fc2_outputs : (N, L_in, out_dims_2)
        fc2_outputs = self.fc2(layer_1_outputs)
        relu2_outputs = self.relu(fc2_outputs)
        layer_2_outputs = self.drop(relu2_outputs)

        # layer_2_outputs : (N, L_in, out_dims_2)
        return layer_2_outputs


class HighwayNet(Module):
    def __init__(self, in_dims, out_dims=None):
        # Attention !  in_dims == out_dims
        super(HighwayNet, self).__init__()

        if out_dims is None:
            out_dims = in_dims

        assert in_dims == out_dims

        self.fc1 = Linear(in_dims, out_dims)
        self.fc2 = Linear(in_dims, out_dims)
        torch.nn.init.constant_(self.fc2.bias, -1.0)

        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, inputs):
        # All Tensor in same shape
        # inputs : (N, L_in, in_dimes)
        # H : (N, L_in, in_dimes)
        h = self.fc1(inputs)
        H = self.relu(h)

        # T : (N, L_in, in_dimes)
        t = self.fc2(inputs)
        T = self.sigmoid(t)

        C = 1.0 - T

        outputs = H * T + inputs * C

        debug = False
        if debug:
            print("input.shape : " + str(inputs.shape))
            print("H.shape : " + str(H.shape))
            print("input.shape : " + str(inputs.shape))
            print("T.shape : " + str(T.shape))
            print("C.shape : " + str(C.shape))

        return outputs


class Conv1dNorm(Module):
    def __init__(self, in_dims, out_dims, kernel_size, activation_fn):
        super(Conv1dNorm, self).__init__()

        # Padding with kernel_size ensures that 'padding = "same"'
        if kernel_size % 2 is not 0:
            self.conv1d = Conv1d(in_dims, out_dims, kernel_size, padding=(kernel_size - 1) // 2)
        else:
            # padding + 1
            self.conv1d = Conv1d(in_dims, out_dims, kernel_size, padding=(kernel_size + 1) // 2)

        # self.conv1d = Conv1d(in_dims, out_dims, kernel_size, padding=(kernel_size - 1) // 2)

        self.batch_norm = BatchNorm1d(out_dims)
        self.activation_fn = activation_fn

        self.k_size = kernel_size

    def forward(self, inputs):

        # inputs (N, in_dims, L_in)
        # conv1d_outputs (N, out_dims, L_out)
        # # L_out = (L_in + 2*padding - dilation*(kernel_size-1)-1)//stride + 1
        # L_out = L_in
        conv1d_outputs = self.conv1d(inputs)

        if self.k_size % 2 is 0:
            conv1d_outputs = conv1d_outputs[:, :, :-1]

        # L_out = (inputs.shape[-1] + 2*sum(self.conv1d.padding) -
        #          sum(self.conv1d.dilation) * (sum(self.conv1d.kernel_size) - 1) - 1) \
        #         // sum(self.conv1d.stride) + 1

        L_out = inputs.shape[-1]
        assert L_out == conv1d_outputs.shape[-1]

        conv1d_norm_outputs = self.batch_norm(conv1d_outputs)
        assert L_out == conv1d_norm_outputs.shape[-1]

        if self.activation_fn is not None:
            conv1d_norm_outputs = self.activation_fn(conv1d_norm_outputs)

        debug = False
        if debug:
            print("inputs.shape : " + str(inputs.shape))
            print("conv1d_outputs.shape : " + str(conv1d_outputs.shape))
            print("conv1d_norm_outputs.shape : " + str(conv1d_norm_outputs.shape))

        # conv1d_norm_outputs (N, out_dims, L_out)
        return conv1d_norm_outputs


class Conv1dBanks(Module):
    def __init__(self, k, in_dims, out_dims, activation):
        super(Conv1dBanks, self).__init__()

        self.conv1_norm_outputs = []

        self.k = k
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.activation = activation

    def forward(self, inputs):
        # inputs : (N, in_dims , L_in)
        for k_size in range(1, 1 + self.k):
            conv1d_norm = Conv1dNorm(self.in_dims, self.out_dims, k_size, self.activation)
            # conv1d_norm_outputs : (N, out_dims, L_out)
            # L_in == L_out
            conv1d_norm_outputs = conv1d_norm(inputs)
            self.conv1_norm_outputs.append(conv1d_norm_outputs)

        # conv1d_banks : (N, k*out_dims, L_out)
        conv1d_banks = torch.cat(self.conv1_norm_outputs, 1)

        return conv1d_banks


# TODO : Add GRU Model


class CBHG(Module):
    def __init__(self, k, num_highway_blocks, in_dims, out_dims, activation):
        super(CBHG, self).__init__()

        self.num_highway_blocks = num_highway_blocks

        self.conv1d_banks = Conv1dBanks(k, in_dims, out_dims, activation)

        # Since kernel_size = 2, padding + 1
        self.max_pool1d = MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.projection1 = Conv1dNorm(in_dims=k * out_dims, out_dims=out_dims,
                                      kernel_size=3, activation_fn=activation)
        self.projection2 = Conv1dNorm(in_dims=out_dims, out_dims=out_dims,
                                      kernel_size=3, activation_fn=None)

        self.highway = HighwayNet(in_dims=out_dims)

    def forward(self, inputs):
        # inputs : (N, in_dims, L_in)
        # conv1d_banks_outputs : (N, k*out_dims, L_in)
        conv1d_banks_outputs = self.conv1d_banks(inputs)
        print("conv1d_banks_outputs : " + str(conv1d_banks_outputs.shape))

        # Cut out the rest
        max_pool1d_outputs = self.max_pool1d(conv1d_banks_outputs)
        max_pool1d_outputs = max_pool1d_outputs[:, :, :-1]
        # max_pool1d_outputs : (N, k*out_dims, L_in)
        print("max_pool1d_outputs : " + str(max_pool1d_outputs.shape))

        assert conv1d_banks_outputs.shape == max_pool1d_outputs.shape

        # projection1_outputs : (N, out_dims, L_in)
        projection1_outputs = self.projection1(max_pool1d_outputs)
        print("projection1_outputs : " + str(projection1_outputs.shape))
        # projection2_outputs : (N, out_dims, L_in)
        projection2_outputs = self.projection2(projection1_outputs)
        print("projection2_outputs : " + str(projection2_outputs.shape))

        # residual_connections : (N, out_dims, L_in)
        residual_connections = projection2_outputs + inputs
        highway_data = residual_connections.transpose(2, 1)

        for i in range(self.num_highway_blocks):
            highway_data = self.highway(highway_data)

        # TODO : Add GRU Model

        return highway_data
