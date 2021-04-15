import torch
from torch.nn import Module, Linear
from .modules import PreNet, CBHG

import hparams


class Net2(Module):
    def __init__(self, in_dims, hidden_units, dropout_rate, num_conv1d_banks, num_highway_blocks):
        super(Net2, self).__init__()

        # in_dims = phones_len, out_dims_1 = 2*out_dims_2 = net2_hidden_units
        self.pre_net = PreNet(in_dims=in_dims,
                              out_dims_1=hidden_units,
                              dropout_rate=dropout_rate)

        # num_conv1d_banks = net2_num_conv1d_banks, num_highway_blocks = net2_num_highway_blocks
        # in_dims = net2_hidden_units // 2, out_dims = net2_hidden_units // 2
        # activation=torch.nn.ReLU()
        self.cbhg_mel = CBHG(num_conv1d_banks=num_conv1d_banks,
                             num_highway_blocks=num_highway_blocks,
                             in_dims=hidden_units // 2,
                             out_dims=hidden_units // 2,
                             activation=torch.nn.ReLU())

        # num_conv1d_banks = net2_num_conv1d_banks, num_highway_blocks = net2_num_highway_blocks
        # in_dims = net2_hidden_units // 2, out_dims = net2_hidden_units // 2
        # activation=torch.nn.ReLU()
        self.cbhg_spec = CBHG(num_conv1d_banks=num_conv1d_banks,
                              num_highway_blocks=num_highway_blocks,
                              in_dims=hidden_units // 2,
                              out_dims=hidden_units // 2,
                              activation=torch.nn.ReLU())

        self.pred_mel = Linear(in_features=hidden_units, out_features=hparams.timit_n_mels)
        self.prepare_spec = Linear(in_features=hparams.timit_n_mels, out_features=hidden_units // 2)
        self.pred_spec = Linear(in_features=hidden_units, out_features=hparams.timit_n_fft // 2 + 1)

    def forward(self, inputs):
        # inputs : (N, L_in, in_dims)
        # in_dims = phns_len

        # PreNet
        pre_net_outputs = self.pre_net(inputs)
        # pre_net_outputs : (N, L_in, net2_hidden_units // 2)

        # Change data format
        cbhg_mel_inputs = pre_net_outputs.transpose(1, 2)
        # cbhg_mel_inputs : (N, net2_hidden_units // 2, L_in)

        # CBHG : mel-scale
        cbhg_mel_outputs = self.cbhg_mel(cbhg_mel_inputs)
        # cbhg_mel_outputs : (N, L_in, net2_hidden_units)

        # Pred mel
        pred_mel = self.pred_mel(cbhg_mel_outputs)
        # pred_mel : (N, L_in, n_mels)

        # Change data format
        cbhg_spec_inputs = self.prepare_spec(pred_mel)
        # cbhg_spec_inputs : (N, L_in, net2_hidden_units // 2)
        cbhg_spec_inputs = cbhg_spec_inputs.transpose(1, 2)
        # cbhg_spec_inputs : (N, net2_hidden_units // 2, L_in)

        # Pred spec
        cbhg_spec_outputs = self.cbhg_spec(cbhg_spec_inputs)
        # cbhg_spec_outputs : (N, L_in, net2_hidden_units)

        # Pred spec
        pred_spec = self.pred_spec(cbhg_spec_outputs)
        # pred_spec : (N, L_in, n_fft//2 + 1)

        debug = False
        if debug:
            print("inputs.shape : " + str(inputs.shape))
            print("pre_net_outputs.shape : " + str(pre_net_outputs.shape))
            print("cbhg_mel_inputs.shape : " + str(cbhg_mel_inputs.shape))
            print("cbhg_mel_outputs.shape : " + str(cbhg_mel_outputs.shape))
            print("pred_mel.shape : " + str(pred_mel.shape))
            print("cbhg_spec_inputs.shape : " + str(cbhg_spec_inputs.shape))
            print("cbhg_spec_outputs.shape : " + str(cbhg_spec_outputs.shape))
            print("pred_spec.shape : " + str(pred_spec.shape))

        return pred_spec, pred_mel
