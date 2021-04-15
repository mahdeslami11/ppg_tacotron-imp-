import torch
from torch.nn import Module, Linear, Softmax, CrossEntropyLoss
from .modules import PreNet, CBHG

import hparams


class Net1(Module):
    def __init__(self, in_dims, hidden_units, dropout_rate, num_conv1d_banks, num_highway_blocks):
        super().__init__()

        # in_dims = n_mfcc, out_dims_1 = 2*out_dims_2 = net1_hidden_units
        self.pre_net = PreNet(in_dims=in_dims,
                              out_dims_1=hidden_units,
                              dropout_rate=dropout_rate)

        # num_conv1d_banks = net1_num_conv1d_banks, num_highway_blocks = net1_num_highway_blocks
        # in_dims = net1_hidden_units // 2, out_dims = net1_hidden_units // 2
        # activation=torch.nn.ReLU()
        self.cbhg = CBHG(num_conv1d_banks=num_conv1d_banks,
                         num_highway_blocks=num_highway_blocks,
                         in_dims=hidden_units // 2,
                         out_dims=hidden_units // 2,
                         activation=torch.nn.ReLU())

        # in_features = net1_hidden_units, out_features = phns_len
        self.logits = Linear(in_features=hidden_units, out_features=hparams.phns_len)
        self.softmax = Softmax(dim=-1)

    def forward(self, inputs):
        # inputs : (N, L_in, in_dims)
        # in_dims = n_mfcc

        # PreNet
        pre_net_outputs = self.pre_net(inputs)
        # pre_net_outputs : (N, L_in, net1_hidden_units // 2)

        # Change data format
        cbhg_inputs = pre_net_outputs.transpose(2, 1)
        # cbhg_inputs : (N, net1_hidden_units // 2, L_in)

        # CBHG
        cbhg_outputs = self.cbhg(cbhg_inputs)
        # cbhg_outputs : (N, L_in, net1_hidden_units)

        # Final linear projection
        logits_outputs = self.logits(cbhg_outputs)
        # logits_outputs : (N, L_in, phns_len)

        ppgs = self.softmax(logits_outputs / hparams.net1_logits_t)
        # ppgs : (N, L_in, phns_len)

        preds = torch.argmax(logits_outputs, dim=-1).int()
        # preds = (N, L_in)

        debug = False
        if debug:
            print("pre_net_outputs : " + str(pre_net_outputs.shape))
            print("cbhg_inputs : " + str(cbhg_inputs.shape))
            print("cbhg_outputs : " + str(cbhg_outputs.shape))
            print("logits_outputs : " + str(logits_outputs.shape))
            print("ppgs : " + str(ppgs.shape))
            print("preds : " + str(preds.shape) + " , preds.type : " + str(preds.dtype))

        # ppgs : (N, L_in, phns_len)
        # preds : (N, L_in)
        # logits_outputs : (N, L_in, phns_len)
        return ppgs, preds, logits_outputs


def get_net1_loss(logits, phones, mfccs):
    is_target = torch.sign(torch.abs(torch.sum(mfccs, -1)))

    compute_loss = CrossEntropyLoss()
    loss = compute_loss(logits.transpose(1, 2) / hparams.net1_logits_t, phones)

    loss = loss * is_target
    loss = torch.mean(loss)

    return loss


def get_net1_acc(preds, phones, mfccs):
    is_target = torch.sign(torch.abs(torch.sum(mfccs, -1)))

    hits = torch.eq(preds, phones.int()).float()

    num_hits = torch.sum(hits * is_target)
    num_targets = torch.sum(is_target)

    acc = num_hits / num_targets

    return acc
