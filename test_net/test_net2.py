import argparse
import torch
import os
import time
import datetime

import hparams

from model.Net1 import Net1
from model.Net2 import Net2
from dataloader.Net2DataLoader import get_net2_data_loader


def test(arg):
    device = torch.device(arg.device)

    # Build model
    # Build net1 model
    net1 = Net1(in_dims=hparams.net1_in_dims,
                hidden_units=hparams.net1_hidden_units,
                dropout_rate=hparams.net1_dropout_rate,
                num_conv1d_banks=hparams.net1_num_conv1d_banks,
                num_highway_blocks=hparams.net1_num_highway_blocks)

    net2 = Net2(in_dims=hparams.net2_in_dims,
                hidden_units=hparams.net2_hidden_units,
                dropout_rate=hparams.net2_dropout_rate,
                num_conv1d_banks=hparams.net2_num_conv1d_banks,
                num_highway_blocks=hparams.net2_num_highway_blocks)

    # Move model into the computing device
    net1.to(device)
    net2.to(device)

    # Set data loader
    # # Set data loader
    data_loader = get_net2_data_loader(data_path=hparams.net2_test_dataset,
                                       batch_size=hparams.net2_batch_size,
                                       num_workers=0)

    # Resume net1 model
    if arg.resume_net1_model is None:
        raise Exception(print("Need net1 pre-trained model!"))

    resume_net1_model_path = os.path.join(hparams.net1_train_checkpoint_path, arg.resume_net1_model)
    resume_log = "Resume net1 model from : " + resume_net1_model_path
    print(resume_log)

    checkpoint_net1 = torch.load(resume_net1_model_path)
    print("Load net1 model successfully!")

    net1.load_state_dict(checkpoint_net1["net"])

    # Resume net2 model
    if arg.resume_net2_model is None:
        raise Exception(print("Need net2 pre-trained model!"))

    resume_net2_model_path = os.path.join(hparams.net2_train_checkpoint_path, arg.resume_net2_model)
    resume_log = "Resume net2 model from : " + resume_net2_model_path
    print(resume_log)

    checkpoint_net2 = torch.load(resume_net2_model_path)
    print("Load net1 model successfully!")

    net2.load_state_dict(checkpoint_net2["net"])

    # Start testing
    print("Start testing ... ")
    start_time = time.time()

    data_iter = iter(data_loader)

    # Get input data
    mfccs, spec, mel = next(data_iter)

    # Moving input data into the computing device
    mfccs = mfccs.to(device)
    spec = spec.to(device)
    mel = mel.to(device)

    # Set net1 and net2 model
    net1 = net1.eval()
    net2 = net2.eval()

    # Compute net1
    net1_outputs_ppgs, _, _ = net1(mfccs)

    net2_inputs_ppgs = net1_outputs_ppgs.detach()

    # Compute net2
    pred_spec, pred_mel = net2(net2_inputs_ppgs)
    # pred_spec = net2(net2_inputs_ppgs)

    # Compute the loss
    criterion = torch.nn.MSELoss(reduction='mean')
    loss_spec = criterion(pred_spec, spec)
    loss_mel = criterion(pred_mel, mel)
    loss = loss_spec + loss_mel
    # loss = loss_spec

    # Print loss
    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    log = "Elapsed [{}], Loss : [{:.6f}], Loss_spec : [{:.6f}], Loss_mel : [{:.6f}]" \
        .format(et, loss, loss_spec, loss_mel)
    # log = "Elapsed [{}], Loss : [{:.6f}], Loss_spec : [{:.6f}]" \
    #     .format(et, loss, loss_spec)
    print(log)


def get_arguments():
    parser = argparse.ArgumentParser()

    # Set test net1 config
    parser.add_argument('-device', default=hparams.net2_train_device, type=str,
                        help='Net2 training device.')
    parser.add_argument('resume_net1_model', type=str,
                        help='Net1 resume model checkpoint.')
    parser.add_argument('resume_net2_model', type=str,
                        help='Net2 resume model checkpoint.')

    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()

    print("Test Net2 parameters : \n " + str(args))

    test(args)
