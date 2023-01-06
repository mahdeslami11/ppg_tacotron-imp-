import argparse
import os
import torch
import time
import datetime
import numpy
import hparams

from model.Net1 import Net1, get_net1_loss, get_net1_acc
from dataloader.Net1DataLoader import get_net1_data_loader


def test(arg):
    device = torch.device(arg.device)

    # Build model
    net1 = Net1(in_dims=hparams.net1_in_dims,
                hidden_units=hparams.net1_hidden_units,
                dropout_rate=hparams.net1_dropout_rate,
                num_conv1d_banks=hparams.net1_num_conv1d_banks,
                num_highway_blocks=hparams.net1_num_highway_blocks)

    # Move net1 model into the computing device
    net1.to(device)

    # Set data loader
    data_loader = get_net1_data_loader(data_path=hparams.net1_test_dataset,
                                       batch_size=hparams.net1_batch_size,
                                       num_workers=hparams.net1_num_workers)

    # Resume net1 model
    resume_model_path = os.path.join(hparams.net1_train_checkpoint_path, arg.resume_model)
    resume_log = "Resume net1 model from : " + resume_model_path
    print(resume_log)

    checkpoint = torch.load(resume_model_path)
    print("Load model successfully!")

    net1.load_state_dict(checkpoint["net"])

    # Start testing
    print("Start testing ... ")
    start_time = time.time()

    data_iter = iter(data_loader)

    # Get input data
    mfccs, phones = next(data_iter)

    # Moving input data into the computing device
    mfccs = mfccs.to(device)
    phones = phones.long().to(device)

    # Test net1 model
    net1 = net1.eval()
    ppgs, preds, logits = net1(mfccs)

    # Compute the loss
    loss = get_net1_loss(logits, phones, mfccs)

    # Compute the accuracy
    acc = get_net1_acc(preds, phones, mfccs)

    # Print loss and acc
    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    log = "Elapsed [{}], Loss : [{:.6f}], Accuracy : [{:.6f}]".format(et, loss, acc)
    print(log)


def get_arguments():
    parser = argparse.ArgumentParser()

    # Set test net1 config
    parser.add_argument('-device', default=hparams.net1_train_device, type=str,
                        help='Net1 training device.')
    parser.add_argument('resume_model', type=str,
                        help='Net1 resume model checkpoint.')

    arguments = parser.parse_args()
    return arguments


if __name__ == '__main-net__':
    args = get_arguments()

    print("Test Net1 parameters : \n " + str(args))

    test(args)
