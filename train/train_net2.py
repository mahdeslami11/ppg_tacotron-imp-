import os
import argparse
import torch
import time
import datetime

import hparams

from model.Net1 import Net1
from model.Net2 import Net2
from dataloader.Net2DataLoader import get_net2_data_loader

from tensorboardX import SummaryWriter


def train(arg):
    device = torch.device(arg.device)

    # Build Net1 model
    net1 = Net1(in_dims=hparams.net1_in_dims,
                hidden_units=hparams.net1_hidden_units,
                dropout_rate=hparams.net1_dropout_rate,
                num_conv1d_banks=hparams.net1_num_conv1d_banks,
                num_highway_blocks=hparams.net1_num_highway_blocks)

    # Move net1 model into the computing device
    net1.to(device)

    # Build Net2 model
    net2 = Net2(in_dims=arg.in_dims,
                hidden_units=arg.hidden_units,
                dropout_rate=arg.dropout_rate,
                num_conv1d_banks=arg.num_conv1d_banks,
                num_highway_blocks=arg.num_highway_blocks)

    # Create optimizer
    net2_optimizer = torch.optim.Adam(net2.parameters(), lr=arg.learning_rate)

    # Move net2 model into the computing device
    net2.to(device)

    # Set data loader
    data_loader = get_net2_data_loader(data_path=arg.data_path,
                                       batch_size=arg.batch_size,
                                       num_workers=arg.num_workers)

    start_step = 1

    # Resume net1 model
    if arg.resume_net1_model is None:
        raise Exception(print("Need net1 pre-trained model!"))

    resume_net1_model_path = os.path.join(hparams.net1_train_checkpoint_path, arg.resume_net1_model)
    resume_log = "Resume net1 model from : " + resume_net1_model_path
    print(resume_log)

    checkpoint_net1 = torch.load(resume_net1_model_path)
    print("Load net1 model successfully!")

    net1.load_state_dict(checkpoint_net1["net"])

    # Fixed parameters of the net1 model
    for p in net1.parameters():
        p.requires_grad = False

    # Resume net2 model
    if arg.resume_net2_model is not None:
        resume_net2_model_path = os.path.join(arg.checkpoint_path, arg.resume_net2_model)
        resume_log = "Resume net2 model from : " + resume_net2_model_path
        print(resume_log)

        checkpoint_net2 = torch.load(resume_net2_model_path)
        print("Load net2 model successfully!")

        net2.load_state_dict(checkpoint_net2["net"])
        net2_optimizer.load_state_dict(checkpoint_net2["optimizer"])
        start_step = checkpoint_net2["step"]

        if start_step >= arg.train_steps:
            raise Exception(print(" Training completed !"))

    # Set TensorBoard writer
    writer = SummaryWriter('loss_log/train2', flush_secs=1)

    # Start training
    print("Start training ... ")
    start_time = time.time()

    data_iter = iter(data_loader)

    for step in range(start_step, arg.train_steps + 1):

        # Get input data
        try:
            mfccs, spec, mel = next(data_iter)
        except:
            data_iter = iter(data_loader)
            mfccs, spec, mel = next(data_iter)

        # Moving input data into the computing device
        mfccs = mfccs.to(device)
        spec = spec.to(device)
        mel = mel.to(device)

        # Set net1 and net2 model
        net1 = net1.eval()
        net2 = net2.train()

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

        # Backward and optimize
        net2_optimizer.zero_grad()
        loss.backward()
        net2_optimizer.step()

        # Print out training info
        if step % arg.log_step == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            writer.add_scalar("net2_loss", loss, global_step=step)
            writer.add_scalar("net2_spec_loss", loss_spec, global_step=step)
            # writer.add_scalar("net2_mel_loss", loss_mel, global_step=step)
            log = "Elapsed [{}], Iteration [{}/{}], Loss : [{:.6f}], Loss_spec : [{:.6f}], Loss_mel : [{:.6f}]" \
                .format(et, step, arg.train_steps, loss, loss_spec, loss_mel)
            # log = "Elapsed [{}], Iteration [{}/{}], Loss : [{:.6f}], Loss_spec : [{:.6f}]" \
            #     .format(et, step, arg.train_steps, loss, loss_spec)
            print(log)

        # Save model
        if step % arg.save_step == 0:
            checkpoint = {
                "net": net2.state_dict(),
                "optimizer": net2_optimizer.state_dict(),
                "step": step
            }

            if not os.path.isdir(arg.checkpoint_path):
                os.mkdir(arg.checkpoint_path)

            torch.save(checkpoint, os.path.join(arg.checkpoint_path, 'ckpt_%s.pth' % str(step)))

            log = "Net2 training result has been saved to pth : ckpt_%s.pth ." % str(step)
            print(log)


def get_arguments():
    parser = argparse.ArgumentParser()

    # Set Net1
    parser.add_argument('-in_dims', default=hparams.net2_in_dims, type=int,
                        help='Number of Net2 input dimensions.')
    parser.add_argument('-hidden_units', default=hparams.net2_hidden_units, type=int,
                        help='Number of Net2 hidden units.')
    parser.add_argument('-dropout_rate', default=hparams.net2_dropout_rate, type=float,
                        help='Rate of net2 Dropout layers.')
    parser.add_argument('-num_conv1d_banks', default=hparams.net2_num_conv1d_banks, type=int,
                        help='Number of Net2 conv1d banks.')
    parser.add_argument('-num_highway_blocks', default=hparams.net2_num_highway_blocks, type=int,
                        help='Number of Net2 Highway blocks.')

    # Set DataLoader
    parser.add_argument('-data_path', default=hparams.net2_dataset, type=str,
                        help='Path of Net2 dataset.')
    parser.add_argument('-batch_size', default=hparams.net2_batch_size, type=int,
                        help='Batch size.')
    parser.add_argument('-num_workers', default=hparams.net2_num_workers, type=int,
                        help='Number of workers.')

    # Set Train config
    parser.add_argument('-device', default=hparams.net2_train_device, type=str,
                        help='Net2 training device.')
    parser.add_argument('-checkpoint_path', default=hparams.net2_train_checkpoint_path, type=str,
                        help='Net2 model checkpoint path.')
    parser.add_argument('-resume_net1_model', default=None, type=str,
                        help='Net1 resume model checkpoint.')
    parser.add_argument('-resume_net2_model', default=None, type=str,
                        help='Net2 resume model checkpoint.')
    parser.add_argument('-train_steps', default=hparams.net2_train_steps, type=int,
                        help='Net2 training steps.')
    parser.add_argument('-learning_rate', default=hparams.net2_train_lr, type=float,
                        help='Net2 learning rate.')
    parser.add_argument('-log_step', default=hparams.net2_train_log_step, type=int,
                        help='Net2 training log steps.')
    parser.add_argument('-save_step', default=hparams.net2_train_save_step, type=int,
                        help='Net2 training save steps.')

    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()

    print("Train Net2 parameters : \n " + str(args))

    train(args)
