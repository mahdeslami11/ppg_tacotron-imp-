import argparse
import hparams
import torch
import time
import datetime
import logging
import os

from model.Net1 import Net1, get_net1_loss, get_net1_acc
from dataloader.Net1DataLoader import get_net1_data_loader


def train(arg):
    device = torch.device(arg.device)

    # Build net1 model
    net1 = Net1(in_dims=arg.in_dims,
                hidden_units=arg.hidden_units,
                dropout_rate=arg.dropout_rate,
                num_conv1d_banks=arg.num_conv1d_banks,
                num_highway_blocks=arg.num_highway_blocks)

    # Create optimizer
    net1_optimizer = torch.optim.Adam(net1.parameters(), lr=arg.learning_rate)

    # Set Multi-GPU training mode
    if arg.multiple_train:
        device = torch.device('cuda')
        net1 = torch.nn.DataParallel(net1)

    # Move net1 model into the computing device
    net1.to(device)

    # Set data loader
    data_loader = get_net1_data_loader(data_path=arg.data_path,
                                       batch_size=arg.batch_size,
                                       num_workers=arg.num_workers)

    start_step = 1

    # Resume checkpoint
    if arg.resume_model is not None:
        resume_model_path = os.path.join(arg.checkpoint_path, arg.resume_model)
        resume_log = "Resume model from : " + resume_model_path
        print(resume_log)
        logger.info(resume_log)

        checkpoint = torch.load(resume_model_path)
        print("Load model successfully!")
        logger.info("Load model successfully!")

        net1.load_state_dict(checkpoint["net"])
        net1_optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]

        if start_step >= arg.train_steps:
            logger.error(" Training completed !")
            raise Exception(print(" Training completed !"))

    # Start training
    print("Start training ... ")
    start_time = time.time()

    data_iter = iter(data_loader)

    for step in range(start_step, arg.train_steps + 1):

        # Get input data
        try:
            mfccs, phones = next(data_iter)
        except:
            data_iter = iter(data_loader)
            mfccs, phones = next(data_iter)

        # Moving input data into the computing device
        mfccs = mfccs.to(device)
        phones = phones.long().to(device)

        # Train net1 model
        net1 = net1.train()
        ppgs, preds, logits = net1(mfccs)

        # Compute the loss
        loss = get_net1_loss(logits, phones, mfccs)

        # Compute the accuracy
        acc = get_net1_acc(preds, phones, mfccs)

        # Backward and optimize
        net1_optimizer.zero_grad()
        loss.backward()
        net1_optimizer.step()

        debug = False
        if debug:
            print("net1 info : " + str(net1))
            print("net1_optimizer info : " + str(net1_optimizer))

            print("mfccs shape : " + str(mfccs.shape) + " , mfccs type : " + str(mfccs.dtype))
            print("phones shape : " + str(phones.shape) + " , phones type : " + str(phones.dtype))

            print("ppgs shape : " + str(ppgs.shape) + " , ppgs type : " + str(ppgs.dtype))
            print("preds shape : " + str(preds.shape) + " , preds type : " + str(preds.dtype))
            print("logits shape : " + str(logits.shape) + " , logits type : " + str(logits.dtype))

        # Print out training info
        if step % arg.log_step == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}], Loss : [{:.6f}], Accuracy : [{:.6f}]".format(et, step,
                                                                                                 arg.train_steps, loss,
                                                                                                 acc)
            print(log)
            logger.info(log)

        # Save model
        if step % arg.save_step == 0:
            checkpoint = {
                "net": net1.state_dict(),
                "optimizer": net1_optimizer.state_dict(),
                "step": step
            }

            if not os.path.isdir(arg.checkpoint_path):
                os.mkdir(arg.checkpoint_path)

            torch.save(checkpoint, os.path.join(arg.checkpoint_path, 'ckpt_%s.pth' % str(step)))

            log = "Net1 training result has been saved to pth : ckpt_%s.pth ." % str(step)
            print(log)
            logger.info(log)


def get_arguments():
    parser = argparse.ArgumentParser()

    # Set Net1
    parser.add_argument('-in_dims', default=hparams.net1_in_dims, type=int,
                        help='Number of Net1 input dimensions.')
    parser.add_argument('-hidden_units', default=hparams.net1_hidden_units, type=int,
                        help='Number of Net1 hidden units.')
    parser.add_argument('-dropout_rate', default=hparams.net1_dropout_rate, type=float,
                        help='Rate of net1 Dropout layers.')
    parser.add_argument('-num_conv1d_banks', default=hparams.net1_num_conv1d_banks, type=int,
                        help='Number of Net1 conv1d banks.')
    parser.add_argument('-num_highway_blocks', default=hparams.net1_num_highway_blocks, type=int,
                        help='Number of Net1 Highway blocks.')

    # Set DataLoader
    parser.add_argument('-data_path', default=hparams.net1_dataset, type=str,
                        help='Path of Net1 dataset.')
    parser.add_argument('-batch_size', default=hparams.net1_batch_size, type=int,
                        help='Batch size.')
    parser.add_argument('-num_workers', default=hparams.net1_num_workers, type=int,
                        help='Number of workers.')

    # Set Train config
    parser.add_argument('-device', default=hparams.net1_train_device, type=str,
                        help='Net1 training device.')
    parser.add_argument('-checkpoint_path', default=hparams.net1_train_checkpoint_path, type=str,
                        help='Net1 model checkpoint path.')
    parser.add_argument('-resume_model', default=None, type=str,
                        help='Net1 resume model checkpoint.')
    parser.add_argument('-train_steps', default=hparams.net1_train_steps, type=int,
                        help='Net1 training steps.')
    parser.add_argument('-learning_rate', default=hparams.net1_train_lr, type=float,
                        help='Net1 learning rate.')
    parser.add_argument('-log_step', default=hparams.net1_train_log_step, type=int,
                        help='Net1 training log steps.')
    parser.add_argument('-save_step', default=hparams.net1_train_save_step, type=int,
                        help='Net1 training save steps.')
    parser.add_argument('-multiple_train', default=hparams.net1_train_multiple_flag, type=bool,
                        help='Net1 training log steps.')

    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':

    # Set log
    logging.basicConfig(level=logging.INFO, filename="../log.txt", filemode="w")
    logger = logging.getLogger("log_test")

    args = get_arguments()
    print("Train Net1 parameters : \n " + str(args))
    logger.info(args)

    if args.multiple_train and args.device is not 'cuda':
        raise Exception("Multi-GPU training mode enabled, but the default computing device does not support it. "
                        "Calculate device conflicts, please check the Settings.")

    train(args)
