import argparse
import torch
import os
import numpy as np
import calc
import hparams
import decorate 

from model.Net1 import Net1
from model.Net2 import Net2
from dataloader.Net2DataLoader import get_net2_data_loader
from audio_operation import db2amp, spec2wav, save_wav, denormalize_0_1, inv_preemphasis


def convert(x_spec, y_spec, x_mel, y_mel):
    x_spec = x_spec.cpu().detach().numpy()
    y_spec = y_spec.cpu().detach().numpy()

    # Denormalizatoin
    x_spec = denormalize_0_1(x_spec, hparams.timit_max_db, hparams.timit_min_db)
    y_spec = denormalize_0_1(y_spec, hparams.timit_max_db, hparams.timit_min_db)

    # Db to amp
    x_spec = db2amp(x_spec)
    y_spec = db2amp(y_spec)

    # Emphasize the magnitude
    x_spec = np.power(x_spec, hparams.convert_emphasis_magnitude)
    x_spec = np.power(x_spec, hparams.convert_emphasis_magnitude)

    print(x_spec.shape)
    print(y_spec.shape)

    x_spec = np.squeeze(x_spec)
    y_spec = np.squeeze(y_spec)

    # Spectrogram to waveform
    spec_x_audio = np.array(spec2wav(mag=x_spec.T,
                                     n_fft=hparams.timit_n_fft,
                                     win_length=hparams.timit_win_length,
                                     hop_length=hparams.timit_hop_length,
                                     num_iters=hparams.convert_num_iters))

    spec_y_audio = np.array(spec2wav(mag=y_spec.T,
                                     n_fft=hparams.timit_n_fft,
                                     win_length=hparams.timit_win_length,
                                     hop_length=hparams.timit_hop_length,
                                     num_iters=hparams.convert_num_iters))

    print(spec_x_audio.shape)
    print(spec_y_audio.shape)

    spec_x_audio = inv_preemphasis(spec_x_audio, hparams.timit_preemphasis).astype(np.float32)
    spec_y_audio = inv_preemphasis(spec_y_audio, hparams.timit_preemphasis).astype(np.float32)

    return spec_x_audio, spec_y_audio


def do_convert(arg):
    device = torch.device(arg.device)

    # Build model
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
    data_loader = get_net2_data_loader(data_path=arg.data_path,
                                       batch_size=arg.batch_size,
                                       num_workers=arg.num_workers)

    # Resume net1 model
    if arg.resume_net1_model is null:
        raise Exception(print("Need net1 pre-trained model!"))

    resume_net1_model_path = os.path.join(hparams.net1_convert_checkpoint_path, arg.resume_net1_model)
    resume_log = "Resume net1 model from : " + resume_net1_model_path
    print(resume_log)

    checkpoint_net1 = torch.load(resume_net1_model_path)
    print("Load net1 model successfully!")

    net1.load_state_dict(checkpoint_net1["net"])

    # Resume net2 model
    if arg.resume_net2_model is None:
        raise Exception(print("Need net2 pre-trained model!"))

    resume_net2_model_path = os.path.join(hparams.net2_convert_checkpoint_path, arg.resume_net2_model)
    resume_log = "Resume net2 model from : " + resume_net2_model_path
    print(resume_log)

    checkpoint_net2 = torch.load(resume_net2_model_path)
    print("Load net1 model successfully!")

    net2.load_state_dict(checkpoint_net2["net"])

    # Start Converting
    print("Start Converting ... ")

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

    pred_spec_audio, real_spec_audio = convert(pred_spec, spec, pred_mel, mel)
    # pred_spec_audio, real_spec_audio = convert(pred_spec, spec, None, mel)

    # Write the result
    save_wav(hparams.convert_save_path + "-" + arg.resume_net2_model + "-spec_converted.wav",
             pred_spec_audio, sr=hparams.timit_sr)
    save_wav(hparams.convert_save_path + "-" + arg.resume_net2_model + "-spec_real.wav",
             real_spec_audio, sr=hparams.timit_sr)


def get_arguments():
    parser = argparse.ArgumentParser()

    # Set DataLoader
    parser.add_argument('-data_path', default=hparams.convert_dataset, type=str,
                        help='Path of Net2 dataset.')
    parser.add_argument('-batch_size', default=hparams.convert_batch_size, type=int,
                        help='Batch size.')
    parser.add_argument('-num_workers', default=hparams.convert_num_workers, type=int,
                        help='Number of workers.')

    # Set convert config
    parser.add_argument('-device', default=hparams.convert_device, type=str,
                        help='Convert device.')
    parser.add_argument('-resume_net1_model', default=None, type=str,
                        help='Net1 resume model checkpoint.')
    parser.add_argument('-resume_net2_model', default=None, type=str,
                        help='Net2 resume model checkpoint.')

    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()

    print("Convert parameter : \n " + str(args))

    do_convert(args)
