import torch
from audio_operation import phns

phns_len = len(phns)

# TIMIT
timit_sr = 16000
timit_n_mfcc = 40
timit_n_fft = 1024
timit_hop_length = 256
timit_wim_length = 1024
timit_n_mels = 80

timit_default_duration = 2

# Net1
# net1 dataset
net1_dataset = "../data/dataset/TIMIT/TRAIN/*/*/*.wav"
net1_batch_size = 64
net1_num_workers = 8

# net1 model
net1_in_dims = timit_n_mfcc
net1_hidden_units = 128
net1_dropout_rate = 0
net1_num_conv1d_banks = 8
net1_num_highway_blocks = 4
# temperature
net1_logits_t = 1.0

# net1 train
net1_train_device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
net1_train_steps = 10000
net1_train_lr = 0.0003
net1_train_log_step = 1
net1_train_multiple_flag = True
