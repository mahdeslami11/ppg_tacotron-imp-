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
net1_train_steps = 100000
net1_train_checkpoint_path = "../checkpoint/net1"
net1_train_lr = 0.0003
net1_train_log_step = 10
net1_train_save_step = 1000
net1_train_multiple_flag = False

# Net2
# net2 dataset
net2_dataset = "../data/dataset/arctic/slt/*.wav"
net2_batch_size = 16
net2_num_workers = 5

# net2 model
net2_in_dims = phns_len
net2_hidden_units = 256
net2_dropout_rate = 0
net2_num_conv1d_banks = 8
net2_num_highway_blocks = 8

# net2 train
net2_train_device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
net2_train_steps = 100000
net2_train_checkpoint_path = "../checkpoint/net2"
net2_train_lr = 0.0003
net2_train_log_step = 10
net2_train_save_step = 10000
