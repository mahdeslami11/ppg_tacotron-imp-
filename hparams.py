# TIMIT
timit_sr = 16000
timit_n_mfcc = 40
timit_n_fft = 1024
timit_hop_length = 256
timit_wim_length = 1024
timit_n_mels = 80

timit_default_duration = 2

# Net1
net1_train_dataset = "./data/dataset/TIMIT/TRAIN/*/*/*.wav"
net1_train_batch_size = 32
net1_train_num_workers = 4
