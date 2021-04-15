import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from audio_operation import get_mfccs_and_spectrogram


class Net2Dataset(Dataset):

    def __init__(self, data_path):
        self.wav_files = glob.glob(data_path)

    def __getitem__(self, item):
        wav = self.wav_files[item]
        return get_mfccs_and_spectrogram(wav)

    def __len__(self):
        return len(self.wav_files)


def get_net2_data_loader(data_path, batch_size, num_workers):
    dataset = Net2Dataset(data_path)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             drop_last=True,
                             worker_init_fn=np.random.seed((torch.initial_seed()) % (2 ** 32)))

    return data_loader
