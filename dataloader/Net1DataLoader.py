import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from audio_operation import get_mfccs_and_phones


class Net1Dataset(Dataset):

    def __init__(self, data_path):
        self.wav_files = glob.glob(data_path)

    def __getitem__(self, item):
        wav_file = self.wav_files[item]
        return get_mfccs_and_phones(wav_file=wav_file)

    def __len__(self):
        return len(self.wav_files)


def get_net1_data_loader(data_path, batch_size, num_workers):
    dataset = Net1Dataset(data_path)

    # worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             drop_last=True,
                             worker_init_fn=np.random.seed((torch.initial_seed()) % (2 ** 32)))

    return data_loader
