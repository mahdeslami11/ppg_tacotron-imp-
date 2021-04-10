import glob
from torch.utils.data import Dataset

from audio_operation import get_mfccs_and_phones


class Net1Dataset(Dataset):

    def __init__(self, data_path):
        self.wav_files = glob.glob(data_path)

    def __getitem__(self, item):
        wav_file = self.wav_files[item]
        return get_mfccs_and_phones(wav_file=wav_file)

    def __len__(self):
        return len(self.wav_files)
