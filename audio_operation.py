import librosa
import scipy
import scipy.fftpack
import numpy as np

import hparams


def read_wav(path, sr, duration=None, mono=True):
    wav, sr = librosa.load(path=path, sr=sr, mono=mono, duration=duration)
    return wav


def amp2db(amp):
    return librosa.amplitude_to_db(amp)


def db2amp(db):
    return librosa.db_to_amplitude(db)


def _get_mfcc_and_spec(wav, sr, n_fft, hop_length, win_length, n_mels, n_mfcc):

    print(wav.shape)

    # Get spectrogram
    # (1 + n_fft/2, t)
    spec = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(spec)
    print(mag.shape)

    # Get mel-spectrogram
    # (n_mels, 1+n_fft//2)
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    # mel spectrogram (n_mels, t)
    mel = np.dot(mel_basis, mag)
    print(mel.shape)

    # amp to db
    mag_db = amp2db(mag)
    mel_db = amp2db(mel)
    print(mag_db.shape)
    print(mel_db.shape)

    # Get mfccs
    mfccs = scipy.fftpack.dct(mel_db, axis=0, type=2, norm='ortho')[:n_mfcc]
    print(mfccs.shape)

    # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)
    return mfccs.T, mag_db.T, mel_db.T


def get_mfccs_and_phones(wav_file, trim=False, random_crop=True):
    sr = hparams.timit_sr
    n_fft = hparams.timit_n_fft
    hop_length = hparams.timit_hop_length
    win_length = hparams.timit_wim_length
    n_mels = hparams.timit_n_mels
    n_mfcc = hparams.timit_n_mfcc

    # Load wav
    wav = read_wav(wav_file, sr)

    mfcc, _, _ = _get_mfcc_and_spec(wav, sr, n_fft, hop_length, win_length, n_mels, n_mfcc)

    # TODO : get phones and return

    return mfcc
