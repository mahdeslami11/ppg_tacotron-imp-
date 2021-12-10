import librosa
import scipy
import scipy.fftpack
import numpy as np
from scipy import signal

import hparams

phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']


def load_vocab():
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn


def read_wav(path, sr, duration=None, mono=True):
    wav, sr = librosa.load(path=path, sr=sr, mono=mono, duration=duration)
    return wav


def save_wav(path, wav, sr):
    import soundfile as sf    
    sf.write(path, wav, sr)
    pass


def amp2db(amp):
    return librosa.amplitude_to_db(amp)


def db2amp(db):
    return librosa.db_to_amplitude(db)


def normalize_0_1(values, max_db, min_db):
    normalized = np.clip((values - min_db) / (max_db - min_db), 0, 1)
    return normalized


def denormalize_0_1(normalized, max_db, min_db):
    values = np.clip(normalized, 0, 1) * (max_db - min_db) + min_db
    return values


def preemphasis(wav, coeff=0.97):
    """
    Emphasize high frequency range of the waveform by increasing power(squared amplitude).

    Parameters
    ----------
    wav : np.ndarray [shape=(n,)]
        Real-valued the waveform.

    coeff: float <= 1 [scalar]
        Coefficient of pre-emphasis.

    Returns
    -------
    preem_wav : np.ndarray [shape=(n,)]
        The pre-emphasized waveform.
    """
    preem_wav = signal.lfilter([1, -coeff], [1], wav)
    return preem_wav


def inv_preemphasis(preem_wav, coeff=0.97):
    """
    Invert the pre-emphasized waveform to the original waveform.

    Parameters
    ----------
    preem_wav : np.ndarray [shape=(n,)]
        The pre-emphasized waveform.

    coeff: float <= 1 [scalar]
        Coefficient of pre-emphasis.

    Returns
    -------
    wav : np.ndarray [shape=(n,)]
        Real-valued the waveform.
    """
    wav = signal.lfilter([1], [1, -coeff], preem_wav)
    return wav


def get_random_crop(length, crop_length):
    start = np.random.choice(range(np.maximum(1, length - crop_length)), 1)[0]
    end = start + crop_length

    return start, end


def _get_mfcc_and_spec(wav, sr, n_fft, hop_length, win_length, n_mels, n_mfcc):
    # Pre-emphasis
    wav = preemphasis(wav, coeff=hparams.timit_preemphasis)

    # Get spectrogram
    # (1 + n_fft/2, t)
    spec = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(spec)

    # Get mel-spectrogram
    # (n_mels, 1+n_fft//2)
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    # mel spectrogram (n_mels, t)
    mel = np.dot(mel_basis, mag)

    # amp to db
    mag_db = amp2db(mag)
    mel_db = amp2db(mel)

    # Get mfccs
    mfccs = scipy.fftpack.dct(mel_db, axis=0, type=2, norm='ortho')[:n_mfcc]

    mag_db = normalize_0_1(mag_db, hparams.timit_max_db, hparams.timit_min_db)
    mel_db = normalize_0_1(mel_db, hparams.timit_max_db, hparams.timit_min_db)

    debug = False
    if debug:
        print("wav.shape:" + str(wav.shape))
        print("mag.shape:" + str(mag.shape))
        print("mel.shape:" + str(mel.shape))
        print("mag_db.shape:" + str(mag_db.shape))
        print("mel_db.shape:" + str(mel_db.shape))
        print("mfccs.shape:" + str(mfccs.shape))

    # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)
    return mfccs.T, mag_db.T, mel_db.T


def get_mfccs_and_phones(wav_file, trim=False, random_crop=True):

    sr = hparams.timit_sr
    n_fft = hparams.timit_n_fft
    hop_length = hparams.timit_hop_length
    win_length = hparams.timit_win_length
    n_mels = hparams.timit_n_mels
    n_mfcc = hparams.timit_n_mfcc
    default_duration = hparams.timit_default_duration

    # Load wav
    wav = read_wav(wav_file, sr)

    mfccs, _, _ = _get_mfcc_and_spec(wav, sr, n_fft, hop_length, win_length, n_mels, n_mfcc)

    # time steps
    num_time_steps = mfccs.shape[0]

    # phones (target)
    phn_file = wav_file.replace("_train.wav", ".PHN").replace("_test.wav", ".PHN")
    phn2idx, idx2phn = load_vocab()
    phones = np.zeros(shape=(num_time_steps,))
    bnd_list = []

    for line in open(phn_file, encoding='utf-8').read().splitlines():
        start_point, _, phn = line.split()
        bnd = int(start_point) // hop_length
        phones[bnd:] = phn2idx[phn]
        bnd_list.append(bnd)

    # Trim
    if trim:
        start, end = bnd_list[1], bnd_list[-1]
        mfccs = mfccs[start:end]
        phones = phones[start:end]
        assert (len(mfccs) == len(phones))

    # Random crop
    default_time_steps = (default_duration * sr) // hop_length + 1
    if random_crop:
        start, end = get_random_crop(len(mfccs), default_time_steps)
        # start = np.random.choice(range(np.maximum(1, len(mfccs) - default_time_steps)), 1)[0]
        # end = start + default_time_steps
        mfccs = mfccs[start:end]
        phones = phones[start:end]
        assert (len(mfccs) == len(phones))

    # Padding or crop
    mfccs = librosa.util.fix_length(mfccs, default_time_steps, axis=0)
    phones = librosa.util.fix_length(phones, default_time_steps, axis=0)

    debug = False
    if debug:
        print("mfccs.shape :" + str(mfccs.shape))
        print("num_time_steps :" + str(num_time_steps))
        print("default_time_steps : " + str(default_time_steps))
        print("mfccs : " + str(mfccs))
        print("phones : " + str(phones))

    return mfccs, phones


def get_mfccs_and_spectrogram(wav_file, trim=True, random_crop=False):
    sr = hparams.timit_sr
    hop_length = hparams.timit_hop_length
    win_length = hparams.timit_win_length
    n_fft = hparams.timit_n_fft
    n_mels = hparams.timit_n_mels
    n_mfcc = hparams.timit_n_mfcc
    default_duration = hparams.timit_default_duration

    # Load wav
    wav = read_wav(wav_file, sr)

    # Trim
    if trim:
        wav, _ = librosa.effects.trim(wav, frame_length=win_length, hop_length=hop_length)

    # Random crop
    if random_crop:
        start, end = get_random_crop(len(wav), sr * default_duration)
        wav = wav[start:end]

    # Padding or crop
    length = sr * default_duration
    wav = librosa.util.fix_length(wav, length)

    debug = False
    if debug:
        print("wav.shape : " + str(wav.shape))

    return _get_mfcc_and_spec(wav, sr, n_fft, hop_length, win_length, n_mels, n_mfcc)


def spec2wav(mag, n_fft, win_length, hop_length, num_iters, phase=None):
    """
        Get a waveform from the magnitude spectrogram by Griffin-Lim Algorithm.

        Parameters
        ----------
        mag : np.ndarray [shape=(1 + n_fft/2, t)]
            Magnitude spectrogram.

        n_fft : int > 0 [scalar]
            FFT window size.

        win_length  : int <= n_fft [scalar]
            The window will be of length `win_length` and then padded
            with zeros to match `n_fft`.

        hop_length : int > 0 [scalar]
            Number audio of frames between STFT columns.

        num_iters: int > 0 [scalar]
            Number of iterations of Griffin-Lim Algorithm.

        phase : np.ndarray [shape=(1 + n_fft/2, t)]
            Initial phase spectrogram.

        Returns
        -------
        wav : np.ndarray [shape=(n,)]
            The real-valued waveform.

        """
    assert (num_iters > 0)
    if phase is None:
        phase = np.pi * np.random.rand(*mag.shape)
    stft = mag * np.exp(1.j * phase)
    wav = None
    for i in range(num_iters):
        wav = librosa.istft(stft, win_length=win_length, hop_length=hop_length)
        if i != num_iters - 1:
            stft = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            _, phase = librosa.magphase(stft)
            phase = np.angle(phase)
            a, b = phase.shape
            # phase = phase.reshape(a, b, 1)
            phase = phase.reshape(a, b)
            stft = mag * np.exp(1.j * phase)
    return wav
