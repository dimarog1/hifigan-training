import numpy as np
from tqdm import tqdm
from meldataset import mel_spectrogram
import argparse
import torchaudio
import os
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--wavs_dir', default=None)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--method', default='aggressive')

    a = parser.parse_args()

    method = a.method

    for number in tqdm(range(len(os.listdir(a.wavs_dir)))):
        # audio, _ = torchaudio.load(a.wavs_dir + f'/{method if method == "aggressive" else "smoothed"}{number}.wav')
        audio, _ = torchaudio.load(a.wavs_dir + f'/{"orig"}{number}.wav')
        mel_splice = mel_spectrogram(audio, 1024, 80, 22050, 256, 1024, 0, 8000, center=False)

        np.save(a.save_dir + f'/orig{number}.npy', mel_splice)

    print('done')


if __name__ == '__main__':
    main()
