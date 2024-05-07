import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

for subdir, dirs, files in os.walk('sounds'):
    for file in files:
        if file[-4:] == '.wav':
            print(f'Computing {file}... ', end='')
            file_name = os.path.join(subdir, file)

            plt.figure(figsize=(1, 1))
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)

            y, sr = librosa.load(file_name, mono=True)

            p = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1, n_mels=512, win_length=1024)
            librosa.display.specshow(librosa.power_to_db(p, ref=np.max), x_axis='s', y_axis='mel', cmap='binary_r')

            plt.savefig(f'images/{file[0]}/{file[:-4]}.png')
            plt.close()

            print('Done.')
