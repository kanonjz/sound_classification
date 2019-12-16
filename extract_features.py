import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd

AUDIO_PATH = '../ESC-50-master/audio/'


def get_features(file_name):

    if file_name:
        data, sample_rate = sf.read(file_name, dtype='float32')

    # get mfcc features
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=100)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled


def extract_features():

    sub_dirs = os.listdir(AUDIO_PATH)
    features_list = []
    for file_name in sub_dirs:
        label = file_name.split('.')[0].split('-')[-1]
        print("Extracting file ", file_name)
        try:
            mfcc = get_features(AUDIO_PATH + file_name)
            print(type(mfcc))
        except Exception as e:
            print("Extraction error")
            continue
        features_list.append([mfcc, label])
    features_df = pd.DataFrame(features_list, columns=['feature', 'class_label'])
    print(features_df.head())
    return features_df


if __name__ == '__main__':

    df = extract_features()
    df.to_csv('features.csv')
