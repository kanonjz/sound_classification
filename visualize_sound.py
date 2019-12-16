import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import argparse

AUDIO_PATH = '../ESC-50-master/audio/'


def visualize_wav(wav_file):
    spf = wave.open(AUDIO_PATH + wav_file)
    sample_rate = spf.getframerate()
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, 'Int16')

    if spf.getnchannels() == 2:
        print('just mono files. not stereo')
        sys.exit(0)

    # plotting x axis in seconds. create time vector spaced linearly with size of audio file.
    # divide size of signal by frame rate to get stop limit
    time = np.linspace(0, len(signal) / sample_rate, num=len(signal))
    plt.figure(1)
    plt.title('Signal Wave Vs Time(in sec)')
    plt.plot(time, signal)
    plt.savefig('sample_wav/sample_waveplot_Fire.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_file', help='Relative path to a .wav file', required=True)
    args = parser.parse_args()
    visualize_wav(args.wav_file)
