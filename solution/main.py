import numpy as np
from scipy.signal import hann
from scipy.io import wavfile
import scipy.io.wavfile as wav


def phase_vocoder(inputVector, speed, win_size=1024, hop=256):
    hop_out = round(hop / speed)

    wn = hann(win_size * 2 + 1)[1: win_size + 1]

    x = np.concatenate((np.zeros(2 * hop), inputVector))

    N = int((len(x) - win_size) / hop)

    x = x[: N * hop + win_size]

    times_start = np.arange(N) * hop
    splits_x = np.zeros((N, win_size))
    for i in range(N):
        splits_x[i, :] = x[times_start[i]: times_start[i] + win_size]

    proced_y = np.zeros((N, win_size))

    cum_phase = 0
    phase_prev = 0

    for index in range(N):
        cur_frame = splits_x[index, :] * wn / np.sqrt(2)
        cur_frame_fft = np.fft.fft(cur_frame)

        phase_cur = np.angle(cur_frame_fft)
        delta_phi = phase_cur - phase_prev
        phase_prev = phase_cur

        freq_dev = delta_phi - 2 * hop * np.pi * np.arange(win_size) / win_size
        delta_phi_wrapped = np.mod(freq_dev + np.pi, 2 * np.pi) - np.pi
        true_freq = 2 * np.pi * np.arange(win_size) / win_size + delta_phi_wrapped / hop

        cum_phase += hop_out * true_freq

        new_frame = np.real(np.fft.ifft(np.abs(cur_frame_fft) * np.exp(1j * cum_phase)))

        proced_y[index, :] = new_frame * wn / np.sqrt(win_size / (2 * hop_out))

    output_record = np.zeros(proced_y.shape[0] * hop_out + proced_y.shape[1] - hop_out)
    ind = 0
    for cur_frame in proced_y:
        output_record[ind: ind + proced_y.shape[1]] += cur_frame
        ind += hop_out

    return output_record




fs, x = wavfile.read("test_mono.wav")
stretched_audio = phase_vocoder(x, 0.5)

wav.write('test_mono_r05.wav', fs, stretched_audio)

stretched_audio = phase_vocoder(x, 2)

wav.write('test_mono_r2.wav', fs, stretched_audio)