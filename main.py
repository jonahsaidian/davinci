# %%
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pedalboard.io import AudioFile
from scipy.fft import fft
from scipy.interpolate import RegularGridInterpolator

# %%
fp = os.path.join(os.path.abspath("."), "HappyBirthday.mp3")
with AudioFile(fp) as af:
    sample_rate = af.samplerate
    song_length = af.frames
    song = af.read(2717568)
    channels = af.num_channels
    # chunk = af.read(af.samplerate)
song_flat = np.sum(song, 0) / channels
# %%
window = sample_rate // 3
best_freqs = []
amps = []
for i in range(song_length // window):
    # i = 200
    chunk = song_flat[i * window : (i + 1) * window]
    if len(chunk) == 0:
        break
    amps.append(max(chunk))
    mystery_signal_fft = fft(chunk)

    # Compute the amplitude spectrum
    amplitude_spectrum = np.abs(mystery_signal_fft)

    # Normalize the amplitude spectrum
    amplitude_spectrum = amplitude_spectrum / np.max(amplitude_spectrum)

    # Compute the frequency array
    freqs = np.fft.fftfreq(chunk.shape[0], 1 / sample_rate)

    # Plot the amplitude spectrum in the frequency domain
    # plt.plot(freqs[:window // 2], amplitude_spectrum[:window // 2])
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Normalized Amplitude")
    # plt.title("Amplitude Spectrum of the Mystery Signal")
    # plt.show()

    # Find the dominant frequencies
    threshold = 0.5
    dominant_freq_indices = np.where(amplitude_spectrum[: window // 2] >= threshold)[0]
    best_freq_index = np.where(
        amplitude_spectrum[: window // 2] == amplitude_spectrum[: window // 2].max()
    )[0]
    dominant_freqs = freqs[dominant_freq_indices]

    # print("Dominant Frequencies: ", dominant_freqs)
    if best_freq_index.any():
        best_freqs.append(freqs[best_freq_index][0])
    else:
        best_freqs.append(0)
amps = amps / max(amps)
# %%
octaves_start = np.geomspace(16.35, 4186.01, 9)
octaves_end = np.geomspace(30.87, 7902.13, 9)
notes = {}
for i in range(len(octaves_start)):
    notes[i] = np.geomspace(octaves_start[i], octaves_end[i], 12)


def find_note(f):
    min_f = octaves_start[0]
    max_f = octaves_end[-1]
    if f < min_f:
        return [0, 0]
    if f > max_f:
        return [8, 12]
    o = 0
    while f > octaves_end[o]:
        o += 1
    sub_notes = notes[o]
    n = np.abs(sub_notes - f).argmin()
    return [o, n]


# %%
n_octaves = len(octaves_start)


def pair_to_color(pair):
    scale = (pair[0] + 7) / 15
    color = mpl.colormaps["gist_rainbow"].resampled(9 * 12)(pair[1] * 9 + pair[0])
    new_color = (scale * color[0], scale * color[1], scale * color[2], color[3])
    return new_color


def note_to_color(f):
    pair = find_note(f)
    return pair_to_color(pair)


def sound_to_color(f, amp_scale):
    color = note_to_color(f)
    new_color = color[:3] + (amp_scale,)
    return new_color


# %%
# for octave in notes.keys():
#     for i, note in enumerate(notes[octave]):
#         plt.scatter(octave,i,c=note_to_color(note))
# %%
best_freqs = np.array(best_freqs)[amps > 0]
amps = amps[amps > 0]
n_freqs = len(best_freqs)
side = int(np.ceil(np.sqrt(n_freqs)))
# %%
xx = np.linspace(0, side**2, side)
yy = np.linspace(0, side**2, side)
data_ = np.ones((len(xx), len(yy)))
data_r = np.ones((len(xx), len(yy)))
data_g = np.ones((len(xx), len(yy)))
data_b = np.ones((len(xx), len(yy)))
data_a = np.ones((len(xx), len(yy)))

for row in range(0, data_.shape[0]):
    for col in range(0, data_.shape[1]):
        index = (row * side) + (col)
        if index < n_freqs - 1:
            f = best_freqs[index]
            amp_scale = amps[index]
            color = sound_to_color(f, amp_scale)
            data_r[row][col] = color[0]
            data_g[row][col] = color[1]
            data_b[row][col] = color[2]
            data_a[row][col] = color[3]

interp_r = RegularGridInterpolator((xx, yy), data_r)
interp_g = RegularGridInterpolator((xx, yy), data_g)
interp_b = RegularGridInterpolator((xx, yy), data_b)
interp_a = RegularGridInterpolator((xx, yy), data_a)
# %%
data = np.ones((side**2, side**2)) * np.nan
c_dict = {}
index = 0
for row in range(0, data.shape[0]):
    for col in range(0, data.shape[1]):
        r = interp_r([row, col])[0]
        g = interp_g([row, col])[0]
        b = interp_b([row, col])[0]
        a = interp_a([row, col])[0]
        color = (r, g, b, a)
        if not color in c_dict.keys():
            n = len(c_dict)
            c_dict[color] = n
        data[row][col] = c_dict[color]
# %%
# make a figure + axes
fig, ax = plt.subplots(1, 1, tight_layout=True)
# make color map
my_cmap = mpl.colors.ListedColormap(list(c_dict.keys()))
# set the 'bad' values (nan) to be white and transparent
my_cmap.set_bad(color="w", alpha=0)
# draw the grid
for x in range(side + 1):
    ax.axhline(x, lw=0, color="k", zorder=5)
    ax.axvline(x, lw=0, color="k", zorder=5)
# draw the boxes
ax.imshow(data, interpolation="none", cmap=my_cmap, extent=[0, side, 0, side], zorder=0)
# turn off the axis labels
ax.axis("off")

# %%
