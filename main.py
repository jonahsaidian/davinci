# %%

import queue
import threading
from tkinter import Tk, ttk
from tkinter.filedialog import askopenfilename

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from pedalboard.io import AudioFile
from scipy.fft import fft
from scipy.interpolate import RegularGridInterpolator

# %%
davinci = Tk()
davinci.title("Davinci")
davinci.geometry("350x100")
frm = ttk.Frame(davinci, padding=10, height=200, width=350)
frm.place(x=0, y=0)
q = queue.Queue()
progressbar = ttk.Progressbar(frm, mode="determinate", maximum=100)
q.put(progressbar)


# %%
def make_plot(best_freqs, amps, interp_factor):
    progressbar = q.get()
    best_freqs = np.array(best_freqs)[amps > 0]
    amps = amps[amps > 0]
    n_freqs = len(best_freqs)
    side = int(np.ceil(np.sqrt(n_freqs)))
    xx = np.linspace(0, side * interp_factor, side)
    yy = np.linspace(0, side * interp_factor, side)
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

    interp_a = RegularGridInterpolator((xx, yy), data_a, "slinear")
    interp_r = RegularGridInterpolator((xx, yy), data_r, "slinear")
    interp_g = RegularGridInterpolator((xx, yy), data_g, "slinear")
    interp_b = RegularGridInterpolator((xx, yy), data_b, "slinear")
    data = np.ones((side * interp_factor, side * interp_factor)) * np.nan
    c_dict = {}
    index = 0
    for row in range(0, data.shape[0]):
        # print(f"{100*row/data.shape[0]} \n")
        progressbar.step(100 / data.shape[0])
        davinci.update()
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
    progressbar.step(99.9)
    frm.update()
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
    frm.destroy()
    davinci.update()
    l3 = ttk.Label(
        davinci,
        text="Enjoy your musical artwork. You may now save the image below or close this application. \n",
        font=("Arial", 15),
        wraplength=600,
        justify="center",
    )
    l3.place(x=50, y=0, width=600)
    canvas = FigureCanvasTkAgg(fig, master=davinci)
    ax.imshow(
        data,
        interpolation="none",
        cmap=my_cmap,
        extent=[0, side * interp_factor, 0, side * interp_factor],
        zorder=0,
    )
    # turn off the axis labels
    ax.axis("off")
    canvas.draw()
    canvas.get_tk_widget().place(x=10, y=80)

    davinci.geometry(f"700x600")

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas, davinci)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().place(x=20, y=80)


# fp = os.path.join(os.path.abspath("."), "Mozart.mp3")
def loadaudiofile():
    progressbar = q.get()
    fp = askopenfilename()
    b1["state"] = "disabled"
    l1["text"] = "Please wait as your audio file is processed."
    # progressbar = ttk.Progressbar(frm,mode="determinate",maximum=100)
    progressbar.place(x=70, y=60, width=200)
    l2 = ttk.Label(frm, text="Progress: ")
    # l2.grid(row=1,column=1)
    l2.place(x=10, y=60, width=60)
    davinci.geometry("350x100")
    q.put(progressbar)
    with AudioFile(fp) as af:
        sample_rate = af.samplerate
        song_length = af.frames
        song = af.read(song_length)
        channels = af.num_channels
        # chunk = af.read(af.samplerate)
    song_flat = np.sum(song, 0) / channels
    scale_factor = max(1, round(240 * sample_rate / song_length))
    interp_factor = 5 + round(240 * sample_rate / song_length)
    window = sample_rate // scale_factor
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
        if np.max(amplitude_spectrum) > 0:
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
        dominant_freq_indices = np.where(
            amplitude_spectrum[: window // 2] >= threshold
        )[0]
        best_freq_index = np.where(
            amplitude_spectrum[: window // 2] == amplitude_spectrum[: window // 2].max()
        )[0]
        freqs[dominant_freq_indices]

        # print("Dominant Frequencies: ", dominant_freqs)
        if best_freq_index.any():
            best_freqs.append(freqs[best_freq_index][0])
        else:
            best_freqs.append(0)
    amps = amps / max(amps)
    make_plot(best_freqs, amps, interp_factor)


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
    new_color = color[:3] + (0.5 + amp_scale / 2,)
    return new_color


# %%
l1 = ttk.Label(
    frm, text="Welcome to Davinci, where your music can become art!\n", justify="center"
)
l1.place(x=0, y=0)
b1 = ttk.Button(
    frm, text="Load File", command=threading.Thread(target=loadaudiofile).start
)
b1.place(x=90, y=20)
davinci.mainloop()
# %%
# for octave in notes.keys():
#     for i, note in enumerate(notes[octave]):
#         plt.scatter(octave,i,c=note_to_color(note))
# %%
