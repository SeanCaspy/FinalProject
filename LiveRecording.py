import numpy as np
import scipy.signal as signal
import pyaudio
import struct
import scipy.fftpack as spfft
from ipywidgets import ToggleButton, Button
from ipywidgets import HBox
import threading
import cv2

# Signal Processing Parameters
N = 576  # Number of subbands and block size
CHUNK_SIZE = N
FORMAT = pyaudio.paInt16  # Conversion format for PyAudio stream
CHANNELS = 1  # Audio Channels
RATE = 32000  # Sampling Rate in Hz
FFT_LEN = N  # FFT Length

rows = 500
cols = CHUNK_SIZE
fftlen = cols
frame = 0.0 * np.ones((rows, cols, 3))


# The D(z) matrix:
def Dmatrix(samples):
    # implementation of the delay matrix D(z)
    # Delay elements:
    out = np.zeros(N)
    out[0:int(N / 2)] = Dmatrix.z
    Dmatrix.z = samples[0:int(N / 2)]
    out[int(N / 2):N] = samples[int(N / 2):N]
    return out


Dmatrix.z = np.zeros(int(N / 2))


# The inverse D(z) matrix:
def Dmatrixinv(samples):
    # implementation of the delay matrix D(z)
    # Delay elements:
    out = np.zeros(N)
    out[int(N / 2):N] = Dmatrixinv.z
    Dmatrixinv.z = samples[int(N / 2):N]
    out[0:int(N / 2)] = samples[0:int(N / 2)]
    return out


Dmatrixinv.z = np.zeros(int(N / 2))

# The F Matrix:
fcoeff = np.sin(np.pi / (2 * N) * (np.arange(0, 2 * N) + 0.5))
Fmatrix = np.zeros((N, N))
Fmatrix[0:int(N / 2), 0:int(N / 2)] = np.fliplr(np.diag(fcoeff[0:int(N / 2)]))
Fmatrix[int(N / 2):N, 0:int(N / 2)] = np.diag(fcoeff[int(N / 2):N])
Fmatrix[0:int(N / 2), int(N / 2):N] = np.diag(fcoeff[N:int(N + N / 2)])
Fmatrix[int(N / 2):N, int(N / 2):N] = -np.fliplr(np.diag(fcoeff[int(N + N / 2):(2 * N)]))
# The inverse F matrix:
Finv = np.linalg.inv(Fmatrix)


# The DCT4 transform:
def DCT4(samples):
    # use a DCT3 to implement a DCT4:
    samplesup = np.zeros(2 * N)
    # upsample signal:
    samplesup[1::2] = samples
    y = spfft.dct(samplesup, type=3) / 2
    return y[0:N]


# The complete MDCT, Analysis:
def MDCT(samples):
    y = np.dot(samples, Fmatrix)
    y = Dmatrix(y)
    y = DCT4(y)
    return y


def MDCTinv(y):
    x = DCT4(y) * 2 / N
    x = Dmatrixinv(x)
    x = np.dot(x, Finv)
    return x


toggle_run = ToggleButton(description='Stop')
button_start = Button(description='Start')


def start_button(button_start):
    thread.start()
    button_start.disabled = True


button_start.on_click(start_button)


def on_click_toggle_run(change):
    if not change['new']:
        stream.stop_stream()
        stream.close()
        p.terminate()
        cv2.destroyAllWindows()


toggle_run.observe(on_click_toggle_run, 'value')

box_buttons = HBox([button_start, toggle_run])


def run_mdct(toggle_run):
    while True:
        if toggle_run.value:
            break

        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

        shorts = (struct.unpack('h' * CHUNK_SIZE, data))
        samples = np.array(list(shorts), dtype=float)

        frame[0:(rows - 1), :] = frame[1:rows, :]
        y = MDCT(samples[0:fftlen])

        yfilt = y

        R = 0.25 * np.log((np.abs(yfilt / np.sqrt(fftlen)) + 1)) / np.log(10.0)

        frame[rows - 1, :, 2] = R

        frame[rows - 1, :, 1] = np.abs(1 - 2 * R)

        frame[rows - 1, :, 0] = 1.0 - R

        cv2.imshow('frame', frame)

        xrek = MDCTinv(yfilt).astype(int)
        xrek = np.clip(xrek, -32000, 32000)
        data = struct.pack('h' * len(xrek), *xrek)
        stream.write(data, CHUNK_SIZE)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop_stream()
    stream.close()
    p.terminate()
    cv2.destroyAllWindows()


p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK_SIZE)

if __name__ == '__main__':
    run_mdct(toggle_run)
