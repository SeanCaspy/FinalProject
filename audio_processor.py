from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pyaudio
import struct
import scipy.fftpack as spfft
import threading
import cv2

app = Flask(__name__)
CORS(app)

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

# Initialize global variables for threading and PyAudio
stop_event = threading.Event()
processing_thread = None
stream = None
p = None

# Define matrices and transform functions
def Dmatrix(samples):
    out = np.zeros(N)
    out[0:int(N / 2)] = Dmatrix.z
    Dmatrix.z = samples[0:int(N / 2)]
    out[int(N / 2):N] = samples[int(N / 2):N]
    return out

Dmatrix.z = np.zeros(int(N / 2))

def Dmatrixinv(samples):
    out = np.zeros(N)
    out[int(N / 2):N] = Dmatrixinv.z
    Dmatrixinv.z = samples[int(N / 2):N]
    out[0:int(N / 2)] = samples[0:int(N / 2)]
    return out

Dmatrixinv.z = np.zeros(int(N / 2))

fcoeff = np.sin(np.pi / (2 * N) * (np.arange(0, 2 * N) + 0.5))
Fmatrix = np.zeros((N, N))
Fmatrix[0:int(N / 2), 0:int(N / 2)] = np.fliplr(np.diag(fcoeff[0:int(N / 2)]))
Fmatrix[int(N / 2):N, 0:int(N / 2)] = np.diag(fcoeff[int(N / 2):N])
Fmatrix[0:int(N / 2), int(N / 2):N] = np.diag(fcoeff[N:int(N + N / 2)])
Fmatrix[int(N / 2):N, int(N / 2):N] = -np.fliplr(np.diag(fcoeff[int(N + N / 2):(2 * N)]))
Finv = np.linalg.inv(Fmatrix)

def DCT4(samples):
    samplesup = np.zeros(2 * N)
    samplesup[1::2] = samples
    y = spfft.dct(samplesup, type=3) / 2
    return y[0:N]

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

def run_mdct(stop_event):
    global stream, p
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK_SIZE)

    while not stop_event.is_set():
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        shorts = struct.unpack('h' * CHUNK_SIZE, data)
        samples = np.array(list(shorts), dtype=float)

        frame[0:(rows - 1), :] = frame[1:rows, :]
        y = MDCT(samples[0:fftlen])

        # Noise Reduction Filter
        threshold = 10
        yfilt = y * (np.abs(y) > threshold)

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

@app.route('/start', methods=['POST'])
def start_audio_processing():
    global processing_thread, stop_event

    if processing_thread is None or not processing_thread.is_alive():
        stop_event.clear()
        processing_thread = threading.Thread(target=run_mdct, args=(stop_event,))
        processing_thread.start()
        return jsonify({"status": "Audio processing started"}), 200
    else:
        return jsonify({"status": "Audio processing is already running"}), 400

@app.route('/stop', methods=['POST'])
def stop_audio_processing():
    global stop_event, processing_thread

    if processing_thread is not None and processing_thread.is_alive():
        stop_event.set()
        processing_thread.join()
        return jsonify({"status": "Audio processing stopped"}), 200
    else:
        return jsonify({"status": "Audio processing is not running"}), 400

@app.route('/')
def index():
    return send_from_directory('venv', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
