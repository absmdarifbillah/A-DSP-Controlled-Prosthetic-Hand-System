import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from scipy.signal import spectrogram

# Parameters
movements = list(range(1, 11))
data_folder = "/Users/arif/Desktop/Project 312/WyoFlex_Dataset/DIGITAL DATA"
num_participants = 28
num_cycles = 3
num_sensors = 4
num_forearms = 2
sampling_rate = 13000
window_size = 256
overlap = 128

X = []
Y = []

# Ask for signal type
while True:
    b = input("Enter 1 to load the signal with offset or 2 for signal without offset: ")
    if b in ['1', '2']:
        b = int(b)
        break
    else:
        print("Invalid input, try again.")
def fft_custom(x):
    x = np.asarray(x, dtype=np.complex128)
    N = x.size

    if N == 0:
        return np.array([], dtype=np.complex128)
    if N == 1:
        return x

    if is_power_of_two(N):
        return iterative_fft(x)
    else:
        return bluestein_fft(x)

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def bluestein_fft(x):
    N = len(x)
    a = np.asarray(x, dtype=np.complex128)
    n = np.arange(N)
    chirp = np.exp(-1j * np.pi * (n ** 2) / N)

    a_mod = a * chirp

    M = 2 ** int(np.ceil(np.log2(2 * N - 1)))
    A = np.zeros(M, dtype=np.complex128)
    B = np.zeros(M, dtype=np.complex128)

    A[:N] = a_mod
    B[:N] = np.exp(1j * np.pi * (n ** 2) / N)
    B[-(N - 1):] = B[1:N][::-1]

    FA = iterative_fft(A)
    FB = iterative_fft(B)
    FC = FA * FB
    c = iterative_ifft(FC)

    return c[:N] * chirp

def iterative_fft(x):
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    j = bit_reverse_indices(N)
    x = x[j]

    stages = int(np.log2(N))
    for s in range(1, stages + 1):
        m = 2 ** s
        wm = np.exp(-2j * np.pi / m)
        for k in range(0, N, m):
            w = 1
            for j in range(m // 2):
                t = w * x[k + j + m // 2]
                u = x[k + j]
                x[k + j] = u + t
                x[k + j + m // 2] = u - t
                w *= wm
    return x

def iterative_ifft(X):
    X = np.asarray(X, dtype=np.complex128)
    return np.conj(iterative_fft(np.conj(X))) / X.size

def bit_reverse_indices(n):
    bits = int(np.log2(n))
    indices = np.arange(n)
    reversed_indices = indices.copy()
    for i in range(n):
        b = format(i, f'0{bits}b')[::-1]
        reversed_indices[i] = int(b, 2)
    return reversed_indices

# Define manual_spectrogram function OUTSIDE the loops
def manual_spectrogram(signal, fs, nperseg=256, noverlap=128):
    """
    Computes spectrogram manually without scipy's spectrogram function
    """
    #  Create window function (Hamming window)
    window = np.hamming(nperseg)
    
    #  Calculate number of segments
    step = nperseg - noverlap
    n_segments = 1 + (len(signal) - nperseg) // step
    
    #  Initialize spectrogram matrix
    Sxx = np.zeros((nperseg // 2 + 1, n_segments))
    
    #  Compute FFT for each segment
    for seg_idx in range(n_segments):
        # Extract segment with overlap
        start = seg_idx * step
        segment = signal[start:start + nperseg]
        
        # Apply window function
        windowed = segment * window
        
        # Compute FFT (real-valued FFT)
        fft_result_full = fft_custom(windowed)
        fft_result = fft_result_full[:nperseg // 2 + 1]
        
        # Compute power spectral density
        Sxx[:, seg_idx] = np.abs(fft_result) ** 2
    
    #  Create frequency and time vectors
    f = np.fft.rfftfreq(nperseg, d=1/fs)
    t = np.arange(n_segments) * step / fs
    
    #  Scale for power spectral density
    scale = 1.0 / (fs * (window**2).sum())
    Sxx *= scale
    
    return f, t, Sxx

# Load and process signals
for i in range(1, num_participants + 1):
    for j in range(1, num_cycles + 1):
        for l, move in enumerate(movements):
            for m in range(1, num_forearms + 1):
                sensor_signals = []
                
                for k in range(1, num_sensors + 1):
                    file_name = f"P{i}C{j}S{k}M{move}F{m}O{b}"
                    file_path = os.path.join(data_folder, file_name)
                    
                    try:
                        # Load signal
                        signal = np.loadtxt(file_path, delimiter=',')[:sampling_rate]
                        
                        # Compute STFT manually
                        f, t, Sxx = manual_spectrogram(
                            signal, 
                            fs=sampling_rate,
                            nperseg=window_size,
                            noverlap=overlap
                        )
                        
                        # Apply log scaling
                        Sxx_log = np.log1p(Sxx)  # Log scale for better contrast
                        sensor_signals.append(Sxx_log)
                        
                    except Exception as e:
                        print(f"Error loading {file_name}: {e}")
                        continue
                
                # After processing all 4 sensors
                if len(sensor_signals) == 4:
                    # Find minimum dimensions
                    min_shape = min(s.shape for s in sensor_signals)
                    
                    # Crop all spectrograms to minimum dimensions
                    cropped = [s[:min_shape[0], :min_shape[1]] for s in sensor_signals]
                    
                    # Stack sensors as 4-channel spectrogram
                    stacked = np.stack(cropped, axis=-1)  # Shape: (freq, time, 4)
                    
                    # Ensure minimum size and crop to 64x64
                    if stacked.shape[0] >= 64 and stacked.shape[1] >= 64:
                        stacked = stacked[:64, :64, :]  # Crop to 64x64x4
                        X.append(stacked)
                        Y.append(l)
# Convert to arrays
X = np.array(X)
Y = tf.keras.utils.to_categorical(Y, num_classes=10)

# Normalize data per channel
for i in range(4):
    mean = X[:, :, :, i].mean()
    std = X[:, :, :, i].std()
    X[:, :, :, i] = (X[:, :, :, i] - mean) / std

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# CNN Model for 2D spectrograms with 4 channels
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

# Evaluate
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.grid(True)
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
