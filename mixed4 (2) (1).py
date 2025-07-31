import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, Input, Model, callbacks
import tensorflow as tf
from scipy.signal import spectrogram
import pywt
import random
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
movements = list(range(1, 11))
data_folder = "/Users/arif/Desktop/Project 312/WyoFlex_Dataset/DIGITAL DATA"
num_participants = 28
num_cycles = 3
num_sensors = 4
num_forearms = 2
sampling_rate = 13000
wavelet = 'db4'
level = 4
fft_size = 1024  # size for FFT and reshaping
X = []
Y = []

def dwt_filter(signal, wavelet='db4'):
    """
    Function to apply wavelet filtering using Daubechies wavelet 'db4'.
    Returns approximation and detail coefficients.
    """
    # Define the wavelet filters for 'db4' (Daubechies 4)
    low_pass = np.array([0.48296, 0.8365, 0.22414, -0.12940])
    high_pass = np.array([-0.12940, -0.22414, 0.8365, -0.48296])

    # Apply convolution and downsample by 2 (for 1-level DWT)
    approx = np.convolve(signal, low_pass, mode='same')[::2]
    detail = np.convolve(signal, high_pass, mode='same')[::2]

    return approx, detail


def wavedec(signal, wavelet='db4', level=4):
    """
    Manually compute the wavelet decomposition for a signal.
    This function performs multiple levels of decomposition.
    """
    coeffs = []
    current_signal = signal

    for _ in range(level):
        # Decompose the signal at the current level
        approx, detail = dwt_filter(current_signal, wavelet)
        coeffs.append(detail)  # Append the detail coefficients
        current_signal = approx  # Update signal with the approximation

    coeffs.append(current_signal)  # Append the last approximation (after final decomposition)
    return coeffs


def coeffs_to_array(coeffs):
    """
    Manually convert wavelet coefficients to a flattened array.
    """
    arr = []
    for coeff in coeffs:
        arr.extend(coeff)  # Append all coefficients in the list
    return np.array(arr)


def dwt_transform(signal, wavelet='db4', level=4):
    coeffs = wavedec(signal, wavelet, level)
    arr = coeffs_to_array(coeffs)
    return arr[:1024].reshape((32, 32)) if arr.size >= 1024 else np.pad(arr, (0, 1024 - arr.size)).reshape((32, 32))



# Ask for signal type
while True:
    b = input("Enter 1 to load the signal with offset or 2 for signal without offset: ")
    if b in ['1', '2']:
        b = int(b)
        break
    else:
        print("Invalid input, try again.")

def augment_dwt(dwt_img):
    shift = np.random.randint(-2, 3)
    dwt_img = np.roll(dwt_img, shift, axis=1)
    noise = np.random.normal(0, 0.01, dwt_img.shape)
    return dwt_img + noise

# Load and process signals
for participant in range(1, num_participants + 1):
    for cycle in range(1, num_cycles + 1):
        for movement_index, movement in enumerate(movements):
            for forearm in range(1, num_forearms + 1):
                sensor_signals = []

                for sensor in range(1, num_sensors + 1):
                    file_name = f"P{participant}C{cycle}S{sensor}M{movement}F{forearm}O{b}"
                    file_path = os.path.join(data_folder, file_name)

                    try:
                        signal = np.loadtxt(file_path, delimiter=',')[:sampling_rate]
                        dwt_img = dwt_transform(signal, wavelet=wavelet, level=level)
                        sensor_signals.append(dwt_img)
                    except Exception as e:
                        print(f"Error loading {file_name}: {e}")
                        break

                if len(sensor_signals) == 4:
                    stacked = np.stack(sensor_signals, axis=-1)
                    X.append(stacked)
                    Y.append(movement_index)

X = np.array(X)
Y = tf.keras.utils.to_categorical(Y, num_classes=len(movements))
idx = random.randint(0, len(X) - 1)
sample_dwt = X[idx]  # shape: (32, 32, 4)
label = np.argmax(Y[idx])
import random
import os
import numpy as np
import matplotlib.pyplot as plt

# Randomly select participant, cycle, forearm, movement
person = random.randint(1, num_participants)
cycle = random.randint(1, num_cycles)
forearm = random.randint(1, num_forearms)
movement = random.randint(1, 10)

raw_signals = []
sensor_names = ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4']

for sensor in range(1, 5):
    file_name = f"P{person}C{cycle}S{sensor}M{movement}F{forearm}O{b}"
    file_path = os.path.join(data_folder, file_name)
    try:
        signal = np.loadtxt(file_path, delimiter=',')[:sampling_rate]
        raw_signals.append(signal)
    except Exception as e:
        print(f"Failed to load {file_name}: {e}")
        raw_signals.append(None)

# Plot the 4 raw signals if all loaded successfully
if all(s is not None for s in raw_signals):
    plt.figure(figsize=(12, 8))
    for i, signal in enumerate(raw_signals):
        plt.subplot(2, 2, i + 1)
        plt.plot(signal)
        plt.title(f"{sensor_names[i]} - Movement {movement}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid()
    plt.tight_layout()
    plt.suptitle(f"Participant {person}, Cycle {cycle}, Forearm {forearm}", y=1.02, fontsize=16)
    plt.show()
else:
    print("Some signals failed to load. Cannot plot all 4 sensors.")

sensor_names = ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4']
plt.figure(figsize=(12, 6))

for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(sample_dwt[:, :, i], cmap='jet', aspect='auto')
    plt.title(f"DWT Image\n{sensor_names[i]}")
    plt.axis('off')

    # Reconstruct original signal (optional)
    raw_file = None
    for ext in ['', '.txt', '.csv']:
        try:
            raw_file = os.path.join(data_folder, f"P{participant}C{cycle}S{i+1}M{label+1}F{forearm}O{b}{ext}")
            signal = np.loadtxt(raw_file, delimiter=',')[:sampling_rate]
            break
        except:
            continue

    if raw_file and 'signal' in locals():
        plt.subplot(2, 4, i + 5)
        plt.plot(signal)
        plt.title(f"Raw Signal\n{sensor_names[i]}")
        plt.xlabel("Samples")
        plt.grid()

plt.tight_layout()
plt.suptitle("Random Signal and Its DWT (All 4 Sensors)", y=1.02, fontsize=14)
plt.show()


# Pick one random sample
idx = random.randint(0, len(X) - 1)
sample_dwt = X[idx]  # shape: (32, 32, 4)
label = np.argmax(Y[idx])
sensor_idx = 0  # Choose sensor 0 (Sensor 1)

# Get corresponding raw signal file (backtrack file name based on movement label)
# This assumes your dataset is well-structured and matches your file naming
found = False
for person in range(1, num_participants + 1):
    for cycle in range(1, num_cycles + 1):
        for forearm in range(1, num_forearms + 1):
            file_name = f"P{person}C{cycle}S{sensor_idx+1}M{label+1}F{forearm}O{b}"
            file_path = os.path.join(data_folder, file_name)
            if os.path.exists(file_path):
                try:
                    raw_signal = np.loadtxt(file_path, delimiter=',')[:sampling_rate]
                    found = True
                    break
                except:
                    continue
        if found:
            break
    if found:
        break

# Re-perform DWT on that raw signal to get 1D coefficients
coeffs = pywt.wavedec(raw_signal, wavelet=wavelet, level=level)
dwt_1d = np.concatenate(coeffs)
dwt_2d = dwt_transform(raw_signal, wavelet=wavelet, level=level)

# Plotting
plt.figure(figsize=(15, 6))

# Raw signal (1D)
plt.subplot(1, 3, 1)
plt.plot(raw_signal)
plt.title("Raw EMG Signal (1D)")
plt.xlabel("Time (samples)")
plt.grid()

# DWT coefficients (1D)
plt.subplot(1, 3, 2)
plt.plot(dwt_1d)
plt.title("DWT Coefficients (1D)")
plt.xlabel("Coefficient Index")
plt.grid()

# DWT image (2D)
plt.subplot(1, 3, 3)
plt.imshow(dwt_2d, cmap='jet', aspect='auto')
plt.title("DWT as 2D Image (32×32)")
plt.colorbar()

plt.tight_layout()
plt.suptitle(f"Movement {label + 1} - Sensor {sensor_idx + 1}", y=1.05, fontsize=14)
plt.show()

# Normalize per channel
means = [X[:, :, :, i].mean() for i in range(4)]
stds = [X[:, :, :, i].std() for i in range(4)]
for i in range(4):
    X[:, :, :, i] = (X[:, :, :, i] - means[i]) / stds[i]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(10), y=np.argmax(Y, axis=1))
class_weights = dict(enumerate(class_weights))

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test),
                    class_weight=class_weights,
                    callbacks=[early_stop], verbose=1)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy For DWT')
plt.grid(True)
plt.legend()
plt.show()  

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc * 100:.2f}%")

def predict_random_sample(model, means, stds):
    while True:
        person = random.randint(1, num_participants)
        cycle = random.randint(1, num_cycles)
        forearm = random.randint(1, num_forearms)
        movement = random.randint(1, 10)
        sensor_signals = []

        for sensor in range(1, num_sensors + 1):
            file_name = f"P{person}C{cycle}S{sensor}M{movement}F{forearm}O{b}"
            file_path = os.path.join(data_folder, file_name)
            try:
                signal = np.loadtxt(file_path, delimiter=',')[:sampling_rate]
                dwt_img = dwt_transform(signal, wavelet=wavelet, level=level)
                sensor_signals.append(dwt_img)
            except:
                break

        if len(sensor_signals) == 4:
            stacked = np.stack(sensor_signals, axis=-1)
            for i in range(4):
                stacked[:, :, i] = (stacked[:, :, i] - means[i]) / stds[i]

            sample = np.expand_dims(stacked, axis=0)
            prediction = model.predict(sample)
            predicted = np.argmax(prediction)

            names = ["Extension", "Flexion", "Ulnar Deviation", "Radial Deviation",
                     "Hook Grip", "Power Grip", "Spherical Grip", "Precision Grip",
                     "Lateral Grip", "Pinch Grip"]

            print(f"\nTrue Movement Index: {movement - 1}")
            print(f"Predicted Movement Index: {predicted}")
            print(f"Predicted Movement: {names[predicted]}")
            return

predict_random_sample(model, means, stds)

# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
labels = ["Extension", "Flexion", "Ulnar Deviation", "Radial Deviation",
           "Hook Grip", "Power Grip", "Spherical Grip", "Precision Grip",
           "Lateral Grip", "Pinch Grip"]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)
plt.grid(False)
plt.title('Confusion Matrix for DWT')
plt.show() 




def fft_transform(signal, fft_size=1024):
    # Compute FFT magnitude of signal
    fft_vals = np.abs(np.fft.fft(signal, n=fft_size))
    fft_vals = fft_vals[:fft_size // 2]  # take positive frequencies
    # Normalize magnitude (optional)
    fft_vals = fft_vals / np.max(fft_vals)
    # Pad or truncate to length 1024 if needed (already 512 here)
    # For uniformity, zero pad to 1024 (32x32)
    fft_padded = np.pad(fft_vals, (0, fft_size - len(fft_vals)), 'constant')
    # Reshape to 32x32 image
    return fft_padded.reshape((32, 32))

X = []
Y = []

# Load and process signals using FFT transform
for participant in range(1, num_participants + 1):
    for cycle in range(1, num_cycles + 1):
        for movement_index, movement in enumerate(movements):
            for forearm in range(1, num_forearms + 1):
                sensor_signals = []

                for sensor in range(1, num_sensors + 1):
                    file_name = f"P{participant}C{cycle}S{sensor}M{movement}F{forearm}O{b}"
                    file_path = os.path.join(data_folder, file_name)

                    try:
                        signal = np.loadtxt(file_path, delimiter=',')[:sampling_rate]
                        fft_img = fft_transform(signal, fft_size=fft_size)
                        sensor_signals.append(fft_img)
                    except Exception as e:
                        print(f"Error loading {file_name}: {e}")
                        break

                if len(sensor_signals) == 4:
                    stacked = np.stack(sensor_signals, axis=-1)  # shape (32, 32, 4)
                    X.append(stacked)
                    Y.append(movement_index)

X = np.array(X)
Y = tf.keras.utils.to_categorical(Y, num_classes=len(movements))

# Normalize per channel
means = [X[:, :, :, i].mean() for i in range(4)]
stds = [X[:, :, :, i].std() for i in range(4)]
for i in range(4):
    X[:, :, :, i] = (X[:, :, :, i] - means[i]) / stds[i]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(10), y=np.argmax(Y, axis=1))
class_weights = dict(enumerate(class_weights))

# CNN Model (same as before)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test),
                    class_weight=class_weights,
                    callbacks=[early_stop], verbose=1)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy For FFT')
plt.grid(True)
plt.legend()
plt.show()

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc * 100:.2f}%")

# --------- Visualization for one random sample ----------

idx = random.randint(0, len(X) - 1)
sample_fft = X[idx]  # shape: (32, 32, 4)
label = np.argmax(Y[idx])
sensor_names = ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4']

plt.figure(figsize=(12, 6))

for i in range(4):
    plt.subplot(2, 4, i + 1)
    # Show unnormalized FFT magnitude image by denormalizing
    unnormalized_img = (sample_fft[:, :, i] * stds[i]) + means[i]
    plt.imshow(unnormalized_img, cmap='jet', aspect='auto')
    plt.title(f"FFT Image Before Norm\n{sensor_names[i]}")
    plt.axis('off')

    plt.subplot(2, 4, i + 5)
    plt.imshow(sample_fft[:, :, i], cmap='jet', aspect='auto')
    plt.title(f"FFT Image After Norm\n{sensor_names[i]}")
    plt.axis('off')

plt.tight_layout()
plt.suptitle("Random Signal FFT Images Before and After Normalization", y=1.02, fontsize=14)
plt.show()


X_spec, X_time, Y = [], [], []
window_size = 256
overlap = 128
# Ask for signal type

def rms(signal): return np.sqrt(np.mean(np.square(signal)))
def mav(signal): return np.mean(np.abs(signal))
def wl(signal): return np.sum(np.abs(np.diff(signal)))
def zc(signal, threshold=0.01): return np.sum(((signal[:-1] * signal[1:]) < 0) & (np.abs(signal[:-1] - signal[1:]) >= threshold))
def ssc(signal, threshold=0.01): return np.sum(((np.diff(signal[:-1]) * np.diff(signal[1:])) < 0) & (np.abs(np.diff(signal[:-1]) - np.diff(signal[1:])) >= threshold))
def var(signal): return np.var(signal)

# Load data
for i in range(1, num_participants + 1):
    for j in range(1, num_cycles + 1):
        for l, move in enumerate(movements):
            for m in range(1, num_forearms + 1):
                sensor_signals_spec = []
                sensor_features_time = []

                for k in range(1, num_sensors + 1):
                    file_name = f"P{i}C{j}S{k}M{move}F{m}O{b}"
                    file_path = os.path.join(data_folder, file_name)

                    try:
                        signal = np.loadtxt(file_path, delimiter=',')[:sampling_rate]

                        # Spectrogram
                        f, t, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=window_size, noverlap=overlap)
                        Sxx_log = np.log1p(Sxx)
                        sensor_signals_spec.append(Sxx_log)

                        # Time-domain features
                        feats = [rms(signal), mav(signal), wl(signal), zc(signal), ssc(signal), var(signal)]
                        sensor_features_time.extend(feats)

                    except Exception as e:
                        print(f"Error loading {file_name}: {e}")
                        sensor_signals_spec, sensor_features_time = [], []
                        break

                if len(sensor_signals_spec) == 4 and len(sensor_features_time) == 24:
                    min_shape = min(s.shape for s in sensor_signals_spec)
                    cropped = [s[:min_shape[0], :min_shape[1]] for s in sensor_signals_spec]
                    stacked_spec = np.stack(cropped, axis=-1)

                    if stacked_spec.shape[0] >= 64 and stacked_spec.shape[1] >= 64:
                        stacked_spec = stacked_spec[:64, :64, :]
                        X_spec.append(stacked_spec)
                        X_time.append(sensor_features_time)
                        Y.append(l)

# Convert to arrays
X_spec = np.array(X_spec)
X_time = np.array(X_time)
Y = tf.keras.utils.to_categorical(Y, num_classes=10)
idx = random.randint(0, len(X) - 1)
sample_dwt = X[idx]  # shape: (32, 32, 4)
label = np.argmax(Y[idx])

sensor_names = ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4']
plt.figure(figsize=(12, 6))

for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(sample_dwt[:, :, i], cmap='jet', aspect='auto')
    plt.title(f"STFT Image\n{sensor_names[i]}")
    plt.axis('off')

    # Reconstruct original signal (optional)
    raw_file = None
    for ext in ['', '.txt', '.csv']:
        try:
            raw_file = os.path.join(data_folder, f"P{participant}C{cycle}S{i+1}M{label+1}F{forearm}O{b}{ext}")
            signal = np.loadtxt(raw_file, delimiter=',')[:sampling_rate]
            break
        except:
            continue

    if raw_file and 'signal' in locals():
        plt.subplot(2, 4, i + 5)
        plt.plot(signal)
        plt.title(f"Raw Signal\n{sensor_names[i]}")
        plt.xlabel("Samples")
        plt.grid()

plt.tight_layout()
plt.suptitle("Random Signal and Its STFT (All 4 Sensors)", y=1.02, fontsize=14)
plt.show()


# Pick one random sample
idx = random.randint(0, len(X) - 1)
sample_dwt = X[idx]  # shape: (32, 32, 4)
label = np.argmax(Y[idx])
sensor_idx = 0  # Choose sensor 0 (Sensor 1)

from scipy.signal import stft

# Pick one random sample
idx = random.randint(0, len(X) - 1)
sample_dwt = X[idx]  # shape: (32, 32, 4)
label = np.argmax(Y[idx])
sensor_idx = 0  # Choose sensor 0 (Sensor 1)

# Get corresponding raw signal file
found = False
for person in range(1, num_participants + 1):
    for cycle in range(1, num_cycles + 1):
        for forearm in range(1, num_forearms + 1):
            file_name = f"P{person}C{cycle}S{sensor_idx+1}M{label+1}F{forearm}O{b}"
            file_path = os.path.join(data_folder, file_name)
            if os.path.exists(file_path):
                try:
                    raw_signal = np.loadtxt(file_path, delimiter=',')[:sampling_rate]
                    found = True
                    break
                except:
                    continue
        if found:
            break
    if found:
        break

# STFT
f, t, Zxx = stft(raw_signal, fs=sampling_rate, nperseg=256, noverlap=128)
stft_magnitude = np.abs(Zxx)
stft_2d = stft_magnitude[:128, :128]  # crop/pad to 128×128 for display if needed
stft_1d = stft_2d.flatten()

# Plotting
plt.figure(figsize=(15, 6))

# Raw signal (1D)
plt.subplot(1, 3, 1)
plt.plot(raw_signal)
plt.title("Raw EMG Signal (1D)")
plt.xlabel("Time (samples)")
plt.grid()

# STFT magnitude (1D)
plt.subplot(1, 3, 2)
plt.plot(stft_1d)
plt.title("STFT Magnitude (1D Flattened)")
plt.xlabel("Frequency-Time Index")
plt.grid()

# STFT Spectrogram (2D)
plt.subplot(1, 3, 3)
plt.imshow(stft_2d, cmap='jet', aspect='auto', origin='lower', extent=[t[0], t[-1], f[0], f[127]])
plt.title("STFT Spectrogram (2D)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label='Magnitude')

plt.tight_layout()
plt.suptitle(f"Movement {label + 1} - Sensor {sensor_idx + 1} (STFT View)", y=1.05, fontsize=14)
plt.show()

# Normalize
for i in range(4):
    mean = X_spec[:, :, :, i].mean()
    std = X_spec[:, :, :, i].std()
    X_spec[:, :, :, i] = (X_spec[:, :, :, i] - mean) / std

idx = random.randint(0, len(X) - 1)
normalized_sample = X[idx]  # shape: (32, 32, 4)
label = np.argmax(Y[idx])

sensor_names = ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4']
plt.figure(figsize=(14, 8))

for i in range(4):
    # Recover the original unnormalized STFT image
    unnormalized_img = (normalized_sample[:, :, i] * stds[i]) + means[i]

    # --------- Before Normalization ---------
    plt.subplot(2, 4, i + 1)
    plt.imshow(unnormalized_img, cmap='jet', aspect='auto')
    plt.title(f"Before Normalization\n{sensor_names[i]}")
    plt.axis('off')

    # --------- After Normalization ---------
    plt.subplot(2, 4, i + 5)
    plt.imshow(normalized_sample[:, :, i], cmap='jet', aspect='auto')
    plt.title(f"After Normalization\n{sensor_names[i]}")
    plt.axis('off')

plt.tight_layout()
plt.suptitle("STFT Images Before and After Normalization (All 4 Sensors)", y=1.02, fontsize=16)
plt.show()

scaler = StandardScaler()
X_time = scaler.fit_transform(X_time)

# Split
X_train_spec, X_test_spec, X_train_time, X_test_time, y_train, y_test = train_test_split(
    X_spec, X_time, Y, test_size=0.2, stratify=Y, random_state=42
)

# Hybrid Model
input_spec = Input(shape=(64, 64, 4))
x = layers.Conv2D(32, (3, 3), activation='relu')(input_spec)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling2D()(x)

input_time = Input(shape=(24,))
y = layers.Dense(64, activation='relu')(input_time)
y = layers.Dropout(0.3)(y)

combined = layers.concatenate([x, y])
z = layers.Dense(128, activation='relu')(combined)
z = layers.Dropout(0.5)(z)
output = layers.Dense(10, activation='softmax')(z)

model = Model(inputs=[input_spec, input_time], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

history = model.fit(
    [X_train_spec, X_train_time], y_train,
    validation_data=([X_test_spec, X_test_time], y_test),
    epochs=100, batch_size=32, callbacks=[early_stop], verbose=1
)

# Plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Hybrid Model Accuracy')
plt.grid(True)
plt.legend()
plt.show() 

# Evaluate
test_loss, test_acc = model.evaluate([X_test_spec, X_test_time], y_test, verbose=0)
print(f"Test Accuracy For Hybrid Model: {test_acc * 100:.2f}%")

# Save means and stds for all 4 spectrogram channels
means = [X_spec[:, :, :, i].mean() for i in range(4)]
stds = [X_spec[:, :, :, i].std() for i in range(4)]


def predict_random_sample_from_folder(model, means, stds):
    import random

    while True:
        person = random.randint(1, num_participants)
        cycle = random.randint(1, num_cycles)
        forearm = random.randint(1, num_forearms)
        movement = random.randint(1, 10)
        sensor_signals_spec = []
        sensor_features_time = []
        success = True

        print(f"\nAttempting to load: Person={person}, Cycle={cycle}, Movement={movement}, Forearm={forearm}")

        for sensor in range(1, num_sensors + 1):
            file_name = f"P{person}C{cycle}S{sensor}M{movement}F{forearm}O{b}"
            file_path = os.path.join(data_folder, file_name)

            try:
                signal = np.loadtxt(file_path, delimiter=',')[:sampling_rate]

                # Spectrogram
                f, t, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=window_size, noverlap=overlap)
                Sxx_log = np.log1p(Sxx)
                sensor_signals_spec.append(Sxx_log)

                # Time-domain features
                feats = [rms(signal), mav(signal), wl(signal), zc(signal), ssc(signal), var(signal)]
                sensor_features_time.extend(feats)

            except Exception as e:
                print(f"Could not load {file_name}: {e}")
                success = False
                break

        if not success:
            continue

        if len(sensor_signals_spec) != 4 or len(sensor_features_time) != 24:
            print("Incomplete data. Trying again...")
            continue

        # Preprocess spectrogram
        min_shape = min(s.shape for s in sensor_signals_spec)
        cropped = [s[:min_shape[0], :min_shape[1]] for s in sensor_signals_spec]
        stacked_spec = np.stack(cropped, axis=-1)

        if stacked_spec.shape[0] < 64 or stacked_spec.shape[1] < 64:
            print("Spectrogram too small. Trying again...")
            continue

        stacked_spec = stacked_spec[:64, :64, :]
        for i in range(4):
            stacked_spec[:, :, i] = (stacked_spec[:, :, i] - means[i]) / stds[i]
        stacked_spec = np.expand_dims(stacked_spec, axis=0)

        # Normalize time-domain features
        time_feats = scaler.transform([sensor_features_time])

        # Predict
        prediction = model.predict([stacked_spec, time_feats])
        predicted_label = np.argmax(prediction)
        true_label = movement - 1  # 0-indexed

        print(f"\n✅ True Movement     : {true_label} (Class {true_label + 1})")
        print(f"✅ Predicted Movement: {predicted_label} (Class {predicted_label + 1})")

        print(f"Confidence Scores   : {np.round(prediction[0], 3)}")
        break

predict_random_sample_from_folder(model, means, stds)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random

# --- Confusion Matrix ---
# Predict class labels
y_pred_probs = model.predict([X_test_spec, X_test_time])
y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Compute and plot confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
    "Extension", "Flexion", "Ulnar Dev", "Radial Dev",
    "Hook", "Power", "Spherical", "Precision", "Lateral", "Pinch"
])
disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
plt.grid(False)
plt.title("Confusion Matrix for Hybrid model for 10 class")
plt.tight_layout()
plt.show() 

#using 6 class
# Parameters
selected_movements = [1, 2, 3, 4, 6, 8]  # M1, M2, M3, M4, M6, M8
movement_to_index = {m: i for i, m in enumerate(selected_movements)}
X, Y = [], []

# Load and preprocess spectrograms
for participant in range(1, num_participants + 1):
    for cycle in range(1, num_cycles + 1):
        for movement in selected_movements:
            for forearm in range(1, num_forearms + 1):
                sensor_signals = []
                for sensor in range(1, num_sensors + 1):
                    file_name = f"P{participant}C{cycle}S{sensor}M{movement}F{forearm}O{b}"
                    file_path = os.path.join(data_folder, file_name)
                    try:
                        signal = np.loadtxt(file_path, delimiter=',')[:sampling_rate]
                        f, t, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=window_size, noverlap=overlap)
                        sensor_signals.append(np.log1p(Sxx))
                    except Exception as e:
                        print(f"Error loading {file_name}: {e}")
                        break
                if len(sensor_signals) == 4:
                    min_shape = np.min([s.shape for s in sensor_signals], axis=0)
                    resized = [s[:min_shape[0], :min_shape[1]] for s in sensor_signals]
                    stacked = np.stack(resized, axis=-1)
                    if stacked.shape[0] >= 96 and stacked.shape[1] >= 96:
                        X.append(stacked[:96, :96, :])
                        Y.append(movement_to_index[movement])


X = np.array(X)
Y = np.array(Y)

# Normalize per channel
means = [X[:, :, :, i].mean() for i in range(4)]
stds = [X[:, :, :, i].std() for i in range(4)]
for i in range(4):
    X[:, :, :, i] = (X[:, :, :, i] - means[i]) / stds[i]

# One-hot encode labels
Y_cat = tf.keras.utils.to_categorical(Y, num_classes=len(selected_movements))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.2, stratify=Y, random_state=42)

# Compute class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

# CNN model definition
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(selected_movements), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test),
                    class_weight=class_weights_dict,
                    callbacks=[early_stop], verbose=1)

# Accuracy plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.grid(True)
plt.legend()
plt.show() 

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Confusion matrix
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

movement_labels = ["Extension", "Flexion", "Ulnar Deviation", "Radial Deviation", "Power Grip", "Precision Grip"]
cm = confusion_matrix(y_true_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=movement_labels)
disp.plot(xticks_rotation=45)
plt.grid(False)
plt.show(block=False) 

# Predict a random valid sample
def predict_random_sample_from_folder(model, means, stds):
    while True:
        person = random.randint(1, num_participants)
        cycle = random.randint(1, num_cycles)
        forearm = random.randint(1, num_forearms)
        movement = random.choice(selected_movements)
        sensor_signals, file_names = [], []

        for sensor in range(1, num_sensors + 1):
            file_name = f"P{person}C{cycle}S{sensor}M{movement}F{forearm}O{b}"
            file_path = os.path.join(data_folder, file_name)
            file_names.append(file_name)
            try:
                signal = np.loadtxt(file_path, delimiter=',')[:sampling_rate]
                f, t, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=window_size, noverlap=overlap)
                sensor_signals.append(np.log1p(Sxx))
            except Exception as e:
                print(f"Skipping {file_name}: {e}")
                break

        if len(sensor_signals) == 4:
            print(f"\nLoaded files:")
            for name in file_names:
                print(name)
            print(f"True Movement Index: {movement_to_index[movement]}")

            min_shape = min(s.shape for s in sensor_signals)
            cropped = [s[:min_shape[0], :min_shape[1]] for s in sensor_signals]
            stacked = np.stack(cropped, axis=-1)[:96, :96, :]

            for i in range(4):
                stacked[:, :, i] = (stacked[:, :, i] - means[i]) / stds[i]

            sample = np.expand_dims(stacked, axis=0)
            prediction = model.predict(sample)
            predicted_label = np.argmax(prediction)

            movement_names = {
                1: "Extension", 2: "Flexion", 3: "Ulnar Deviation", 4: "Radial Deviation",
                6: "Power Grip", 8: "Precision Grip"
            }

            print(f"\nPredicted Movement Index: {predicted_label}")
            predicted_movement = selected_movements[predicted_label]
            print(f"Predicted Movement: {movement_names[predicted_movement]}")
            return

# Run prediction
tf.keras.backend.clear_session()
predict_random_sample_from_folder(model, means, stds) 
