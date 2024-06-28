# Import necessary libraries
import wfdb
import os
from google.colab import drive
import zipfile

drive.mount('/content/gdrive') # Pair Drive with Colab

zip_path = '/content/gdrive/MyDrive/mit-bih-arrhythmia-database-1.0.0.zip' # Get the Dataset zip file path
extract_dir = '/content/' # Select where to extract the file

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir) # Unzip Dataset file


# Get Dataset's file path
data_dir = '/content/mit-bih-arrhythmia-database-1.0.0'

# Get a list of all the record names in the directory
records = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.dat')]

records.sort(key=int) # Sort the records based on the record number

ecg_signals = {} # Create an empty dictionary


for record_name in records: # Loop over each record in the dataset
    record = wfdb.rdrecord(os.path.join(data_dir, record_name)) # Read the record
    ecg_signals[record_name] = record.p_signal # Store the whole signal in the dictionary

import numpy as np

def moving_average(signal, window_size):
    window = np.ones(int(window_size))/float(window_size) # Create a one-dimensional window
    smoothed_signal = np.convolve(signal, window, 'same') # Convolve the signal with the window
    return smoothed_signal

window_size = 3 # Define the window size for the moving average filter

# Apply the moving average filter to each record in the ECG signals dictionary
smoothed_ecg_signals = {} # Create an empty dictionary
for record, signal in ecg_signals.items():
    smoothed_signal1 = moving_average(signal[:, 0], window_size) # Apply to the first lead
    smoothed_signal2 = moving_average(signal[:, 1], window_size) # Apply to the second lead
    smoothed_ecg_signals[record] = np.column_stack((smoothed_signal1, smoothed_signal2)) # combine the two processed leads

from scipy.signal import butter, filtfilt

N = 5 # Order of filter
Wn = 25  # Cutoff frequency in Hz

# Create the Butterworth filter
b, a = butter(N, Wn, fs=360, btype='low', analog=False)

# Apply the filter to each record in the ECG signals dictionary
filtered_signals = {} # Create an empty dictionary
for record, signal in smoothed_ecg_signals.items():
    filtered_signal = filtfilt(b, a, signal, axis=0)
    filtered_signals[record] = filtered_signal

normalized_signals = {} # Create an empty dictionary

# Loop over each record in the 'filtered_signals' dictionary and normalize each lead separately
for record_name, signal in filtered_signals.items():
    normalized_signal_lead1 = (signal[:, 0] - np.min(signal[:, 0])) / (np.max(signal[:, 0]) - np.min(signal[:, 0])) * 2 - 1
    normalized_signal_lead2 = (signal[:, 1] - np.min(signal[:, 1])) / (np.max(signal[:, 1]) - np.min(signal[:, 1])) * 2 - 1

    normalized_signal = np.column_stack((normalized_signal_lead1, normalized_signal_lead2)) # Combine the processed leads
    normalized_signals[record_name] = normalized_signal # Store normalized signals in the new normalization dictionary

# Function to get the annotations object ( record_name, extension, sample, symbol, aux_note)
def get_annotations(data_dir, records):
    annotations = {}
    for record_name in records:
        annotation = wfdb.rdann(os.path.join(data_dir, record_name), 'atr')
        annotations[record_name] = annotation
    return annotations

# Function to filter out Non_beat annotations and only keep the beat annotations
def filter_annotations(annotations, wanted_symbols):
    filtered_annotations = {}
    for record_name in annotations:
        record_annotations = annotations[record_name]
        filtered_record_annotations = []
        filtered_samples = []
        filtered_aux_notes = []
        for i, symbol in enumerate(record_annotations.symbol):
            if symbol in wanted_symbols:
                filtered_record_annotations.append(symbol)
                filtered_samples.append(record_annotations.sample[i])
                filtered_aux_notes.append(record_annotations.aux_note[i])
        # Create a new Annotation object with filtered symbols, samples, and aux_notes
        filtered_annotation = wfdb.Annotation(record_name=record_annotations.record_name,
                                              extension=record_annotations.extension,
                                              sample=filtered_samples,
                                              symbol=filtered_record_annotations,
                                              aux_note=filtered_aux_notes)
        filtered_annotations[record_name] = filtered_annotation
    return filtered_annotations


beat_ann = ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'F', 'e', 'j', 'E', '/', 'f', 'Q'] # list of beat annotations

annotations = get_annotations(data_dir, records) # Upload annotations object
f_annotations = filter_annotations(annotations, beat_ann) # Filter annotations based on 'beat_ann' list

def segment_ecg_signals(ecg_signals, annotations):
    segmented_ecg_signals = {} # Create an empty dict to store segments (heartbeats)

    # Loop over each record and get the signal and the corresponding annotation object
    for record_name in ecg_signals:
        signal = ecg_signals[record_name] # Get signal
        annotation = annotations[record_name] # Get annotation object
        segments = [] # Create a list to store segments 'heartbeats' of the current record
        for loc in annotation.sample: # Get the R-peak location based on the annotation location
            start = max(0, loc - 60) # 60 samples before R-peak
            end = min(len(signal), loc + 90) # 90 samples after R-peak
            segment = signal[start:end] # Create the segment (heartbeat)
            segments.append(segment) # Append heartbeat to the list
        segmented_ecg_signals[record_name] = segments # store the list of segments (heartbeats) in the dictionary
    return segmented_ecg_signals

segmented_beats = segment_ecg_signals(normalized_signals, f_annotations)

import matplotlib.pyplot as plt

# Get the signal for record
segmented_beat = segmented_beats['200']

# Calculate the number of samples for 5 seconds
fs = 360
num_samples = 4 * fs

# Get the first 5 seconds of the normalized signal
filtered_segment = {record_name: signal[:num_samples] for record_name, signal in normalized_signals.items()}

# Create a time vector
time = [i/fs for i in range(num_samples)]

# Plot the first 5 seconds of the normalized signal for record
plt.figure(figsize=(7, 3))
plt.plot(time, filtered_segment['200'])  # select the specific record here
plt.title('First 5 seconds of filtered signal from record')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Plot the first 5 beats of the segmented beats
for i in range(5):
    plt.figure(figsize=(7, 3))
    plt.plot(segmented_beat[i])
    plt.title(f'Beat {i+1} from segmented beats of record')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

annotations = {} # Create a dict to store annotations symbols, earlier we worked annotations object which contains info about the R peaks locations,
#now we only need the symbols for the classification task
for record_name in records:
    annotation = f_annotations[record_name]
    annotations[record_name] = annotation.symbol

# Check for compatibility error between beats and annotations
for record_name in records:
        if record_name in segmented_beats and record_name in annotations:
          assert len(segmented_beats[record_name]) == len(annotations[record_name]), f"Mismatch in record {record_name}"

# Filter for specific arrhythmia classes
def filter_beats(segmented_beats, annotations, wanted_symbols):
    filtered_beats = {}
    for record_name in segmented_beats.keys():
        if record_name in annotations:
            record_annotations = annotations[record_name]
            record_beats = segmented_beats[record_name]
            filtered_record_beats = [beat for beat, symbol in zip(record_beats, record_annotations) if symbol in wanted_symbols]
            filtered_beats[record_name] = filtered_record_beats
    return filtered_beats

def filter_annotations(segmented_beats, annotations, wanted_symbols):
    filtered_annotations = {}
    for record_name in segmented_beats.keys():
        if record_name in annotations:
            record_annotations = annotations[record_name]
            record_beats = segmented_beats[record_name]
            filtered_record_annotations = [symbol for beat, symbol in zip(record_beats, record_annotations) if symbol in wanted_symbols]
            filtered_annotations[record_name] = filtered_record_annotations
    return filtered_annotations

wanted_symbols = ['N', 'L', 'R', 'A', 'V', '/']

up_segmented_beats = filter_beats(segmented_beats, annotations, wanted_symbols) # Filter heartbeats based on wanted_symbols
up_annotations = filter_annotations(segmented_beats, annotations, wanted_symbols) # Filter annotations based on wanted_symbols

# Flatten data (from a dict to a list of lists)
X = [up_segmented_beats[key] for key in sorted(up_segmented_beats.keys())]
y = [up_annotations[key] for key in sorted(up_annotations.keys())]

# Initialize two empty lists to hold the separated leads
X_lead1 = []
X_lead2 = []

for heartbeat in X:
    # Initialize two empty lists to hold the separated beats for this heartbeat
    heartbeat_lead1 = []
    heartbeat_lead2 = []

    for beat in heartbeat:
        # Separate the two leads and add them to their respective lists
        lead1, lead2 = beat[:, 0], beat[:, 1]
        heartbeat_lead1.append(lead1)
        heartbeat_lead2.append(lead2)

    # Add the separated beats for this heartbeat to X_lead1 and X_lead2
    X_lead1.append(heartbeat_lead1)
    X_lead2.append(heartbeat_lead2)

import pandas as pd

# Flatten the lists of heartbeats
X_lead1_flattened = [beat for heartbeat in X_lead1 for beat in heartbeat]
X_lead2_flattened = [beat for heartbeat in X_lead2 for beat in heartbeat]

# Flatten the list of annotations
y_flattened = [annotation for annotations in y for annotation in annotations]

# Create a DataFrame
df = pd.DataFrame({
    'lead1': X_lead1_flattened,
    'lead2': X_lead2_flattened,
    'annotation': y_flattened
})


import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Convert the lists of arrays into lists of lists
X_lead1_lists = df['lead1'].to_list()
X_lead2_lists = df['lead2'].to_list()

# Pad the lists of lists
X_lead1_padded = pad_sequences(X_lead1_lists, dtype='float32', padding='post')
X_lead2_padded = pad_sequences(X_lead2_lists, dtype='float32', padding='post')

# Convert the padded lists of lists into 3D numpy arrays
X_lead1 = np.array(X_lead1_padded)
X_lead2 = np.array(X_lead2_padded)


# Stack lead1 and lead2 along the last dimension
X = np.stack((X_lead1, X_lead2), axis=-1)

import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Convert the lists of arrays into lists of lists
X_lead1_lists = df['lead1'].to_list()
X_lead2_lists = df['lead2'].to_list()

# Pad the lists of lists
X_lead1_padded = pad_sequences(X_lead1_lists, dtype='float32', padding='post')
X_lead2_padded = pad_sequences(X_lead2_lists, dtype='float32', padding='post')

# Convert the padded lists of lists into 3D numpy arrays
X_lead1 = np.array(X_lead1_padded)
X_lead2 = np.array(X_lead2_padded)


# Stack lead1 and lead2 along the last dimension
X = np.stack((X_lead1, X_lead2), axis=-1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from keras.layers import LeakyReLU

model = Sequential([
    Conv1D(256, kernel_size=3, activation=LeakyReLU(alpha=0.01), input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),

    Bidirectional(GRU(128, return_sequences=True)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation=LeakyReLU(alpha=0.01)),
    Dense(y_train.shape[1], activation='softmax')
    ])

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(X_train, y_train, epochs=20, batch_size=1000, validation_split=0.1)

model.evaluate(X_test, y_test)


y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

y_true = y_test
y_true = np.argmax(y_true, axis=1)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.colors import LinearSegmentedColormap

# Convert encoded labels back to original values
y_true_labels = encoder.inverse_transform(y_true)
y_pred_labels = encoder.inverse_transform(y_pred)

# Calculate confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=encoder.classes_)

# Convert confusion matrix to DataFrame for easier plotting
cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)

# Create a custom colormap
cmap = LinearSegmentedColormap.from_list('custom', ['white', 'green'], 256)

# Plot confusion matrix with the custom colormap
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='g', cmap=cmap)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Generate the classification report
cr = classification_report(y_true_labels, y_pred_labels)

# Print the classification report
print("Classification Report:")
print(cr)


