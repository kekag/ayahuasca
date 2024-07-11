import os
import librosa
import pretty_midi
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Activation
from sklearn.model_selection import train_test_split

# Constants
SR = 22050  # Sampling rate
N_MELS = 128  # Number of Mel bands
HOP_LENGTH = 512  # Hop length for feature extraction
NUM_DRUM_CLASSES = 128  # MIDI has 128 possible notes


def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def load_midi(file_path):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    drum_notes = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            for note in instrument.notes:
                start_time = int(note.start * SR / HOP_LENGTH)
                end_time = int(note.end * SR / HOP_LENGTH)
                pitch = note.pitch
                drum_notes.append((start_time, end_time, pitch))
    return drum_notes


def match_files(directory):
    audio_files = [f for f in os.listdir(directory) if f.endswith(".wav")]
    midi_files = [f for f in os.listdir(directory) if f.endswith(".mid")]
    matched_pairs = []
    for audio_file in audio_files:
        base_name = os.path.splitext(audio_file)[0]
        midi_file = base_name + ".mid"
        if midi_file in midi_files:
            matched_pairs.append(
                (
                    os.path.join(directory, audio_file),
                    os.path.join(directory, midi_file),
                )
            )
    return matched_pairs


def create_target_sequence(drum_notes, num_frames):
    target_sequence = np.zeros((num_frames, NUM_DRUM_CLASSES), dtype=np.float32)
    for start_time, end_time, pitch in drum_notes:
        target_sequence[start_time:end_time, pitch] = 1.0
    return target_sequence


def load_data(directory):
    matched_pairs = match_files(directory)
    X = []
    y = []
    for audio_path, midi_path in matched_pairs:
        audio_features = load_audio(audio_path)
        drum_notes = load_midi(midi_path)
        num_frames = audio_features.shape[1]
        target_sequence = create_target_sequence(drum_notes, num_frames)
        X.append(
            audio_features.T
        )  # Transpose to have time steps as the first dimension
        y.append(target_sequence)
    X = np.array(X)
    y = np.array(y)
    return X, y


# Example directory
directory = "training"

# Load data
X, y = load_data(directory)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build LSTM model
model = Sequential()
model.add(
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)
)
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributed(Dense(NUM_DRUM_CLASSES)))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Save the model
model.save("drum_generation_model.h5")
