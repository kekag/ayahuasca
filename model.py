import os
import librosa
import pretty_midi
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Activation
from sklearn.model_selection import train_test_split


def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def load_midi(file_path):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    drum_notes = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            for note in instrument.notes:
                drum_notes.append([note.start, note.end, note.pitch, note.velocity])
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


def load_data(directory):
    matched_pairs = match_files(directory)
    X = []
    y = []
    for audio_path, midi_path in matched_pairs:
        audio_features = load_audio(audio_path)
        midi_notes = load_midi(midi_path)
        X.append(audio_features)
        y.append(midi_notes)

    # Further processing to align audio features with MIDI notes will be needed
    # This is a placeholder example
    X = np.array(X)
    y = np.array(y)

    return X, y


# Example directory
directory = "path_to_your_directory"

# Load data
X, y = load_data(directory)

# Prepare data for LSTM (this is a simplified example, further preprocessing is needed)
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
model.add(TimeDistributed(Dense(y_train.shape[2])))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Save the model
model.save("drum_generation_model.h5")
