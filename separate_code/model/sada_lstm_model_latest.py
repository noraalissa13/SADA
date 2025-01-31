# -*- coding: utf-8 -*-
"""SADA_LSTM_model_latest.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1B-U0rae54DCgyXj8S0vXc3nhaEw33Kke
"""

## HAYANI NAZURAH HASRAM 2117628
## NORA ALISSA BINTI ISMAIL 2117862

##This code is the model training part of the SADA project. It involves loading the preprocessed data
##and implementing the RNN LSTM algorithm on it. The result will then be evaluated using confusion matrix
##and visualised with a line chart.

from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Input

file_path = '/content/drive/MyDrive/SADA/k.csv'

# Load the EEG data
df = pd.read_csv(file_path)

# Define features and target
features = ['timestamp', 'alpha0', 'beta0', 'gamma0', 'delta0', 'theta0',
            'alpha1', 'beta1', 'gamma1', 'delta1', 'theta1', 'alpha2',
            'beta2', 'gamma2', 'delta2', 'theta2', 'alpha3', 'beta3',
            'gamma3', 'delta3', 'theta3']
target = 'Label'

X = df[features]
y = df[target]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data into sequences (10 timesteps per sequence)
timesteps = 10
n_features = X_scaled.shape[1]

# Create sequences
X_seq = []
y_seq = []
for i in range(len(X_scaled) - timesteps):
    X_seq.append(X_scaled[i:i + timesteps])
    y_seq.append(y[i + timesteps])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# One-hot encode the target
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
y_seq_encoded = encoder.fit_transform(y_seq.reshape(-1, 1))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq_encoded, test_size=0.2, random_state=42)

# Build the RNN LSTM model
model = Sequential()
model.add(Input(shape=(timesteps, n_features)))  # Define the input shape
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.2)) #add dropout layer to prevent overfitting
model.add(LSTM(32, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model using adam optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Make predictions on the test data
y_pred = model.predict(X_test)

# Convert the one-hot encoded predictions and true values back to labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true_labels, y_pred_labels)
print(f"Test Accuracy: {accuracy:.4f}")

# Calculate precision, recall, and F1-score
precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.categories_[0], yticklabels=encoder.categories_[0])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=encoder.categories_[0]))

import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Test the data on other eeg files

# Load new data
new_data_path = '/content/drive/My Drive/SADA/test.csv'
new_df = pd.read_csv(new_data_path)

# Preprocess the new data
X_new = new_df[features]
X_new = scaler.transform(X_new)  # Apply the same scaling as training data

# Reshape for LSTM (samples, time_steps, features)
X_new_reshaped = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))

# Get predicted labels from the model
predicted_labels = model.predict(X_new_reshaped)
predicted_labels = encoder.inverse_transform(predicted_labels)
predicted_labels = predicted_labels.ravel()

# Create a DataFrame with test data features and predicted labels
results_df = pd.DataFrame(X_new, columns=features)
results_df['Predicted_Label'] = predicted_labels # add predicted label for comparison


# IF the csv file has the 'Label' column for ground truth:
if 'Label' in new_df.columns:
    results_df['Actual_Label'] = new_df['Label'] # get the actual labels from new_df

    # Save the DataFrame to a CSV file
    results_file_path = '/content/drive/My Drive/SADA/prediction_concentration_rnn2.0.csv'
    results_df.to_csv(results_file_path, index=False)

    print(f"Prediction results saved to: {results_file_path}")

else:
    print("Warning: 'test.csv' does not contain an 'Actual_Label' column for comparison.")

    # else if the csv file doesnt have the label column:
    results_file_path = '/content/drive/My Drive/SADA/prediction_concentration_rnn2.0.csv'
    results_df.to_csv(results_file_path, index=False)

    print(f"Prediction results (without ground truth) saved to: {results_file_path}")

##Create the identified attention level line chart

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#read the data
results_file_path = '/content/drive/My Drive/SADA/prediction_concentration_rnn2.0.csv'
results_df = pd.read_csv(results_file_path)

# Extract timestamp and predicted labels
timestamps = results_df['timestamp']
predicted_labels = results_df['Predicted_Label']

# Create a mapping for attention levels to numerical values for plotting
attention_mapping = {'Low': 0.5, 'Moderate': 2, 'High': 3.5}
predicted_attention_values = [attention_mapping[label] for label in predicted_labels]

# Create the line graph
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(timestamps, predicted_attention_values, label='Identified Attention Level', color='blue')

# Set y-axis limits for color bands
ax.set_ylim(0, 4)

# Add color bands for attention levels
ax.axhspan(0, 1.33, facecolor='red', alpha=0.3, label='Low')
ax.axhspan(1.33, 2.66, facecolor='yellow', alpha=0.3, label='Moderate')
ax.axhspan(2.66, 4, facecolor='green', alpha=0.3, label='High')

# Customize the plot
ax.set_xlabel('Timestamp')
ax.set_ylabel('Attention Level')
ax.set_title('Identified Attention Level Over Time')
ax.legend()
plt.grid(True)

# Set y-axis ticks and labels
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['Low', 'Moderate', 'High'])

plt.show()
