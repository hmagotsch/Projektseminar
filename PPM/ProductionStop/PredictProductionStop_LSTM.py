***
A LSTM model is trained so it can predict whether a message with MsgRank 104 (production downtime) will occur in the next minute based on a sequence of past messages with a certain length.
This model turned out to be more suitable that the Random Forest model, because it recognizes temporal connections in the data better. 
It is noticeable that the model performs better when short-term prediction periods or input sequences are used. This indicates that the data under consideration is more likely to be related in the short term and less in the long term. This approach also is better than a most-frequent approach.

Through random sampling, the sequence length of 10 as input and the prediction period of 1 min could be identified as the best performing parameters for model so far (trained and tested on data of Machine C).
***


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Generation of sequences
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i+sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

# Creation of the target variables for each sequence
def create_target_variable_for_sequences(df, sequences, sequence_length, prediction_window_minutes):
    targets = []
    for i in range(len(sequences)):
        sequence_end_time = df['CheckIn'].iloc[i + sequence_length - 1]
        prediction_end_time = sequence_end_time + pd.Timedelta(minutes=prediction_window_minutes)

        # Check whether a message with rank 104 occurs in the next 10 minutes after the sequence
        target_value = int(any((df['CheckIn'] > sequence_end_time) & (df['CheckIn'] <= prediction_end_time) & (df['MsgRank'] == 104)))
        targets.append(target_value)
    return np.array(targets)

# Import and prepare data
df = pd.read_csv('MachineC_with_Job_marked.csv', sep=';')
df['CheckIn'] = pd.to_datetime(df['CheckIn'], errors='coerce', utc=True)
df['CheckOut'] = pd.to_datetime(df['CheckOut'], errors='coerce', utc=True)
df['MsgRank'].fillna(-1, inplace=True)
df['Target'] = 0

# Features and target variables
features = ['LocID1', 'LocID2', 'LocID3', 'MsgRank', 'Speed', 'Gross', 'Net']
X = df[features].values
y = df['Target'].values

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the datetime object for the time from which the split is to take place (train/test split)
split_time = pd.Timestamp('2023-08-13 00:00:22.218000+00:00')

# Indices for training and test data based on the timestamp
train_indices = df[df['CheckIn'] < split_time].index
test_indices = df[df['CheckIn'] >= split_time].index

# Training and test data
X_train = X_scaled[train_indices]
y_train = y[train_indices]
X_test = X_scaled[test_indices]
y_test = y[test_indices]

# Define the length of the input sequences and the size of the prediction period
sequence_length = 10
prediction_window_minutes = 1

# Convert training and test data into sequences of the respective length
X_train_sequences = create_sequences(X_train, sequence_length)
X_test_sequences = create_sequences(X_test, sequence_length)

# Create target variables for training and test data
y_train_sequences = create_target_variable_for_sequences(df, X_train_sequences, sequence_length, prediction_window_minutes)
y_test_sequences = create_target_variable_for_sequences(df[df['CheckIn'] >= split_time], X_test_sequences, sequence_length, prediction_window_minutes)

# Define the LSTM model
model = Sequential([
    LSTM(units=50, input_shape=(X_train_sequences.shape[1], X_train_sequences.shape[2])),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_sequences, y_train_sequences, epochs=5, batch_size=32)

# Make predictions for the test set
y_pred = (model.predict(X_test_sequences) > 0.5).astype("int32")

# Calculation and display of metrics
accuracy = accuracy_score(y_test_sequences, y_pred)
precision = precision_score(y_test_sequences, y_pred)
recall = recall_score(y_test_sequences, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


# Calculation and display of the confusion matrix
conf_matrix = confusion_matrix(y_test_sequences, y_pred)

print("Konfusionsmatrix:")
print(f"True Positive (TP): {conf_matrix[1, 1]}")
print(f"True Negative (TN): {conf_matrix[0, 0]}")
print(f"False Positive (FP): {conf_matrix[0, 1]}")
print(f"False Negative (FN): {conf_matrix[1, 0]}")


# Due to a update in pandas, the miliseconds in a Datetime Object can not be used anymore (update from 2.2.1 to 2.2.2).
#
# To avoid problems, you now have to change the original definition of split_time to split_time = pd.Timestamp('2023-08-13 00:00:22+00:00')
# and replace the following two lines
# df['CheckIn'] = pd.to_datetime(df['CheckIn'], errors='coerce', utc=True)
# df['CheckOut'] = pd.to_datetime(df['CheckOut'], errors='coerce', utc=True)
# with
# date_format = "%Y-%m-%d %H:%M:%S%z"
# df['CheckIn'] = df['CheckIn'].str.replace('\.\d+', '', regex=True)
# df['CheckOut'] = df['CheckOut'].str.replace('\.\d+', '', regex=True)
# df['CheckIn'] = pd.to_datetime(df['CheckIn'], format=date_format)
# df['CheckOut'] = pd.to_datetime(df['CheckOut'], format=date_format)
#
# Note, that the performances of the models may vary slightly due to these adjustments.
