"""
Based on a sequence of messages, the next message is predicted using a multonimial naive bias approach.
The performance in this model is worse than if you only consider individual data points, in other words no sequences. If you take sequences as here, the best performance is with a sequence length of 2.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""The input file consists of only one column 'MsgValueDE', which was previously sorted by time of occurrence"""
data = pd.read_csv('Data.csv', sep=';')
data['NextMsgValueDE'] = data['MsgValueDE'].shift(-1)
data = data.dropna()

#Train test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Function for creating sequences from messages with a defined length
def create_sequences(messages, sequence_length=2):
    sequences = []
    labels = []
    for i in range(len(messages) - sequence_length):
	# Extract a sequence of messages and the corresponding label
        sequence = messages.iloc[i:i + sequence_length].tolist()
        label = messages.iloc[i + sequence_length]
        sequences.append(' '.join(sequence))
        labels.append(label)
    return sequences, labels

# Create sequences and labels from training and test data
train_sequences, train_labels = create_sequences(train_data['MsgValueDE'])
test_sequences, test_labels = create_sequences(test_data['MsgValueDE'])

# Create a model pipeline with CountVectorizer and Multinomial Naive Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(train_sequences, train_labels)

# Make predictions for the test sequences
predictions = model.predict(test_sequences)

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions, average='weighted', zero_division=1)
recall = recall_score(test_labels, predictions, average='weighted', zero_division=1)
f1 = f1_score(test_labels, predictions, average='weighted', zero_division=1)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1: {f1:.4f}')

# Prediction example
example_sequence = ["Programmstart-Taster Anleger betÃ¤tigt", "Vorlauf Fortdruck"]
example_sequence_text = ' '.join(example_sequence)

predicted_next_message = model.predict([example_sequence_text])[0]

print(f'Input sequence: {example_sequence}')
print(f'Next predicted message: {predicted_next_message}')