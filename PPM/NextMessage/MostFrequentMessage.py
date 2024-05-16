"""
Benchmark model for message predictions
The prediction is always the most frequent message.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

csv_file_path = 'MachineD_with_Job.csv'
df = pd.read_csv(csv_file_path, sep=';')
messages = df['MsgValueDE']
messages = messages.dropna()

# Determining the most frequent message
most_frequent_message = messages.value_counts().idxmax()

# Generating predictions: Each prediction is the most frequent message for all entries
predictions = [most_frequent_message] * len(messages)

# Performance metrics
accuracy = accuracy_score(messages, predictions)
precision = precision_score(messages, predictions, average='weighted')
recall = recall_score(messages, predictions, average='weighted')
f1 = f1_score(messages, predictions, average='weighted')

print(f'Most frequent message: {most_frequent_message = messages.value_counts().idxmax()}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')