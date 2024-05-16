"""
Based on a sequence of messages, the next message is predicted using a Markov Chain approach.
The performance of this model is not satisfactory due to the class imbalance in the data.
The MNB approach is prefered.
"""

from collections import defaultdict
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to get all n+1 element sections of a sequence
def get_all_n_plus_1_item_slices_of(sequence, n):
    return [sequence[i:i+n+1] for i in range(len(sequence)-n)]

# Function for creating the mapping
def build_map(event_chain, n):
    event_map = defaultdict(list)
    for events in get_all_n_plus_1_item_slices_of(event_chain, n):
	# Use the first n elements as key
        slice_key = tuple(events[:-1])
	# The last element is the event to be predicted
        last_event = events[-1]
	# Add the event to be predicted to the key return event_map
        event_map[slice_key].append(last_event)
    return event_map

# Function for predicting the next event
def predict_next_event(whats_happened_so_far, event_map, n):
    # Create the key from the last n events
    slice_key = tuple(whats_happened_so_far[-n:])
    # Receive possible next events for the key
    possible_next_events = event_map.get(slice_key, [])
    if possible_next_events:
	# If there are possible next events, select the most frequent event
        return max(set(possible_next_events), key=possible_next_events.count)
    else:
        return "Ende"

"""The input file consists of only one column 'MsgValueDE', which was previously sorted by time of occurrence"""
csv_file_path = "Data.csv"
df = pd.read_csv(csv_file_path, sep=';')
df.sort_values(by=['FileDate', 'SeqNr'], inplace=True)

msg_column = "MsgValueDE"
event_chain = df[msg_column].tolist()

# Set the size of the sequence (n)
n = 12

# train test split
train_data, test_data = train_test_split(event_chain, test_size=0.2, random_state=42)
event_map = build_map(train_data, n)

true_next_events = []
predicted_next_events = []

for sequence in test_data:
    # Iterate over the sequence and make predictions for each event
    for i in range(len(sequence) - n):
        current_sequence = sequence[i:i + n]
        true_next_event = sequence[i + n]
        predicted_next_event = predict_next_event(current_sequence, event_map, n)

        true_next_events.append(true_next_event)
        predicted_next_events.append(predicted_next_event)

# Evaluate the model
accuracy = accuracy_score(true_next_events, predicted_next_events)
precision = precision_score(true_next_events, predicted_next_events, average='macro', zero_division=1)
recall = recall_score(true_next_events, predicted_next_events, average='macro', zero_division=1)
f1 = f1_score(true_next_events, predicted_next_events, average='macro', zero_division=1)

print("n:", n)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)