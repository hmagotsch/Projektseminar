"""
Based on one message, the next message is predicted using a multonimial naive bias approach.
This is our best performing way of predicting the next message.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

"""The input file consists of only one column 'MsgValueDE', which was previously sorted by time of occurrence"""
data = pd.read_csv('Data.csv', sep=';')

# Check the frequency of the message values in the 'MsgValueDE' column
print(data['MsgValueDE'].value_counts())
# Create a column 'NextMsgValueDE' with the next value for 'MsgValueDE'
data['NextMsgValueDE'] = data['MsgValueDE'].shift(-1)
data = data.dropna() # Remove the last line as it has no next value

# Train test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
#Create a model-pipeline with CountVectorizer and MNB
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(train_data['MsgValueDE'], train_data['NextMsgValueDE'])

#Make predictions for the test data
predictions = model.predict(test_data['MsgValueDE'])

# Evaluate the model
precision = precision_score(test_data['NextMsgValueDE'], predictions, average='weighted', zero_division=1)
recall = recall_score(test_data['NextMsgValueDE'], predictions, average='weighted', zero_division=1)
f1 = f1_score(test_data['NextMsgValueDE'], predictions, average='weighted', zero_division=1)
accuracy = accuracy_score(test_data['NextMsgValueDE'], predictions)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1: {f1:.4f}')

# Example of a prediction
sample_message = "StÃ¶rung LÃ¼ftermotor M7 - M12 im Blasrahmen"
vocab_probabilities = model.predict_proba([sample_message])[0]
unique_messages = [message for message in model.classes_ if message != sample_message]
highest_prob_message = max(unique_messages, key=lambda message: vocab_probabilities[model.classes_.tolist().index(message)])

print(f'Input message: {sample_message}')
print(f'Next predicted message: {highest_prob_message}')