***
A random forest model is trained so that it can predict whether a message with MsgRank 104 will occur in the next 10 minutes (production downtime) based on a 10-minute period. In the course of the project, the model was identified as unsuitable for our application and was therefore neglected. The LSTM is more suitable and delivers better results
***

import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import export_text

pd.set_option('display.max_rows', None)

# Step 1: Preprocess the Data
df = pd.read_csv('MachineB_with_Job.csv', sep=';')

# Convert date columns to datetime format
df['FileDate'] = pd.to_datetime(df['FileDate'], format='%d.%m.%Y')
df['CheckIn'] = pd.to_datetime(df['CheckIn'], errors='coerce', utc=True)
df['CheckOut'] = pd.to_datetime(df['CheckOut'], errors='coerce', utc=True)

# Fill NaT values with the original column values
df['CheckIn'].fillna(df['CheckIn'], inplace=True)
df['CheckOut'].fillna(df['CheckOut'], inplace=True)

# Create a target column for the next 10 minutes
df['Target'] = df['MsgRank'].shift(-1)
df['Target'] = df['Target'].eq(104) & (df['CheckIn'] + timedelta(minutes=10) <= df['CheckOut'])

# Step 2: Feature Engineering for 10-Minute Slots
df['TimeWindowStart'] = (df['CheckIn'] - df['CheckIn'].min()) // timedelta(minutes=10) * timedelta(minutes=10)

# Aggregate features for each 10-minute slot
agg_df = df.groupby(['TimeWindowStart', 'Machine']).agg({
    'CheckIn': 'first',  # Keep the first CheckIn timestamp in the slot
    'MsgRank': 'max',
    'Target': 'max'				#feature engineering to be improved
}).reset_index()

# Step 3: Model Training
features = ['MsgRank']

train, test = train_test_split(agg_df, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(train[features], train['Target'])

predictions = model.predict(test[features])

# Evaluate the model
accuracy = accuracy_score(test['Target'], predictions)
precision = precision_score(test['Target'], predictions)
recall = recall_score(test['Target'], predictions)

print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')

# Step 4: Prediction on New Data
new_data = pd.read_csv('MachineB_with_Job.csv', sep=';')
new_data['CheckIn'] = pd.to_datetime(new_data['CheckIn'], errors='coerce', utc=True)
new_data['CheckIn'].fillna(new_data['CheckIn'], inplace=True)

# Feature Engineering for 10-Minute Slots
new_data['TimeWindowStart'] = (new_data['CheckIn'] - new_data['CheckIn'].min()) // timedelta(minutes=10) * timedelta(minutes=10)

# Aggregate features for each 10-minute slot
new_agg_data = new_data.groupby(['TimeWindowStart', 'Machine']).agg({
    'CheckIn': 'first',  # Keep the first CheckIn timestamp in the slot
    'MsgRank': 'max'
}).reset_index()

# Make and display predictions on new data
new_predictions = model.predict(new_agg_data[features])
new_agg_data['Predicted_Target'] = new_predictions
print(new_agg_data[['TimeWindowStart', 'Machine', 'MsgRank', 'Predicted_Target']])

# Closer look on each decision tree and its constraints
trees = model.estimators_
for i, tree in enumerate(trees):
    tree_rules = export_text(tree, feature_names=features)
    print(f"Decision Tree {i+1}:\n{tree_rules}")
