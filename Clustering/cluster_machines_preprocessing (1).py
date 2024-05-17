# -*- coding: utf-8 -*-
"""Cluster_Machines_Preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12Ms3aR4Mc9uj2tj9qr7OdoCasjrWlbAw

The folowing code extracts all the relevant information from the csv files and stores them in one data frame which can be used in advance to build the clusters.
"""

import pandas as pd
import os

#loop through all the csv files, read each file into a DataFrame, and concatenate them into a singke DataFrame


folder_path = '/content/drive/MyDrive/Neue Daten Projekt/Maschinen_mit_Jobeinteilung_v3/'
all_data = pd.DataFrame()



#extract and save the relevant information

#Initialize an empty DataFrame to store the data
new_data = pd.DataFrame()




for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

       #calculate mean speed and total number of jobs for each machine
        mean_speed = df['Speed'].mean()
        total_jobs = df['Job'].max()
        machine = df['Machine'].iloc[0]
        net = df['Net'].max()
        gross = df['Gross'].max()

      #create a DataFrame with calculated values
        machine_data = pd.DataFrame({
       'Machine': [machine],  # Assuming filename represents the machine identifier
       'Mean_Speed': [mean_speed],
       'Total_Jobs': [total_jobs],
       'Net':[net],
       'Gross':[gross]
       #'NetproJob' :
      })

       #Append the machine_data to the aggregated_data
        new_data = pd.concat([new_data, machine_data], ignore_index=True)

new_data.head()

new_data.info()

# Save the aggregated data to a new CSV file
new_data.to_csv('/content/drive/MyDrive/new_data.csv', index=False)