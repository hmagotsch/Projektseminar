""" 
Messages are clustered using a text analysis
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# The input should be a .csv file that has all relevant messages in the column "MsgValueDE"
df = pd.read_csv("MachineA_with_Job.csv", sep=';')
df['MsgValueDE'].fillna('', inplace=True)

# Vectorization of messages with TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['MsgValueDE'])

# Number of clusters (classes)
# The more clusters, the more likely it is that all similar messages will be clustered together. However, a high number of clusters also leads to many clusters that only consist of one message, for example.
num_clusters = 50  

# KMeans-Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

print(df.groupby('Cluster').size())
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.5)
plt.title('PCA-Visualisierung der Cluster')
plt.show()

# Listing of the unique message values and message ranks per cluster
for cluster_num in range(num_clusters):
    cluster_messages_values = df[df['Cluster'] == cluster_num]['MsgValueDE'].unique()
    cluster_messages_ranks = df[df['Cluster'] == cluster_num]['MsgRank'].unique()

    print(f"Cluster {cluster_num}:\n")

    print("Eindeutige Message Values:")
    for message_value in cluster_messages_values:
        print(f"- {message_value}")

    print("\nEindeutige Message Ranks:")
    for message_rank in cluster_messages_ranks:
        print(f"- {message_rank}")

    print("\n")