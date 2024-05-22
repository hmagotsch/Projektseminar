**Daten-Aufbereitung**
JobPartitioning.py
	Contains the job division of the data based on the Koenig & Bauer phase division. 


**Prozessvisualisierung**
create_xes_file.py
	Creates .xes file from dataset, that can later be used for process visualization.



**Clustering**
MessagesTextClustering.py
	Messages are clustered using a text analysis.



**Predictions**

MsgValueDE:
	MostFrequentMessage.py
		Benchmark model for message predictions.

	PredictMessage_MNB1.py
		Based on one message, the next message is predicted using a multonimial naive bias approach.

	PredictMessage_MNB2.py
		Based on a sequence of messages, the next message is predicted using a multonimial naive bias approach.
		The sequence lengths can be adjusted as needed.

	PredictMessage_MarkovChain.py
		Based on a sequence of messages, the next message is predicted using a markov chain approach.
		The sequence lengths can be adjusted as needed.

Production Stop:
	PredictProductionStop_LSTM.py
		A LSTM model is trained so it can predict whether a message with MsgRank 104 (production downtime) will occur in the next minute based on a sequence of past messages with a certain length.
	
	PredictProductionStop_mostFrequent.py
		Benchmark model for the LSTM model in PredictProductionStop_LSTM.py.

	PredictProductionStop_RandomForest.py
		A Random Forest model is trained so it can predict whether a message with MsgRank 104 (production downtime) will occur in the next minute. Model was neglected due to ill fitting.
