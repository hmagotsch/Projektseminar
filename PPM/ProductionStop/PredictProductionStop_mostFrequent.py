***
	Creates a benchmark model for the LSTM approach in PredictProductionStop_LSTM.py using the train and test sets that are created there.
	It predicts either always positiv or alwys negative for a production stop, depending what is more frequent.
	It uses the y_train and y_test sets of the model from PredictProductionStop_LSTM.py
***

from sklearn.dummy import DummyClassifier

def create_most_frequent_model(y_train, y_test):
    # Bestimmen, welche Klasse häufiger ist (0 oder 1)
    most_frequent_class = np.argmax(np.bincount(y_train))

    # Wenn die Klasse 0 häufiger ist, dann Vorhersage immer 0, ansonsten immer 1
    if most_frequent_class == 0:
        strategy = "most_frequent"
    else:
        strategy = "constant"

    # Erstellen des Dummy-Classifiers
    model = DummyClassifier(strategy=strategy, constant=most_frequent_class)
    model.fit(X_train_sequences, y_train)

    # Ausgabe des erstellten Modells
    return model

# Benchmark-Modell erstellen
benchmark_model = create_most_frequent_model(y_train_sequences, y_test_sequences)