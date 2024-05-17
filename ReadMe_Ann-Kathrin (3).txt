**Clustering**
clustering_jobs.py
        clusters the jobs per machine, using k-means (and DBSCAN as well as hierarchical Clustering)




clustering_machines.py
        clusters the machines, using k-means




cluster_machines_preprocessing.py
        preprocessing to clustering_von_maschinen.py: extract relevant information fromm over 200 machines and save relevant data in a single data frame








**Predictions**
        **MsgRank**
benchmark_msgrank_b.py
        calculates the benchmark




xgboost_&_rf_msgrank_.py
creates predictions with random forest and xgboost




smote_&_rf_msgrank_.py
        samples the data with smote and creates predictions with random forest
        
smote_&_xgboostmsgrank_.py
        samples the data with smote and creates predictions with xgboost




adasyn_&_xgboostmsgrank_.py
        samples the data with adasyn and creates predictions with xgboost




adasyn_&_rf_msgrank_.py
        samples the data with adasyn and creates predictions with random forest




fine_tuning_xgboost.py
        fine tuning of the xgboost model by use of different features




without_locids_fine_tuning_xgboost.py
        fine tuning of the xgboost model without using LocIDs