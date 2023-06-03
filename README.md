Final year computer science project at the University of Exeter on inference on physical therapy exercises using inertail sensors.

How to run:
- When ECM3401.py file is initially run, it outputs performance metrics for each window shift and window size for SVM with linear kernel. 
- To change kernel for svm, in line 291, change 'linear' to 'rbf' or 'poly' for a Gaussian or polynomial kernel. 
- To change ML algorithm, change 'svm_model' in the main funciton for 'dt_model' or 'rf_model' for a decision tree or random forest model.
