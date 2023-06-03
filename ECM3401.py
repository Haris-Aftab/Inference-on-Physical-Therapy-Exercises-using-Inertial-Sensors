import numpy as np
import pandas as pd
import os
from scipy.signal import welch
from itertools import product
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from multiprocessing import Pool
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def get_data_from_file(file):
    """
    Reads data from a CSV file.

    Args:
        file (str): Path to the CSV file.

    Returns:
        numpy.ndarray: A 2D numpy array containing the data from the file.
    """
    with open(file, newline='') as csvfile:
        data = np.loadtxt(csvfile, delimiter=',', usecols=(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9), skiprows=1)
    return data


def segment_data(data, window_size, window_shift):
    """
    Segment data into windows of a given size and shift.

    Args:
        data (numpy.ndarray): The data to segment.
        window_size (int): The size of each window.
        window_shift (int): The number of data points to shift the window by.

    Returns:
        numpy.ndarray: The segmented data, with shape (n_windows, window_size, n_features).
    """
    n_windows = int((data.shape[0] - window_size) / window_shift)
    segmented_data = np.zeros((n_windows, window_size, data.shape[1]))
    for i in range(n_windows):
        segmented_data[i] = data[i * window_shift:i *
                                 window_shift + window_size]
    return segmented_data


def extract_features(windows):
    """
    Extracts time and frequency features from each window of data and returns a list of all the features.

    Args:
        windows (np.ndarray): A 3D numpy array containing the segmented data.

    Returns:
        list: A list of extracted features.
    """
    features = []
    for window in windows:
        features.append(time_features(window))
        features.append(freq_features(window))
    features = [item for sublist in features for item in sublist]
    return features


def time_features(window):
    """
    Extract time-domain features from a given window of sensor data.

    Args:
        window (np.ndarray): A 2D numpy array of shape (n_samples, n_features), representing a window of sensor data.

    Returns:
        list: A list of time-domain features extracted from the input window.
    """
    features = []
    for i in range(1, window.shape[1]):
        features.append(np.mean(window[:, i]))
        features.append(np.std(window[:, i]))
        features.append(np.max(window[:, i]))
        features.append(np.min(window[:, i]))
        features.append(np.median(window[:, i]))
        features.append(pd.Series(window[:, i]).skew())
        features.append(pd.Series(window[:, i]).kurt())
        features.append(pd.Series(window[:, i]).mode()[0])

        for j in range(i + 1, window.shape[1]):
            features.append(np.corrcoef(window[:, i], window[:, j])[0, 1])

    return features


def freq_features(window):
    """
    Extract frequency-domain features for a given window of sensor data.

    Args:
        window (np.ndarray): 2D numpy array of shape (n_samples, n_features), representing a window of sensor data.

    Returns:
        list: A list of frequency-domain features extracted from the input window.
    """
    features = []
    for i in range(1, window.shape[1]):
        f, Pxx = welch(window[:, i], 100, nperseg=window.shape[0])
        features.append(np.mean(Pxx))
        features.append(np.std(Pxx))
        features.append(np.max(Pxx))
        features.append(np.min(Pxx))
        features.append(np.median(Pxx))
        features.append(pd.Series(Pxx).skew())
        features.append(pd.Series(Pxx).kurt())

    return features


def my_concatenate(x, y):
    """
    Concatenates two integers x and y by multiplying x by 10 and adding y.

    Args:
        x (int): The first integer.
        y (int): The second integer.

    Returns:
        int: The concatenated integer.
    """
    return x * 10 + y


def process_file(file_path, window_size, window_shift):
    """
    Extract features from a file given the window size and window shift.

    Args:
        file_path (str): The path to the file.
        window_size (int): The size of the sliding window.
        window_shift (int): The shift between consecutive windows.

    Returns:
        list: A list of features extracted from the given file.
    """
    return extract_features(segment_data(get_data_from_file(file_path), window_size, window_shift))


def get_all_features(window_size, window_shift):
    """
    Returns all the features and labels for the inertial data.

    Args:
        window_size (int): The size of the sliding window used to extract features.
        window_shift (int): The amount by which the sliding window is shifted after each feature extraction.

    Returns:
        Tuple: A tuple containing two lists. The first list contains the features extracted from the inertial data, and the second list contains the labels for these features.
    """
    base = 'inertial/'
    body_half = ['upper/', 'lower/']
    age_group = ['A/', 'B/', 'C/', 'D/', 'E/']
    upper_sensor_pos = ['Lforearm/', 'Rforearm/']
    lower_sensor_pos = ['Lshin/', 'Rshin/']
    upper_exercise = ['EFE', 'EAH', 'SQZ']
    lower_exercise = ['KFEL', 'SQT', 'HAAL', 'KFER', 'HAAR']

    input_files = [os.path.join(base, body_half[0], age, sensor_pos, age[0] + '0' + str(i) + exercise + str(j) + '_' + str(k) + '.csv')
                   for age, sensor_pos, i, exercise, j, k in product(age_group, upper_sensor_pos, range(1, 6), upper_exercise, range(2), range(1, 3))
                   if os.path.isfile(os.path.join(base, body_half[0], age, sensor_pos, age[0] + '0' + str(i) + exercise + str(j) + '_' + str(k) + '.csv'))]

    with Pool() as p:
        upper_features = p.starmap(process_file, [(
            file_path, window_size, window_shift) for file_path in input_files])

    upper_labels = [my_concatenate(upper_exercise.index(exercise) + 1, j)
                    for age, sensor_pos, i, exercise, j, k in product(age_group, upper_sensor_pos, range(1, 6), upper_exercise, range(2), range(1, 3))
                    if os.path.isfile(os.path.join(base, body_half[0], age, sensor_pos, age[0] + '0' + str(i) + exercise + str(j) + '_' + str(k) + '.csv'))]

    input_files = [os.path.join(base, body_half[1], age, sensor_pos, age[0] + '0' + str(x) + exercise + str(y) + '_' + str(z) + '.csv')
                   for age, sensor_pos, x, exercise, y, z in product(age_group, lower_sensor_pos, range(1, 6), lower_exercise, range(2), range(1, 3))
                   if os.path.isfile(os.path.join(base, body_half[1], age, sensor_pos, age[0] + '0' + str(x) + exercise + str(y) + '_' + str(z) + '.csv')) and
                   ((sensor_pos[0] == exercise[-1]) or exercise == 'SQT')]

    with Pool() as p:
        lower_features = p.starmap(process_file, [(
            file_path, window_size, window_shift) for file_path in input_files])

    lower_labels = [my_concatenate(lower_exercise.index(exercise) + 4 - 3*(lower_exercise.index(exercise) == 3) - 2*(lower_exercise.index(exercise) == 4), y)
                    for age, sensor_pos, x, exercise, y, z in product(age_group, lower_sensor_pos, range(1, 6), lower_exercise, range(2), range(1, 3))
                    if os.path.isfile(os.path.join(base, body_half[1], age, sensor_pos, age[0] + '0' + str(x) + exercise + str(y) + '_' + str(z) + '.csv')) and
                    ((sensor_pos[0] == exercise[-1]) or exercise == 'SQT')]

    return upper_features + lower_features, upper_labels + lower_labels


def preprocess_data(features, labels):
    """
    Preprocesses the data by padding the input features with zeros, removing NaN values,
    scaling the features with StandardScaler, and returning the preprocessed features and labels.

    Args:
        features (list): A list of 1D numpy arrays, where each array contains the features of a single sample.
        labels (list): A list of labels corresponding to the features.

    Returns:
        Tuple: A tuple containing the preprocessed features as a 2D numpy array and the corresponding labels as a 1D numpy array.

    """
    # Determine the maximum length of the arrays
    max_len = max(len(arr) for arr in features)

    # Pad the arrays with zeros
    padded_feats = []
    for arr in features:
        padded_arr = np.pad(arr, (0, max_len - len(arr)), mode='constant')
        padded_feats.append(padded_arr)

    # Stack the padded arrays into a single 2D array
    features_array = np.vstack(padded_feats)
    labels = np.array(labels)

    # Remove NaN values from features_array and their corresponding labels
    mask = ~np.isnan(features_array).any(axis=1)

    features_array = features_array[mask]
    labels = labels[mask]

    # Padding to make all elements in features_array of equal length
    max_len = max(len(x)
                  for x in features_array[~np.isnan(features_array).any(axis=1)])

    pad_width = ((0, 0), (0, max_len - len(features_array[0])))
    pad_width = np.array(pad_width).reshape(2, -1)
    padded_feat = np.pad(features_array, pad_width, mode='constant')

    # Initialize a StandardScaler object
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(padded_feat)

    return scaled_features, labels


def svm_model(features, labels):
    '''
    Perform Support Vector Machine (SVM) classification using Leave-One-Out Cross-Validation (LOOCV).

    Args:
        features (np.ndarray): 2D array of shape (n_samples, n_features) containing the preprocessed features.
        labels (np.ndarray): 1D array of shape (n_samples,) containing the class labels.

    Returns:
        None: Prints the evaluation metrics for the SVM model on the given data, including accuracy, precision, sensitivity, F1-score, and specificity.
    '''
    # Define multiprocessing pool
    with Pool() as pool:
        # Apply LOOCV in parallel and get predictions and true labels
        predictions = pool.starmap(svm_loocv_predict, [(
            i, features, labels) for i in range(len(features))])
        predictions = np.concatenate(predictions)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    sensitivity = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    tn_fp_fn_tp = confusion_matrix(labels, predictions).ravel()
    tn = tn_fp_fn_tp[0]
    fp = tn_fp_fn_tp[1]
    specificity = tn / (tn + fp)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Sensitivity:", sensitivity)
    print("F1-score:", f1)
    print("Specificity:", specificity)


def svm_loocv_predict(i, features, labels):
    '''
    Implements the prediction step for a single iteration of LOOCV for a SVM model.

    Args:
        i (int): the index of the feature vector to be used for testing
        features (np.ndarray): a 2D array containing the feature vectors
        labels (np.ndarray): a 1D array containing the corresponding labels

    Returns:
        np.ndarray: a 1D array containing the predicted labels for the test feature vector
    '''
    # Create SVM object
    svm = SVC(kernel='linear')

    # Get training features and labels
    train_features = np.delete(features, i, axis=0)
    train_labels = np.delete(labels, i)

    # Fit SVM model
    svm.fit(train_features, train_labels)

    # Predict test feature
    return svm.predict(features[i].reshape(1, -1))


def rf_model(features, labels):
    '''
    Perform Random Forest (RF) classification using Leave-One-Out Cross-Validation (LOOCV).

    Args:
        features (np.ndarray): 2D array of shape (n_samples, n_features) containing the preprocessed features.
        labels (np.ndarray): 1D array of shape (n_samples,) containing the class labels.

    Returns:
        None: Prints the evaluation metrics for the SVM model on the given data, including accuracy, precision, sensitivity, F1-score, and specificity.
    '''
    loo = LeaveOneOut()

    # Define multiprocessing pool
    with Pool() as pool:
        # Apply LOOCV in parallel and get predictions and true labels
        results = pool.starmap(rf_loocv_predict, [(
            train_index, test_index, features, labels) for train_index, test_index in loo.split(features)])

    # Unpack results
    predictions, true_labels = zip(*results)

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    sensitivity = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    tn_fp_fn_tp = confusion_matrix(labels, predictions).ravel()
    tn = tn_fp_fn_tp[0]
    fp = tn_fp_fn_tp[1]
    specificity = tn / (tn + fp)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Sensitivity:", sensitivity)
    print("F1-score:", f1)
    print("Specificity:", specificity)


def rf_loocv_predict(train_index, test_index, features, labels):
    '''
    Implements the prediction step for a single iteration of LOOCV for a RF model.

    Parameters:
        train_index (np.ndarray): The indices of the training samples.
        test_index (np.ndarray): The index of the test sample.
        features (np.ndarray): The feature data as a 2D numpy array.
        labels (np.ndarray): The corresponding label data as a 1D numpy array.

    Returns:
        int: The predicted label of the test sample.
        int: The true label of the test sample.
    '''
    # Define random forest classifier
    rfc = RandomForestClassifier()

    # Get training and test features and labels
    train_features, test_features = features[train_index], features[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

    # Fit random forest classifier
    rfc.fit(train_features, train_labels)

    # Predict test feature
    prediction = rfc.predict(test_features)

    # Return prediction and true label
    return prediction[0], test_labels[0]


def dt_model(features, labels):
    '''
    Perform Decision Tree (DT) classification using Leave-One-Out Cross-Validation (LOOCV).

    Args:
        features (np.ndarray): 2D array of shape (n_samples, n_features) containing the preprocessed features.
        labels (np.ndarray): 1D array of shape (n_samples,) containing the class labels.

    Returns:
        None: Prints the evaluation metrics for the SVM model on the given data, including accuracy, precision, sensitivity, F1-score, and specificity.
    '''
    # Define multiprocessing pool
    with Pool() as pool:
        # Apply LOOCV in parallel and get predictions and true labels
        predictions = pool.starmap(
            dt_loocv_predict, [(i, features, labels) for i in range(len(features))])
        predictions = np.concatenate(predictions)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    sensitivity = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    tn_fp_fn_tp = confusion_matrix(labels, predictions).ravel()
    tn = tn_fp_fn_tp[0]
    fp = tn_fp_fn_tp[1]
    specificity = tn / (tn + fp)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Sensitivity:", sensitivity)
    print("F1-score:", f1)
    print("Specificity:", specificity)


def dt_loocv_predict(i, features, labels):
    '''
    Implements the prediction step for a single iteration of LOOCV for a DT model.

    Args:
        i (int): the index of the feature vector to be used for testing
        features (np.ndarray): a 2D array containing the feature vectors
        labels (np.ndarray): a 1D array containing the corresponding labels

    Returns:
        np.ndarray: a 1D array containing the predicted labels for the test feature vector
    '''
    # Create SVM object
    svm = DecisionTreeClassifier()

    # Get training features and labels
    train_features = np.delete(features, i, axis=0)
    train_labels = np.delete(labels, i)

    # Fit SVM model
    svm.fit(train_features, train_labels)

    # Predict test feature
    return svm.predict(features[i].reshape(1, -1))


if __name__ == "__main__":
    """
    Runs experiments to evaluate the performance of different machine learning (ML) models on different window sizes and window shifts of a dataset.

    The script prints the results of the experiments, including the accuracy, precision, sensitivity, F1-score, and specificity of the ML model on each window size and window shift. 
    """

    window_size = 100

    print("-----------------Window shift: 25%-----------------")
    window_shift = int(window_size * 0.25)

    print("Window size: 5s")
    features, labels = preprocess_data(get_all_features(5*window_size, 5*window_shift))
    svm_model(features, labels)

    print("Window size: 4s")
    features, labels = preprocess_data(get_all_features(4*window_size, 4*window_shift))
    svm_model(features, labels)

    print("Window size: 3s")
    features, labels = preprocess_data(get_all_features(3*window_size, 3*window_shift))
    svm_model(features, labels)

    print("Window size: 2s")
    features, labels = preprocess_data(get_all_features(2*window_size, 2*window_shift))  
    svm_model(features, labels)

    print("Window size: 1s")
    features, labels = preprocess_data(get_all_features(window_size, window_shift))    
    svm_model(features, labels)

    print("-----------------Window shift: 50%-----------------")
    window_shift = int(window_size * 0.5)

    print("Window size: 5s")
    features, labels = preprocess_data(get_all_features(5*window_size, 5*window_shift))   
    svm_model(features, labels)

    print("Window size: 4s")
    features, labels = preprocess_data(get_all_features(4*window_size, 4*window_shift))   
    svm_model(features, labels)

    print("Window size: 3s")
    features, labels = preprocess_data(get_all_features(3*window_size, 3*window_shift))  
    svm_model(features, labels)

    print("Window size: 2s")
    features, labels = preprocess_data(get_all_features(2*window_size, 2*window_shift))  
    svm_model(features, labels)

    print("Window size: 1s")
    features, labels = preprocess_data(get_all_features(window_size, window_shift))   
    svm_model(features, labels)

    print("-----------------Window shift: 75%-----------------")
    window_shift = int(window_size * 0.75)

    print("Window size: 5s")
    features, labels = preprocess_data(get_all_features(5*window_size, 5*window_shift))  
    svm_model(features, labels)

    print("Window size: 4s")
    features, labels = preprocess_data(get_all_features(4*window_size, 4*window_shift))  
    svm_model(features, labels)

    print("Window size: 3s")
    features, labels = preprocess_data(get_all_features(3*window_size, 3*window_shift))
    svm_model(features, labels)

    print("Window size: 2s")
    features, labels = preprocess_data(get_all_features(2*window_size, 2*window_shift))
    svm_model(features, labels)

    print("Window size: 1s")
    features, labels = preprocess_data(get_all_features(window_size, window_shift))
    svm_model(features, labels)

    print("-----------------Window shift: 100%-----------------")
    window_shift = int(window_size * 1)

    print("Window size: 5s")
    features, labels = preprocess_data(get_all_features(5*window_size, 5*window_shift))
    svm_model(features, labels)

    print("Window size: 4s")
    features, labels = preprocess_data(get_all_features(4*window_size, 4*window_shift))
    svm_model(features, labels)

    print("Window size: 3s")
    features, labels = preprocess_data(get_all_features(3*window_size, 3*window_shift))
    svm_model(features, labels)

    print("Window size: 2s")
    features, labels = preprocess_data(get_all_features(2*window_size, 2*window_shift)) 
    svm_model(features, labels)

    print("Window size: 1s")
    features, labels = preprocess_data(get_all_features(window_size, window_shift))
    svm_model(features, labels)