import numpy as np
import copy
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression

def calculate_cosine_distances(random_perturbations, num_slices):
    """
    Calculates the cosine distances between each perturbation vector and the original signal representation.

    Parameters:
    - random_perturbations (np.ndarray): An array of perturbation vectors.
    - num_slices (int): The total number of segments the ECG signal is divided into, matching the dimension of the perturbations.

    Returns:
    - np.ndarray: An array of cosine distances for each perturbation from the original signal representation.
    """
    # Represent the original ECG signal as a perturbation where all segments are enabled (i.e., a vector of ones)
    original_ecg_rep = np.ones((1, num_slices))

    # Calculate cosine distances
    cosine_distances = pairwise_distances(random_perturbations, original_ecg_rep, metric='cosine').ravel()

    return cosine_distances


def analyze_prediction(probability_vector, class_labels):
    """
    Analyzes the probability vector from a model prediction, returning the top predicted classes
    and the most likely predicted class.
    
    Parameters:
        probability_vector (np.ndarray): The probability vector for a given instance, as predicted by the model.
        class_labels (list): A list of class labels, adjusted to be zero-based.
    
    Returns:
        tuple: A tuple containing a list of the top predicted classes and the most likely predicted class.
    """
    # Sort the classes based on the probability vector and select the top N classes
    top_pred_classes = probability_vector[0].argsort()[-len(class_labels):][::-1]
    
    # Use np.argmax to find the index of the maximum value in the probability vector
    predicted_class_index = np.argmax(probability_vector, axis=1)
    
    # Map the predicted index to its corresponding class label
    predicted_classes = [class_labels[i] for i in predicted_class_index]
    
    # Since we're predicting for one instance, access the first element for the predicted class
    predicted_class = predicted_classes[0]
    
    return top_pred_classes, predicted_class


def segment_ecg_signal(instance_ecg, num_slices=40):
    """
    Segments an ECG signal into a fixed number of slices.

    Parameters:
        instance_ecg (np.ndarray): The ECG signal instance to segment.
        num_slices (int): The number of slices to divide the signal into.

    Returns:
        int: The width of each slice in the segmented ECG signal.
    """
    total_length = len(instance_ecg)
    slice_width = total_length // num_slices
    return slice_width

def perturb_total_mean(signal, start_idx, end_idx):
    """
    Perturbs a segment of the signal by replacing it with the overall mean of the signal.
    
    Parameters:
        signal (np.ndarray): The original signal to perturb.
        start_idx (int): The starting index of the segment to perturb.
        end_idx (int): The ending index of the segment to perturb.
        
    Returns:
        np.ndarray: The signal with the specified segment perturbed by the total mean.
    """
    modified_signal = signal.copy()
    modified_signal[start_idx:end_idx] = modified_signal.mean()
    return modified_signal

def perturb_mean(signal, start_idx, end_idx):
    """
    Directly modifies a segment of the signal by replacing it with the mean of that segment.
    
    Parameters:
        signal (np.ndarray): The signal to perturb, modified in place.
        start_idx (int): The starting index of the segment to perturb.
        end_idx (int): The ending index of the segment to perturb.
    """
    mean_value = np.mean(signal[start_idx:end_idx])
    signal[start_idx:end_idx] = mean_value

def perturb_noise(signal, start_idx, end_idx):
    """
    Perturbs a segment of the signal by replacing it with random noise within the signal's range.
    
    Parameters:
        signal (np.ndarray): The original signal to perturb.
        start_idx (int): The starting index of the segment to perturb.
        end_idx (int): The ending index of the segment to perturb.
        
    Returns:
        np.ndarray: The signal with the specified segment perturbed by random noise.
    """
    modified_signal = signal.copy()
    modified_signal[start_idx:end_idx] = np.random.uniform(modified_signal.min(), modified_signal.max(), end_idx - start_idx)
    return modified_signal

def generate_random_perturbations(num_perturbations, num_slices):
    """
    Generates random perturbations for ECG signal segments.
    
    This function creates a binary matrix where each row represents a perturbation,
    and each column corresponds to a segment of the ECG signal. A value of '1' indicates
    the segment is active or unchanged, while '0' indicates the segment is inactive or altered.
    
    Parameters:
        num_perturbations (int): The number of perturbations to generate.
        num_slices (int): The number of slices (segments) each ECG signal is divided into.
        
    Returns:
        np.ndarray: A binary matrix representing random perturbations.
    """
    random_perturbations = np.random.binomial(1, 0.5, size=(num_perturbations, num_slices))
    return random_perturbations

def apply_perturbation_to_ecg(signal, perturbation, num_segments, perturb_function=perturb_mean):
    """
    Apply a perturbation to an ECG signal.

    Parameters:
    - signal (np.ndarray): The original ECG signal.
    - perturbation (np.ndarray): A vector indicating which segments to turn on (1) or off (0).
    - num_segments (int): The total number of segments the ECG signal is divided into.
    - perturb_function (function): The function to use for perturbing the signal (default is perturb_mean).

    Returns:
    - np.ndarray: A perturbed version of the ECG signal.
    """
    # Copy the signal to avoid modifying the original
    perturbed_signal = copy.deepcopy(signal)
    segment_length = len(signal) // num_segments

    # Apply the perturbation based on the provided vector
    for i, active in enumerate(perturbation):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        # Apply perturbation function only to "off" segments
        if not active:
            perturb_function(perturbed_signal, start_idx, end_idx)

    return perturbed_signal

def predict_perturbations(model, instance_ecg, random_perturbations, num_slices, perturb_function):
    """
    Applies a set of perturbations to an ECG signal, predicts the class probabilities for each perturbed signal,
    and collects the predictions.

    Parameters:
    - model: The trained ECG classifier model.
    - instance_ecg (np.ndarray): The original ECG signal instance.
    - random_perturbations (np.ndarray): An array of perturbation vectors.
    - num_slices (int): The total number of segments the ECG signal is divided into.
    - perturb_function (function): The function to use for perturbing the signal (e.g., perturb_mean).

    Returns:
    - np.ndarray: An array of model predictions for each perturbed ECG signal.
    """
    perturbation_predictions = []

    for perturbation in random_perturbations:
        # Apply the current perturbation to the ECG signal
        perturbed_signal = apply_perturbation_to_ecg(instance_ecg, perturbation, num_slices, perturb_function)

        # Reshape as required by the model
        perturbed_signal_reshaped = perturbed_signal.reshape(1, len(perturbed_signal), 1)  

        # Predict the class probabilities
        model_prediction = model.predict(perturbed_signal_reshaped)
        perturbation_predictions.append(model_prediction)

    # Convert the list of predictions into a numpy array
    perturbation_predictions = np.array(perturbation_predictions)
    return perturbation_predictions

def calculate_cosine_distances(random_perturbations, num_slices):
    """
    Calculates the cosine distances between each perturbation vector and the original signal representation.

    Parameters:
    - random_perturbations (np.ndarray): An array of perturbation vectors.
    - num_slices (int): The total number of segments the ECG signal is divided into, matching the dimension of the perturbations.

    Returns:
    - np.ndarray: An array of cosine distances for each perturbation from the original signal representation.
    """
    # Represent the original ECG signal as a perturbation where all segments are enabled (i.e., a vector of ones)
    original_ecg_rep = np.ones((1, num_slices))

    # Calculate cosine distances
    cosine_distances = pairwise_distances(random_perturbations, original_ecg_rep, metric='cosine').ravel()

    return cosine_distances

def calculate_weights_from_distances(cosine_distances, kernel_width=0.25):
    """
    Applies a kernel function to cosine distances to calculate weights for each perturbation.

    Parameters:
    - cosine_distances (np.ndarray): An array of cosine distances for each perturbation from the original signal representation.
    - kernel_width (float): The kernel width parameter for the exponential kernel function. Default is 0.25.

    Returns:
    - np.ndarray: An array of weights for each perturbation, derived from the cosine distances.
    """
    weights = np.sqrt(np.exp(-(cosine_distances ** 2) / kernel_width ** 2))
    return weights

def fit_explainable_model(perturbation_predictions, random_perturbations, weights, target_class):
    """
    Fits a linear regression model to quantify the importance of each segment 
    in the decision-making process for the target class.

    Parameters:
    - perturbation_predictions (np.ndarray): The array of model predictions for each perturbed ECG signal.
    - random_perturbations (np.ndarray): The matrix of perturbation vectors.
    - weights (np.ndarray): The array of weights corresponding to each perturbation.
    - target_class (int): The index of the target class to explain.

    Returns:
    - np.ndarray: The coefficients of the linear regression model, indicating the importance of each segment.
    """
    # Initialize the linear regression model
    explainable_model = LinearRegression()

    # Squeeze the middle dimension out from perturbation_predictions to get a 2D array
    perturbation_predictions_squeezed = np.squeeze(perturbation_predictions, axis=1)

    # Select the predictions for the target class across all perturbations
    target_predictions = perturbation_predictions_squeezed[:, target_class]

    # Fit the model
    explainable_model.fit(X=random_perturbations, y=target_predictions, sample_weight=weights)

    # Extract the coefficients
    segment_importance_coefficients = explainable_model.coef_

    return segment_importance_coefficients


def identify_top_influential_segments(segment_importance_coefficients, number_of_top_features=5):
    """
    Identifies the top influential segments of an ECG signal based on the importance coefficients 
    obtained from a linear regression model.

    Parameters:
    - segment_importance_coefficients (np.ndarray): The coefficients indicating the importance of each segment.
    - number_of_top_features (int): The number of top influential segments to identify.

    Returns:
    - np.ndarray: Indices of the top influential segments.
    """
    # Sort the coefficients based on their absolute magnitude to identify the most influential segments
    top_influential_segments = np.argsort(np.abs(segment_importance_coefficients))[-number_of_top_features:]
    
    return top_influential_segments
