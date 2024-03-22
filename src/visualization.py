import seaborn as sns
import matplotlib.pyplot as plt

def plot_sample_signals(ecg_features, ecg_labels):
    """
    Plots one sample signal from each class in the dataset.
    
    Parameters:
        ecg_features (DataFrame): The features of the ECG dataset, where each row is a signal.
        ecg_labels (Series): The labels for the dataset, indicating the class of each signal.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 4))

    unique_classes = ecg_labels.unique()
    for class_ in unique_classes:
        sample_index = ecg_labels[ecg_labels == class_].index[0]
        if class_ == 1:
            plt.plot(ecg_features.loc[sample_index, :], label=f"Class {class_}", color="r")
        elif class_ == 2:
            plt.plot(ecg_features.loc[sample_index, :], label=f"Class {class_}", color="g")
        elif class_ == 3:
            plt.plot(ecg_features.loc[sample_index, :], label=f"Class {class_}", color="b")
        elif class_ == 4:
            plt.plot(ecg_features.loc[sample_index, :], label=f"Class {class_}", color="k")

    plt.title("Sample ECG Signal from Each Class")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend(title="ECG Classes")
    plt.show()
