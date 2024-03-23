# LIME for Time-Series Explanation in ECG Classification

<div align="center">
  <img src="poster.webp" width="500">
</div>


LIME for TimeSeries is an open-source project dedicated to advancing the interpretability of machine learning models focused on time series data. Utilizing the Local Interpretable Model-agnostic Explanations (LIME) technique, this project aims to demystify the decision-making processes of complex models. By integrating perturbation-based explanations, it provides insights into model predictions, enhancing transparency and trust in AI applications across various domains. Ideal for researchers, data scientists, and anyone invested in explainable AI (XAI), this repository offers tools, documentation, and examples to facilitate the understanding and application of LIME in time series analysis. By employing LIME, we aim to uncover which segments of an electrocardiogram (ECG) signal most influence the model's classification decisions, enhancing the interpretability of time-series models in healthcare.

## Overview

Time-series classification, particularly in the context of ECG signal analysis, plays a crucial role in diagnosing cardiovascular diseases. While deep learning models offer promising results, their "black-box" nature hinders clinical adoption due to the lack of interpretability. This project leverages LIME, a technique for explaining predictions of any classifier in an interpretable and faithful manner, by perturbing the input signal and observing the changes in predictions.

## Features
- **Data Preprocessing:** Techniques for transforming raw ECG signals into a format suitable for model training.
- **Model Training:** A Convolutional Neural Network (CNN) approach for ECG signal classification.
- **LIME Explanations:** Implementation of LIME to identify influential signal segments contributing to each classification decision.
- **Visualization Tools:** Utilities for visualizing ECG signals, their perturbations, and the influence of different segments on the model's predictions.

## Installation
Clone the repository to your local machine:

git clone https://github.com/mdhabibi/LIME-for-Time-Series.git
cd LIME-for-Time-Series

## Usage
- **Data Preparation:** Start by preparing the ECG dataset. The **data_preprocess.py** module provides functions for loading and preprocessing the data.

- **Model Training:** Use the **model_training.py** module to train a CNN on the prepared ECG data. The module outlines the model architecture and training procedure.

- **Applying LIME:** The **lime_explanation.py** module contains the implementation of LIME for time-series data. It includes functions for generating perturbations, applying perturbations to the signal, and fitting an interpretable model to the perturbed data.

- **Visualization:** The **visualization.py** module offers visualization utilities to plot the original and perturbed ECG signals, class distribution, and the impact of each segment on the model's predictions.

A detailed example of using these modules can be found in the **main.ipynb** within the notebooks directory.
