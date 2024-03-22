from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense

def create_cnn_model(input_shape, num_classes):
    """
    Creates a CNN model for ECG signal classification.

    Parameters:
        input_shape (tuple): The shape of the input data, excluding the batch size.
                             For example, (140, 1) for 140 time steps with a single feature per step.
        num_classes (int): The number of unique classes in the target labels.

    Returns:
        model: A compiled TensorFlow Sequential model.
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Summary of the CNN model
    model.summary()

    return model
