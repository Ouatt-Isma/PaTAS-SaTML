import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder
from primaryNN import NeuralNetwork  # Import your library

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Select only two classes: 0 ('airplane') and 1 ('automobile')
selected_classes = [0, 1]
train_mask = np.isin(y_train, selected_classes)
test_mask = np.isin(y_test, selected_classes)

X_train = X_train[train_mask.flatten()]
y_train = y_train[train_mask.flatten()]
X_test = X_test[test_mask.flatten()]
y_test = y_test[test_mask.flatten()]

# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten the images
X_train = X_train.reshape(-1, 32 * 32 * 3)
X_test = X_test.reshape(-1, 32 * 32 * 3)

# One-hot encode the labels
try:
    encoder = OneHotEncoder(sparse=False)
except:
    encoder = OneHotEncoder(sparse_output=False)  
y_train_one_hot = encoder.fit_transform(y_train)
y_test_one_hot = encoder.transform(y_test)

# Initialize and train the neural network
input_size = 32 * 32 * 3  # Flattened CIFAR-10 input size
hidden_size = 64  # Number of neurons in the hidden layer
output_size = 2  # Number of classes

# Create the model
model = NeuralNetwork(input_size, hidden_size, output_size, ptas=False)

# Train the model
model.train(X_train, y_train_one_hot, epochs=10, batch_size=64, learning_rate=0.1)

# Evaluate the model on test data
y_pred = model.predict(X_test)
test_accuracy = np.mean(np.argmax(y_test_one_hot, axis=1) == y_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")
