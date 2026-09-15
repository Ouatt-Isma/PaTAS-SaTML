import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

# Load MNIST dataset using tensorflow.keras.datasets.mnist
def load_mnist():
    # Load the dataset directly from Keras
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize the pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Flatten the images to vectors (28x28 = 784)
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)
    
    # One-hot encode the labels
    try:
        encoder = OneHotEncoder(sparse=False)
    except:
        encoder = OneHotEncoder(sparse_output=False)  
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))
    
    return X_train, X_test, y_train_one_hot, y_test_one_hot

# Create neural network components from scratch
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def softmax(self, x):
        """Softmax function to output probabilities"""
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        """Cross-entropy loss"""
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def forward(self, X):
        """Forward pass"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y_true, learning_rate=0.001):
        """Backward pass to compute gradients"""
        m = X.shape[0]

        # Output layer gradients
        dz2 = self.a2 - y_true
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # Derivative of ReLU
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X_train, y_train, epochs=10, batch_size=64, learning_rate=0.001):
        """Train the model using stochastic gradient descent"""
        for epoch in range(epochs):
            # Shuffle data
            permutation = np.random.permutation(X_train.shape[0])
            X_train = X_train[permutation]
            y_train = y_train[permutation]
            
            # Train in batches
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                # Forward pass
                y_pred = self.forward(X_batch)
                # Backward pass
                self.backward(X_batch, y_batch, learning_rate)
            
            # Compute loss after each epoch
            y_pred = self.forward(X_train)
            loss = self.cross_entropy_loss(y_train, y_pred)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

# Load dataset
X_train, X_test, y_train_one_hot, y_test_one_hot = load_mnist()

# Define neural network parameters
input_size = 28 * 28  # Image size (28x28 pixels flattened)
hidden_size = 128  # Number of neurons in hidden layer
output_size = 10  # Number of output classes (digits 0-9)

# Create neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the model
nn.train(X_train, y_train_one_hot, epochs=10, batch_size=32, learning_rate=0.001)

# Test the model
predictions = nn.predict(X_test)
accuracy = np.mean(predictions == np.argmax(y_test_one_hot, axis=1))
print(f"Test Accuracy: {accuracy * 100:.2f}%")
