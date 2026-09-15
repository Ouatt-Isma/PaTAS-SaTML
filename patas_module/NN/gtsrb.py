
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from primaryNN import NeuralNetwork
from datasets import load_gtsrb_from_kaggle, load_poisoned_all


patch = 4
X, y = load_gtsrb_from_kaggle(small=True)
X = X.reshape(-1, 32 * 32)

# Step 3: One-hot encode labels


# Step 4: Train/test split
# Split both one-hot encoded labels and raw labels
X_train, X_test, y_train, y_test= train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, y_train, n_pois = load_poisoned_all(X_train, y_train, patch_size=patch, img_size=32)
X_test_pois, y_test_pois, _ = load_poisoned_all(X_test, y_test, patch_size=patch, img_size=32)

try:
    encoder = OneHotEncoder(sparse=False)
except:
    encoder = OneHotEncoder(sparse_output=False)  
y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_oh = encoder.fit_transform(y_test.reshape(-1, 1))

# Step 5: Train your neural network
model = NeuralNetwork(input_size=32 * 32, hidden_size=64, output_size=43, ptas=True)
model.train(X_train, y_train_oh, epochs=50, batch_size=64, learning_rate=0.01)

# Step 6: Evaluate
print(f"Eval on poisoned test image")
y_pred = model.predict(X_test_pois)
accuracy = np.mean(y_pred == y_test_pois)
print(f"GTSRB Test Accuracy using poisoned label: {accuracy * 100:.2f}%")
accuracy = np.mean(y_pred == y_test)
print(f"GTSRB Test Accuracy using true label: {accuracy * 100:.2f}%")

print(f"Eval on true test image")
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test_pois)
print(f"GTSRB Test Accuracy using poisoned label: {accuracy * 100:.2f}%")
accuracy = np.mean(y_pred == y_test)
print(f"GTSRB Test Accuracy using true label: {accuracy * 100:.2f}%")
