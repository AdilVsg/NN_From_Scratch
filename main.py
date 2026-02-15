import numpy as np
import pandas as pd
import random as rd

# Data loading
data_train = pd.read_csv(r"C:\Users\adilc\Projet_perso\NN_From_Scratch\data\train.csv")
data_test = pd.read_csv(r"C:\Users\adilc\Projet_perso\NN_From_Scratch\data\test.csv")


# Build Y_train (labels) and X_train (pixel values for each image) as NumPy arrays
Y_train = data_train['label'].to_numpy()
X_train = data_train.drop('label', axis = 1).to_numpy() / 255  # Normalization

# Build X_test
X_test = data_test.to_numpy() / 255  # Normalization

# Number of pixels in each image
nb_pixels = X_train.shape[1]

# Number of images
nb_images = X_train.shape[0]

# Multi-Layer Perceptron
def init(nb_hidden : int, nb_pixel : int) -> tuple :
    """
        Initialize the weights and biases

        Returns:
            The weights and biases
        """
    # Weights and biases initialization
    W1 = np.random.randn(nb_hidden, nb_pixel) * 0.01
    W2 = np.random.randn(10, nb_hidden) * 0.01  # 10 is the number of outputs (0 to 9)
    b1 = np.zeros((nb_hidden , 1))
    b2 = np.zeros((10 , 1))  # 10 is the number of outputs (0 to 9)
    
    return (W1, W2, b1, b2)

def forward(X : np.array, W1 : np.array, W2 : np.array, b1 : np.array, b2 : np.array) -> tuple : 
    """
        Perform the forward pass of the neural network

        Returns:
            The outputs of the first and last layers
        """
    Z1 = np.dot(W1 , np.transpose(X)) + b1
    A1 = np.maximum(0 , Z1)  # ReLU(Z1): first activation function
    Z2 = np.dot(W2 , A1) + b2

    # Numerical stability
    exp_Z2 = np.exp(Z2 - np.max(Z2, axis=0, keepdims=True))  # Max becomes 0 (shifted version of Z2)
    A2 = exp_Z2 / np.sum(exp_Z2, axis=0, keepdims = True)  # Softmax(exp_Z2): second activation function -> gives a probability distribution
    
    return (Z1, A1, Z2, A2)

def one_hot(Y : np.array) -> np.array :
    """
        Convert Y_train into a one-hot encoded array

        Returns:
            The transposed one-hot encoded array of Y_train
        """
    Y_one_hot = np.zeros((len(Y), 10))  # Initialization of Y_one_hot
    for i in range(len(Y)):
        Y_one_hot[i][Y[i]] = 1  # Add a 1 at the position corresponding to the value in Y. Example: if y = 2, we get [0, 0, 1, 0, ...]
        
    return np.transpose(Y_one_hot)  # Return the transpose of the array to match A2 in backward propagation

def backward(X : np.array, A2 : np.array, Y : np.array, A1 : np.array, W2 : np.array, Z1 : np.array, nb_images : int) -> tuple :
    """
        Perform backward propagation to minimize the Loss (Log-loss)

        Returns:
            Tuple of gradient values used to minimize the Loss
        """
    d_Z2 = A2 - Y
    d_W2 = 1/nb_images * (np.dot(d_Z2, np.transpose(A1)))
    d_b2 = 1/nb_images * (np.sum(d_Z2, axis = 1, keepdims = True))
    d_Z1 = np.dot(np.transpose(W2), d_Z2) * (Z1 > 0)
    d_W1 = 1/nb_images * (np.dot(d_Z1, X))
    d_b1 = 1/nb_images * (np.sum(d_Z1, axis = 1, keepdims = True))
    
    return (d_Z2, d_W2, d_b2, d_Z1, d_W1, d_b1)

def update(W1 : np.array, b1 : np.array, W2 : np.array, b2 : np.array, d_W1 : np.array, d_b1 : np.array, d_W2 : np.array, d_b2 : np.array, alpha : float) -> tuple:
    """
        Update the weights and biases

        Returns:
            Tuple of updated weights and biases
        """
    W1 = W1 - alpha * d_W1
    b1 = b1 - alpha * d_b1
    W2 = W2 - alpha * d_W2
    b2 = b2 - alpha * d_b2
    
    return (W1 , b1, W2, b2)


def get_predictions(A2 : np.array) -> int:
    """
        Get the predictions from the output of the last layer

        Returns:
            The predicted class (number with the highest probability)
        """
    return np.argmax(A2, axis = 0)  # Return the number with the highest probability

def get_accuracy(predictions, Y : np.array) -> float:
    """
        Compute the accuracy

        Returns:
            The accuracy of the results by comparing predictions with true labels
        """
    return np.sum(predictions == Y) / Y.size  # Return the mean of correct predictions

nb_hidden = 128  # Number of neurons in the hidden layer
alpha = 0.1  # Learning rate

def gradient_descent(iterations : int):
    """
        Perform gradient descent training

        Returns:
            The trained weights and biases
        """
    W1, W2, b1, b2 = init(nb_hidden, nb_pixels)
    Y = one_hot(Y_train)
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward(X_train, W1, W2, b1, b2)
        d_Z2, d_W2, d_b2, d_Z1, d_W1, d_b1 = backward(X_train, A2, Y, A1, W2, Z1, nb_images)
        W1 , b1, W2, b2 = update(W1, b1, W2, b2, d_W1, d_b1, d_W2, d_b2, alpha)
        if i % 100 == 0:
            print("Iterations : ", i)
            print("Accuracy : ", get_accuracy(get_predictions(A2), Y_train))
    
    return W1, W2, b1, b2

iterations = 3000  # Number of iterations

W1, W2, b1, b2 = gradient_descent(iterations)

# FINAL STEP: PREDICTIONS ON TEST.CSV
print("\n--- Starting final test on test.csv ---")

# Forward propagation on test data
_, _, _, A2_test = forward(X_test, W1, W2, b1, b2)

# Convert probabilities to class labels (0–9)
predictions_test = get_predictions(A2_test)

# Console preview
print(f"Predictions generated for {len(predictions_test)} images.")
print(f"First 20 predictions: {predictions_test[:20]}")

# Save to CSV (Kaggle format: ImageId, Label)
submission = pd.DataFrame({
    'ImageId': range(1, len(predictions_test) + 1),
    'Label': predictions_test
})

submission.to_csv('submission.csv', index=False)
print("File 'submission.csv' successfully generated.")
