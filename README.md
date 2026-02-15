# MNIST-FROM-SCRATCH

![Author](https://img.shields.io/badge/Author-Adil%20CHOUKAIRE-blue)
![Project](https://img.shields.io/badge/Project-Computer%20Vision-blueviolet)
![Project](https://img.shields.io/badge/Project-Deep%20Learning-blueviolet)
![Status](https://img.shields.io/badge/Status-Completed-success)

![Python](https://img.shields.io/badge/Python-3.13%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-Core%20Maths-013243)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458)

Deep Learning solution designed to understand the fundamental mathematics of Artificial Intelligence by building a Neural Network entirely from scratch.

This project implements a Multi-Layer Perceptron (MLP) using raw NumPy matrix operations, bypassing high-level frameworks like TensorFlow or PyTorch to perform handwritten digit classification.

---

## 📁 Project Structure

```bash
NN_FROM_SCRATCH/
├── 📁 data/
│   ├── train.csv   # MNIST Training Data (Labels + Pixels)
│   └── test.csv    # MNIST Test Data (Pixels only)
├── 📁 venv/        # Virtual Environment
├── main.py
├── requirements.txt
├── submission.csv
└── README.md
```

⚠️ **Dataset Source:**  
The `train.csv` and `test.csv` files are derived from the famous MNIST dataset (via the Kaggle Digit Recognizer competition). Ensure these files are located in the `data/` directory before running the script.

---

## 🎯 Objective

The goal is to correctly classify grayscale images of handwritten digits (0–9) based on pixel intensity.

- **Input:** 784 pixels (28x28 flattened image)  
- **Output:** A probability distribution across 10 classes (0–9)  
- **Metric:** Categorical Accuracy  
- **Challenge:** The project prohibits the use of automatic differentiation engines. All gradients, derivatives (ReLU, Softmax), and backpropagation algorithms are computed and implemented manually.

---

## 🚀 Installation & Usage

### Clone the repository

```bash
git clone https://github.com/your-username/nn-from-scratch.git
cd nn-from-scratch
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
python main.py
```

---

## 🛠️ Script Description

### `main.py`

This is the core script containing the entire pipeline. It implements a Vanilla Neural Network architecture designed for mathematical transparency.

---

### 1️⃣ Data Preparation & Engineering

- **Normalization:** Pixel values are scaled from `[0, 255]` to `[0, 1]` to prevent gradient explosion.  
- **Transposition:** Data is reshaped to `(784, m)` to facilitate vectorized matrix operations.  
- **One-Hot Encoding:** Converts categorical labels (`Y`) into binary vectors for the Loss calculation.

---

### 2️⃣ Model Architecture (The MLP)

- **Input Layer:** 784 Neurons  
- **Hidden Layer:** 128 Neurons using ReLU activation (rectifies non-linearity)  
- **Output Layer:** 10 Neurons using Softmax activation (converts logits to probabilities)  
- **Initialization:** Weights are initialized using a scaled Random Normal distribution (`randn * 0.01`) to break symmetry  

---

### 3️⃣ The Mathematical Engine

#### Forward Propagation

- Computes linear combinations:  
  ```
  Z = W · X + b
  ```
- Applies activation functions  
- Includes a **Numerically Stable Softmax** to handle large exponentials  

#### Backward Propagation

Manually calculates gradients using the Chain Rule:

- Computes error at the output (`∂Z₂`)  
- Backpropagates error to the hidden layer (`∂Z₁`) using the derivative of ReLU  

#### Optimization

- Updates parameters (`W`, `b`) using standard **Gradient Descent**  
- Fixed learning rate  

---

### 4️⃣ Output & Submission

The script generates a final output file named:

```
submission.csv
```

This CSV contains:

- `ImageId`
- Predicted `Label` for the unseen test set  

The file follows the submission format required by the Kaggle platform.

---

## ⚠️ Notes

### 📊 Performance

- Training accuracy: **~95.8%**
- Achieved after **3000 iterations**

### ⚙️ Resource Usage

- Optimized for **CPU execution**
- Uses vectorized NumPy operations (SIMD)
- Completes training in a few minutes

### 🔧 Hyperparameters

- Learning rate: **0.1**
- Hidden units: **128**
- Configured for optimal convergence on this specific dataset