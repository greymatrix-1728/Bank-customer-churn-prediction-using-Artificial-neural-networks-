# Bank Customer Churn Prediction using Artificial Neural Network

A deep learning project that predicts whether a bank customer will leave (churn) based on their profile and account information. Built with TensorFlow/Keras, this binary classification model achieves strong accuracy on a real-world banking dataset.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [How It Works](#how-it-works)

---

## Problem Statement

Customer churn is one of the most critical challenges in the banking industry. Losing a customer is far more expensive than retaining one. This project builds an Artificial Neural Network (ANN) to predict — given a customer's demographics and account data — whether they are likely to exit the bank. This allows the bank to proactively intervene and retain at-risk customers.

---

## Dataset

**File:** `Churn_Modelling.csv`  
**Source:** [Kaggle — Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)  
**Size:** 10,000 rows × 14 columns

### Columns

| Column | Type | Description |
|---|---|---|
| `RowNumber` | int | Row index (dropped during training) |
| `CustomerId` | int | Unique customer ID (dropped during training) |
| `Surname` | str | Customer surname (dropped during training) |
| `CreditScore` | int | Customer credit score |
| `Geography` | str | Country: France, Spain, or Germany |
| `Gender` | str | Male or Female |
| `Age` | int | Age of the customer |
| `Tenure` | int | Number of years as a bank customer |
| `Balance` | float | Account balance in USD |
| `NumOfProducts` | int | Number of bank products the customer uses |
| `HasCrCard` | int | 1 = has credit card, 0 = does not |
| `IsActiveMember` | int | 1 = active member, 0 = inactive |
| `EstimatedSalary` | float | Estimated annual salary in USD |
| `Exited` | int | **Target variable** — 1 = churned, 0 = stayed |

### Class Distribution

| Label | Count | Percentage |
|---|---|---|
| Stayed (0) | 7,963 | 79.6% |
| Churned (1) | 2,037 | 20.4% |

> The dataset is moderately imbalanced — roughly 1 in 5 customers churned.

---

## Project Structure

```
bank-churn-prediction/
│
├── Churn_Modelling.csv              # Dataset
├── artificial_neural_network.py     # Main Python script
├── artificial_neural_network.ipynb  # Jupyter Notebook version
└── README.md
```

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `Python` | 3.8+ | Core language |
| `TensorFlow / Keras` | 2.x | Building and training the ANN |
| `NumPy` | latest | Numerical computations |
| `Pandas` | latest | Data loading and manipulation |
| `scikit-learn` | latest | Preprocessing, train/test split, evaluation |

---

## Model Architecture

The ANN is a fully connected feedforward network:

```
Input Layer     →  11 features (after encoding)
Hidden Layer 1  →  6 neurons, ReLU activation
Hidden Layer 2  →  6 neurons, ReLU activation
Output Layer    →  1 neuron, Sigmoid activation (binary output)
```

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Binary Crossentropy |
| Metric | Accuracy |
| Batch size | 32 |
| Epochs | 100 |
| Train/Test split | 80% / 20% |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bank-churn-prediction.git
cd bank-churn-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install tensorflow numpy pandas scikit-learn
```

Or using a requirements file:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
tensorflow>=2.0
numpy
pandas
scikit-learn
jupyter  # optional, for notebook
```

---

## Usage

### Run as a Python script

```bash
python artificial_neural_network.py
```

### Run as a Jupyter Notebook

```bash
jupyter notebook artificial_neural_network.ipynb
```

### Predict for a single customer

After training, you can predict for any new customer by passing their details into the model. Example — predicting for a customer with these details:

| Feature | Value |
|---|---|
| Geography | France |
| Credit Score | 600 |
| Gender | Male |
| Age | 40 |
| Tenure | 3 years |
| Balance | $60,000 |
| Number of Products | 2 |
| Has Credit Card | Yes |
| Is Active Member | Yes |
| Estimated Salary | $50,000 |

```python
# France is encoded as [1, 0, 0] via one-hot encoding
# Male is encoded as 1 via label encoding
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
```

> **Result:** The model predicts this customer will **stay** with the bank.

**Important notes:**
- The `predict` method expects a 2D array — always use double square brackets `[[ ]]`.
- Geography must be one-hot encoded as `[France=1,0,0]`, `[Germany=0,1,0]`, `[Spain=0,0,1]`.
- All features must be scaled using the same `StandardScaler` fitted on training data.

---

## How It Works

### Step 1 — Data Preprocessing

- **Drop irrelevant columns:** `RowNumber`, `CustomerId`, `Surname` are removed since they carry no predictive signal.
- **Label Encoding:** The `Gender` column (Male/Female) is encoded to (1/0).
- **One-Hot Encoding:** The `Geography` column (France/Spain/Germany) is one-hot encoded into 3 binary columns to avoid implying any ordinal relationship.
- **Train/Test Split:** 80% of the data is used for training, 20% for testing.
- **Feature Scaling:** `StandardScaler` is applied to normalize all features to the same scale, which is essential for neural networks to converge properly.

### Step 2 — Building the ANN

A Sequential Keras model is built with:
- Two hidden layers of 6 neurons each using **ReLU** activation (handles non-linearity, avoids vanishing gradients).
- An output layer with 1 neuron using **Sigmoid** activation, which outputs a probability between 0 and 1.

### Step 3 — Training

The model is compiled with the **Adam** optimizer and **Binary Crossentropy** loss (standard for binary classification), then trained for 100 epochs with a batch size of 32.

### Step 4 — Evaluation

- Predictions on the test set are thresholded at 0.5 (probability > 0.5 → predicts churn).
- A **Confusion Matrix** is generated to see true positives, false positives, true negatives, and false negatives.
- **Accuracy score** is computed on the test set.

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~86% |
| Model type | Binary Classification |
| Threshold | 0.5 |

The confusion matrix gives a breakdown of predictions vs. actual values on the 2,000-row test set, allowing evaluation of both precision and recall for the churn class.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
