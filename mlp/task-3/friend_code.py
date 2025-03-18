import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from math import log
import math
import re
from bs4 import BeautifulSoup  # For HTML cleaning
from torch.utils.data import TensorDataset, DataLoader
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import time




STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """Lowercase and remove non-alphanumeric characters."""
    text = str(text).lower()  # Ensure text is string
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def tokenize(text):
    """Split text by whitespace and remove stopwords."""
    tokens = text.split()
    tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens

def build_vocabulary(documents, max_features=5000):
    """Build vocabulary of the top max_features tokens from the documents."""
    counter = Counter()
    for doc in documents:
        tokens = tokenize(doc)
        counter.update(tokens)
    vocab = {word: i for i, (word, _) in enumerate(counter.most_common(max_features))}
    return vocab

def compute_idf(documents, vocab):
    """Compute inverse document frequency for each word in vocab."""
    N = len(documents)
    df = np.zeros(len(vocab))
    for doc in documents:
        tokens = set(tokenize(doc))
        for token in tokens:
            if token in vocab:
                df[vocab[token]] += 1
    idf = np.log((N + 1) / (df + 1)) + 1
    return idf

def compute_tf_idf(documents, vocab, idf):
    """Compute TF-IDF for each document given a vocabulary and idf vector."""
    X = np.zeros((len(documents), len(vocab)))
    for i, doc in enumerate(documents):
        tokens = tokenize(doc)
        if len(tokens) == 0:
            continue
        tf_counter = Counter(tokens)
        total_count = len(tokens)
        for token, count in tf_counter.items():
            if token in vocab:
                j = vocab[token]
                tf = count / total_count
                X[i, j] = tf * idf[j]
    return X

def multi_label_binarize(labels_list, label_mapping=None):
    """Convert list of comma-separated labels to a binary matrix."""
    if label_mapping is None:  # For training data
        label_set = set()
        split_labels = []
        for labels in labels_list:
            topics = [label.strip() for label in str(labels).split(',') if label.strip()]
            split_labels.append(topics)
            label_set.update(topics)
        label_list = sorted(list(label_set))
        label_mapping = {label: idx for idx, label in enumerate(label_list)}
    else:  # For test data, use existing label_mapping
        split_labels = []
        for labels in labels_list:
            topics = [label.strip() for label in str(labels).split(',') if label.strip()]
            split_labels.append(topics)

    Y = np.zeros((len(labels_list), len(label_mapping)))
    for i, topics in enumerate(split_labels):
        for topic in topics:
            if topic in label_mapping:
                Y[i, label_mapping[topic]] = 1
    return Y, label_mapping

def preprocess_data(train_csv_path, test_csv_path=None, max_features=5000, val_size=0.2):
    """Load CSV, clean documents, compute TF-IDF, binarize labels,
       and split into training, validation, and test sets."""

    # Load and preprocess training data
    df_train = pd.read_csv(train_csv_path)
    df_train.dropna(subset=['document', 'category'], inplace=True)
    df_train['document'] = df_train['document'].apply(clean_text)

    train_docs = df_train['document'].tolist()
    train_labels = df_train['category'].tolist()

    # Build vocabulary and IDF from training data only
    vocab = build_vocabulary(train_docs, max_features=max_features)
    idf = compute_idf(train_docs, vocab)

    # Compute TF-IDF for training data
    X_full = compute_tf_idf(train_docs, vocab, idf)
    Y_full, label_mapping = multi_label_binarize(train_labels)

    # Split into train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_full, Y_full, test_size=val_size, random_state=42
    )

    # If test CSV is provided, preprocess it
    if test_csv_path:
        df_test = pd.read_csv(test_csv_path)
        df_test.dropna(subset=['document', 'category'], inplace=True)
        df_test['document'] = df_test['document'].apply(clean_text)

        test_docs = df_test['document'].tolist()
        test_labels = df_test['category'].tolist()

        # Compute TF-IDF for test data using training vocab and IDF
        X_test = compute_tf_idf(test_docs, vocab, idf)
        Y_test, _ = multi_label_binarize(test_labels, label_mapping=label_mapping)  # Use training label mapping

        return X_train, X_val, X_test, Y_train, Y_val, Y_test, vocab, label_mapping

    # If no test CSV, return only train and val
    return X_train, X_val, None, Y_train, Y_val, None, vocab, label_mapping

# Usage example
train_csv_path = "/content/drive/MyDrive/news-article/train.csv"  # Replace with your training CSV path
test_csv_path = "/content/drive/MyDrive/news-article/test.csv"

X_train, X_val, X_test, y_train, y_val, y_test, vocab, label_mapping = preprocess_data(
    train_csv_path, test_csv_path, max_features=5000, val_size=0.2
)

# Convert to PyTorch tensors for use with your MLP
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_train shape: {y_train.shape}")
print(f"Y_val shape: {y_val.shape}")
print(f"Y_test shape: {y_test.shape}")


import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from itertools import product

class MLP:
    def __init__(self, input_size, output_size, hidden_sizes, learning_rate,
                 activation_function, loss_function, threshold=0.3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.learning_rate = learning_rate
        self.activation = activation_function
        self.hidden_sizes = hidden_sizes
        self.loss_function = loss_function.lower()
        self.threshold = threshold
        self.device = torch.device(device)  # Set device (GPU or CPU)

    def _Initialize(self, input_size, output_size):
        self.weights = []
        self.bias = []
        prev_size = input_size
        for hidden_size in self.hidden_sizes:
            weight = torch.randn(hidden_size, prev_size, dtype=torch.float32) * 0.01
            bias = torch.randn(hidden_size, 1, dtype=torch.float32) * 0.01
            self.weights.append(weight.to(self.device))
            self.bias.append(bias.to(self.device))
            prev_size = hidden_size
        weight = torch.randn(output_size, prev_size, dtype=torch.float32) * 0.01
        bias = torch.randn(output_size, 1, dtype=torch.float32) * 0.01
        self.weights.append(weight.to(self.device))
        self.bias.append(bias.to(self.device))

    def _Activation(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'linear':
            return x

    def _ActivationPrime(self, x):
        if self.activation == 'relu':
            return torch.where(x > 0, torch.tensor(1.0, dtype=x.dtype, device=self.device),
                             torch.tensor(0.0, dtype=x.dtype, device=self.device))
        elif self.activation == 'sigmoid':
            sig = torch.sigmoid(x)
            return sig * (1 - sig)
        elif self.activation == 'tanh':
            return 1 - torch.tanh(x) ** 2
        elif self.activation == 'linear':
            return torch.ones_like(x)

    def _Softmax(self, x):
        exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
        return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

    def _CrossEntropy(self, y, y_hat):
        return -torch.mean(torch.sum(y * torch.log(y_hat + 1e-9), dim=1))

    def _MSE(self, y, y_hat):
        return torch.mean((y-y_hat)**2)

    def _BCE(self, y, y_hat):
        return -torch.mean(y * torch.log(y_hat + 1e-9) + (1 - y) * torch.log(1 - y_hat + 1e-9))

    def _loss(self, y, y_hat):
        if self.loss_function == "crossentropy":
            return self._CrossEntropy(y, y_hat)
        elif self.loss_function == "bce":
            return self._BCE(y, y_hat)
        return self._MSE(y, y_hat)

    def _CrossEntropyderivative(self, y, y_hat):
        return (y_hat-y)

    def _MSEderivative(self, y, y_hat):
        return 2 * (y_hat - y)

    def _BCEderivative(self, y, y_hat):
        # Derivative of Binary Cross Entropy: -(y/y_hat - (1-y)/(1-y_hat))
        # return -(y / (y_hat + 1e-9) - (1 - y) / (1 - y_hat + 1e-9))
        return (y_hat - y)

    def _Forward(self, X):
        activations = X
        self.layer_inputs = []
        self.z = []
        for i in range(len(self.weights) - 1):
            z = torch.mm(activations, self.weights[i].T) + self.bias[i].T
            activations = self._Activation(z)
            self.layer_inputs.append(activations)
            self.z.append(z)
        z = torch.mm(activations, self.weights[-1].T) + self.bias[-1].T
        self.layer_inputs.append(z)
        return torch.sigmoid(z) if self.loss_function == "bce" else z

    def _Backward(self, X, y, y_hat):
        batch_size = X.shape[0]
        if self.loss_function == "crossentropy":
            dz = self._CrossEntropyderivative(y, y_hat) / batch_size
        elif self.loss_function == "bce":
            dz = self._BCEderivative(y, y_hat) / batch_size
        else:
            dz = self._MSEderivative(y, y_hat) / batch_size

        grads_w = []
        grads_b = []
        for i in range(len(self.weights) - 1, -1, -1):
            dw = torch.mm(dz.T, self.layer_inputs[i - 1] if i > 0 else X)
            db = torch.sum(dz, dim=0, keepdim=True).T
            if i > 0:
                dz = torch.mm(dz, self.weights[i]) * self._ActivationPrime(self.z[i - 1])
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.bias[i] -= self.learning_rate *grads_b[i]

    def batch_fit(self, X, y, X_val, y_val, epochs, showloss=False):
        X, y = X.to(self.device), y.to(self.device)
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        self._Initialize(X.shape[1], y.shape[1])
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            y_hat = self._Forward(X)
            train_loss = self._loss(y, y_hat)
            train_losses.append(train_loss.item())
            self._Backward(X, y, y_hat)

            with torch.no_grad():
                y_val_hat = self._Forward(X_val)
                val_loss = self._loss(y_val, y_val_hat)
                val_losses.append(val_loss.item())

            if epoch % 10 == 0 and showloss:
                y_pred_binary = (y_val_hat >= self.threshold).float()
                print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} exact_match:{torch.mean(torch.all(y_pred_binary == y_val, dim=1).float())}')
        return train_losses, val_losses

    def Mini_batch_fit(self, X, y, X_val, y_val, epochs, batch_size, showloss=False):
        X, y = X.to(self.device), y.to(self.device)
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self._Initialize(X.shape[1], y.shape[1])
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_hat = self._Forward(X_batch)
                self._Backward(X_batch, y_batch, y_hat)
            with torch.no_grad():
                y_train_pred = self._Forward(X)
                train_loss = self._loss(y, y_train_pred)
                train_losses.append(train_loss.item())
                y_val_pred = self._Forward(X_val)
                val_loss = self._loss(y_val, y_val_pred)
                val_losses.append(val_loss.item())
            if epoch % 10 == 0 and showloss:
                y_pred_binary = (y_val_pred >= self.threshold).float()
                print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} exact_match:{torch.mean(torch.all(y_pred_binary == y_val, dim=1).float())}')
        return train_losses, val_losses

    def SGD_fit(self, X, y, X_val, y_val, epochs, showloss=False):
        X, y = X.to(self.device), y.to(self.device)
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        self._Initialize(X.shape[1], y.shape[1])
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                X_batch = X[i].unsqueeze(0)
                y_batch = y[i].unsqueeze(0)
                y_hat = self._Forward(X_batch)
                self._Backward(X_batch, y_batch, y_hat)
            with torch.no_grad():
                y_train_pred = self._Forward(X)
                train_loss = self._loss(y, y_train_pred)
                train_losses.append(train_loss.item())
                y_val_pred = self._Forward(X_val)
                val_loss = self._loss(y_val, y_val_pred)
                val_losses.append(val_loss.item())
            if  showloss:
                y_pred_binary = (y_val_pred >= self.threshold).float()
                print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} exact_match:{torch.mean(torch.all(y_pred_binary == y_val, dim=1).float())}')
        return train_losses, val_losses

    def predict(self, X):
        X = X.to(self.device)
        y_hat = self._Forward(X)
        if self.loss_function == "crossentropy":
            return torch.argmax(y_hat, dim=1)
        elif self.loss_function == "bce":
            return (y_hat > self.threshold).float()
        return y_hat

    def predict_proba(self, X):
        X = X.to(self.device)
        return self._Forward(X)

    def _RMSE(self, y_true, y_pred):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

    def _R2_score(self, y_true, y_pred):
        ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
        ss_residual = torch.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def _hamming_error(self, y_true, y_pred):
        if y_pred.dim() > 1 and y_pred.shape[1] > 1:
            if self.loss_function == "bce":
                y_pred_binary = (y_pred > self.threshold).float()
            else:
                y_pred_binary = y_pred
        else:
            y_pred_binary = y_pred
        incorrect_predictions = torch.sum(torch.abs(y_true - y_pred_binary))
        total_predictions = y_true.numel()
        return incorrect_predictions / total_predictions

    def evaluate(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        y_pred = self._Forward(X)
        if self.loss_function == "crossentropy":
            accuracy = torch.mean((torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).float())
            loss = self._CrossEntropy(y, y_pred)
            return {
                'accuracy': accuracy.item(),
                'loss': loss.item()
            }
        elif self.loss_function == "bce":
            y_pred_binary = (y_pred > self.threshold).float()
            loss = self._BCE(y, y_pred)
            hamming_error = self._hamming_error(y, y_pred)
            exact_match = torch.mean(torch.all(y_pred_binary == y, dim=1).float())
            return {
                'hamming_error': hamming_error.item(),
                'exact_match': exact_match.item(),
                'loss': loss.item()
            }
        else:
            rmse = self._RMSE(y, y_pred)
            r2 = self._R2_score(y, y_pred)
            return {
                'rmse': rmse.item(),
                'r2': r2.item(),
                'loss': self._MSE(y, y_pred).item()
            }

def experiment(X_train, y_train, X_val, y_val, X_test, y_test, learning_rates, epochs_list,
               architectures, activations, optimizers,
               device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Move all data to GPU at the start
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    results = []
    act_opt_scores = {}  # Dictionary to store scores by (activation, optimizer) pair

    # Iterate over all hyperparameter combinations
    for lr, epoch, arch, act, opt in product(learning_rates, epochs_list, architectures, activations, optimizers):
        print(f"\nTraining with lr={lr}, epochs={epoch}, arch={arch}, act={act}, opt={opt}")
        mlp = MLP(X_train.shape[1], y_train.shape[1], arch, lr, act, "bce", device=device)

        # Train the model based on optimizer
        if opt == "Batch":
            train_losses, val_losses = mlp.batch_fit(X_train, y_train, X_val, y_val, epoch)
        elif opt == "Mini_Batch":
            train_losses, val_losses = mlp.Mini_batch_fit(X_train, y_train, X_val, y_val, epoch, batch_size=32)
        else:  # SGD
            train_losses, val_losses = mlp.SGD_fit(X_train, y_train, X_val, y_val, epoch)

        # Evaluate on validation set
        val_metrics = mlp.evaluate(X_val, y_val)
        results.append((lr, epoch, arch, act, opt, val_metrics, train_losses, val_losses))

        # Evaluate on test set
        test_metrics = mlp.evaluate(X_test, y_test)
        print(f"Validation Metrics: {val_metrics}")
        print(f"Test Metrics: {test_metrics}")

        # Store score for (activation, optimizer) pair (using hamming_error, lower is better)
        key = (act, opt)
        score = val_metrics['hamming_error']  # Negative because lower hamming_error is better
        if key not in act_opt_scores or score < act_opt_scores[key][0]:
            act_opt_scores[key] = (score, val_metrics, test_metrics)

        # Plot training curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Config: lr={lr}, arch={arch}, act={act}, opt={opt}')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Sort results by hamming_error (lower is better)
    results.sort(key=lambda x: x[5]['hamming_error'])

    # Print best configurations
    print("\nTop 3 Configurations (based on validation set):")
    for res in results[:3]:
        print(f"LR: {res[0]}, Epochs: {res[1]}, Arch: {res[2]}, Act: {res[3]}, Opt: {res[4]}, "
              f"Val Metrics: {res[5]}")

    # Report ordered scores for each (activation, optimizer) combination
    print("\nOrdered Scores by Activation and Optimizer (based on validation set):")
    sorted_act_opt = sorted(act_opt_scores.items(), key=lambda x: x[1][0])  # Ascending because score is -hamming_error
    for (act, opt), (score, val_metrics, test_metrics) in sorted_act_opt:
        print(f"Act: {act}, Opt: {opt}, Score (hamming_error): {val_metrics['hamming_error']:.4f}, "
              f"Val Metrics: {val_metrics}, Test Metrics: {test_metrics}")

    # Identify and train best configuration
    best_config = results[0]
    print(f"\nBest Configuration: LR={best_config[0]}, Epochs={best_config[1]}, Arch={best_config[2]}, "
          f"Act={best_config[3]}, Opt={best_config[4]}")

    best_mlp = MLP(X_train.shape[1], y_train.shape[1], best_config[2], best_config[0],
                   best_config[3], "bce", device=device)
    if best_config[4] == "Batch":
        train_losses, val_losses = best_mlp.batch_fit(X_train, y_train, X_val, y_val, best_config[1])
    elif best_config[4] == "Mini_Batch":
        train_losses, val_losses = best_mlp.Mini_batch_fit(X_train, y_train, X_val, y_val, best_config[1], batch_size=32)
    else:
        train_losses, val_losses = best_mlp.SGD_fit(X_train, y_train, X_val, y_val, best_config[1])

    # Final evaluation on test set for best config
    best_test_metrics = best_mlp.evaluate(X_test, y_test)
    print(f"Final Test Metrics for Best Config: {best_test_metrics}")

    # Plot training curves for best config
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Best Config: {best_config[3]}-{best_config[4]}-LR{best_config[0]}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return results, best_config

start = time.time()
input_size = X_train.shape[1]
mlp = MLP(input_size=input_size, output_size=y_train.shape[1], hidden_sizes= [256,128,64], learning_rate=0.1, activation_function='relu', loss_function='bce',threshold = 0.5)

#training using mini_batch
epochs = 200
mlp.Mini_batch_fit(X_train,y_train, X_val, y_val,epochs,32,showloss = True)
metrics = mlp.evaluate(X_val, y_val)

print(f"mini Batch Grad descent")
print(f'exact_match:{metrics["exact_match"]}')
print(f'Hamming error:{metrics["hamming_error"]}')

end = time.time()
print(f"time took:{(end-start):.4f}")


# Example usage (you need to define your data first)
# Hyperparameters
learning_rates = [0.05,0.1]
epochs_list = [100,200]
architectures = [[1024,64], [256,128,64]]
activations = ['relu', 'sigmoid','tanh']
optimizers = ['Batch', 'Mini_Batch']

# For multi-label classification
results, best_config = experiment(X_train, y_train, X_val, y_val, X_test, y_test,
                                learning_rates, epochs_list, architectures, activations,
                                optimizers)