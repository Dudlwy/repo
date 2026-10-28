import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data / 16.0
y_raw = digits.target


def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

y = one_hot_encode(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y_raw, random_state=42
)

def init_params(input_dim, hidden_dim, output_dim):
    
    params = {
        'W1': np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim),
        'b1': np.zeros((hidden_dim,)),
        'W2': np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim),
        'b2': np.zeros((hidden_dim,)),
        'W3': np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim),
        'b3': np.zeros((output_dim,))
    }
    return params

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward_propagation(X, params):
  
    z1 = np.dot(X, params['W1']) + params['b1']
    a1 = relu(z1)
    
    z2 = np.dot(a1, params['W2']) + params['b2']
    a2 = relu(z2)
    
  
    z3 = np.dot(a2, params['W3']) + params['b3']
    out = softmax(z3)
    
    cache = (X, z1, a1, z2, a2, z3, out)
    return out, cache

def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
    return loss

def backward_propagation(params, cache, y_true):
    X, z1, a1, z2, a2, z3, out = cache
    m = X.shape[0]
    
  
    dz3 = (out - y_true) / m
    
    dW3 = np.dot(a2.T, dz3)
    db3 = np.sum(dz3, axis=0)
    
    da2 = np.dot(dz3, params['W3'].T)
    dz2 = da2 * (z2 > 0) 
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)
    
    da1 = np.dot(dz2, params['W2'].T)
    dz1 = da1 * (z1 > 0) 
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0)
    
    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
    return grads

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_network(X_tr, y_tr, X_te, y_te, hidden_dim, lr, epochs=50, batch_size=32):
    params = init_params(64, hidden_dim, 10)
    loss_history = []
    
    for epoch in range(epochs):
   
        permutation = np.random.permutation(X_tr.shape[0])
        X_tr_shuffled = X_tr[permutation]
        y_tr_shuffled = y_tr[permutation]
        
        epoch_loss = 0
        num_batches = int(np.ceil(X_tr.shape[0] / batch_size))
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_tr_shuffled[start_idx:end_idx]
            y_batch = y_tr_shuffled[start_idx:end_idx]
            
            out, cache = forward_propagation(X_batch, params)
            batch_loss = compute_loss(y_batch, out)
            epoch_loss += batch_loss * X_batch.shape[0]
            
            grads = backward_propagation(params, cache, y_batch)
           
            for key in params:
                params[key] -= lr * grads[key]
                
        epoch_loss /= X_tr.shape[0]
        loss_history.append(epoch_loss)
        
  
    test_out, _ = forward_propagation(X_te, params)
    predictions = np.argmax(test_out, axis=1)
    true_labels = np.argmax(y_te, axis=1)
    accuracy = np.mean(predictions == true_labels)
    
    return params, loss_history, accuracy


if __name__ == "__main__":
   
    learning_rates = [0.001, 0.01, 0.1]
    widths = [16, 64]
    
    plt.figure(figsize=(10, 6))
    
    for lr in learning_rates:
        for w in widths:
            print(f"Training with Width={w}, LR={lr}...")
            params, loss_history, test_acc = train_network(
                X_train, y_train, X_test, y_test, hidden_dim=w, lr=lr
            )
            print(f"  -> Test Accuracy: {test_acc * 100:.2f}%")
            plt.plot(loss_history, label=f"W={w}, lr={lr}")
            
    plot_path = os.path.join(RESULTS_DIR, "loss_curves.png")
    plt.title("Loss Curves for Different Hyperparameters")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.show()
    print(f"Saved loss curves to: {plot_path}")