import torch

def activation(x: torch.Tensor, function: str) -> torch.Tensor:
    if function == 'relu':
        return torch.where(x > 0, x, torch.zeros_like(x))
    elif function == 'sigmoid':
        return 1 / (1 + torch.exp(-x))
    elif function == 'tanh':
        return torch.tanh(x)
    elif function == 'linear':
        return x
    else:
        raise ValueError(f"Activation function {function} not supported")

def act_der(x: torch.Tensor, function: str) -> torch.Tensor:
    if function == 'relu':
        return (x > 0).float()
    elif function == 'sigmoid':
        sig = activation(x, 'sigmoid')
        return sig * (1 - sig)
    elif function == 'tanh':
        return 1 - activation(x, 'tanh').pow(2)
    elif function == 'linear':
        return torch.ones_like(x)
    else:
        raise ValueError(f"Activation function {function} not supported")

def loss(gt: torch.Tensor, pred: torch.Tensor, function: str) -> torch.Tensor:
    # Ensure tensors are on the same device
    if gt.device != pred.device:
        gt = gt.to(pred.device)
        
    if function == 'mse':
        return torch.mean((pred - gt).pow(2))
    elif function == 'bce':
        return -torch.mean(gt * torch.log(pred + 1e-7) + (1 - gt) * torch.log(1 - pred + 1e-7))

def loss_der(gt: torch.Tensor, pred: torch.Tensor, function: str) -> torch.Tensor:
    # Ensure tensors are on the same device
    if gt.device != pred.device:
        gt = gt.to(pred.device)
        
    if function == 'mse':
        return 2 * (pred - gt) / gt.size(1)
    elif function == 'bce':
        return -gt / (pred + 1e-7) + (1 - gt) / (1 - pred + 1e-7)
    
class NeuralNetwork:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.layers = []
        self.z_list = []
        self.a_list = []
        self.device = device

    def add_layer(self, input_size: int, output_size: int, activation: str):
        # Initialize weights using He initialization for ReLU or Xavier for sigmoid/tanh
        if activation == 'relu':
            w = torch.randn(output_size, input_size, device=self.device) * torch.sqrt(torch.tensor(2.0 / input_size, device=self.device))
        else:
            w = torch.randn(output_size, input_size, device=self.device) * torch.sqrt(torch.tensor(1.0 / input_size, device=self.device))
            
        b = torch.zeros(output_size, 1, device=self.device)
        self.layers.append([w, b, activation])
        self.z_list.append(None)
        self.a_list.append(None)

    def forwardprop(self, x: torch.Tensor) -> torch.Tensor:
        current_input = x.to(self.device)  # Ensure input is on the correct device
        for i in range(len(self.layers)):
            w, b, activation_func = self.layers[i]
            z = torch.mm(w, current_input) + b
            self.z_list[i] = z
            self.a_list[i] = activation(z, activation_func)
            current_input = self.a_list[i]
        return self.a_list[-1]

    def backprop(self, x: torch.Tensor, y: torch.Tensor, loss_function: str, learning_rate: float = 0.01):
        m = x.size(1)  # Number of samples is now the second dimension
        self.forwardprop(x)
        
        dz = loss_der(y, self.a_list[-1], loss_function)
        for i in range(len(self.layers)-1, -1, -1):
            w, b, activation_func = self.layers[i]
            a_prev = self.a_list[i-1] if i > 0 else x
            
            dz = dz * act_der(self.z_list[i], activation_func)
            dw = (1/m) * torch.mm(dz, a_prev.T)
            db = (1/m) * torch.sum(dz, dim=1, keepdim=True)
            
            self.layers[i][0] = w - learning_rate * dw
            self.layers[i][1] = b - learning_rate * db
            
            if i > 0:
                dz = torch.mm(w.T, dz)

    def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100, 
             batch_size: int = 32, learning_rate: float = 0.01, 
             loss_function: str = 'mse', optimizer: str = 'Mini_Batch', verbose: bool = True):
        n_samples = X.size(1)  # Now samples are columns
        
        X = X.to(self.device)  # Move X to the correct device
        y = y.to(self.device)  # Move y to the correct device
        
        epoch_losses = []

        for epoch in range(epochs):
            total_loss = 0.0
            # Generate random permutation indices on the same device as X
            perm = torch.randperm(n_samples, device=self.device)
            X_shuffled = X[:, perm]
            y_shuffled = y[:, perm]  # Correctly index y with perm
            
            if optimizer == 'Batch':
                # Process all samples at once
                pred = self.forwardprop(X_shuffled)
                batch_loss = loss(y_shuffled, pred, loss_function)
                total_loss += batch_loss.item()
                self.backprop(X_shuffled, y_shuffled, loss_function, learning_rate)
            
            elif optimizer == 'SGD':
                # Stochastic Gradient Descent - process one sample at a time
                for i in range(n_samples):
                    X_sample = X_shuffled[:, i:i+1]  # Get a single sample
                    y_sample = y_shuffled[:, i:i+1]  # Get corresponding label
                    
                    pred = self.forwardprop(X_sample)
                    sample_loss = loss(y_sample, pred, loss_function)
                    total_loss += sample_loss.item()
                    
                    self.backprop(X_sample, y_sample, loss_function, learning_rate)
                
                total_loss = total_loss / n_samples  # Average the loss over samples
            
            else:  # Mini_Batch (default)
                n_batches = (n_samples + batch_size - 1) // batch_size
                
                for batch in range(n_batches):
                    start_idx = batch * batch_size
                    end_idx = min((batch + 1) * batch_size, n_samples)
                    
                    X_batch = X_shuffled[:, start_idx:end_idx]
                    y_batch = y_shuffled[:, start_idx:end_idx]
                    
                    pred = self.forwardprop(X_batch)
                    batch_loss = loss(y_batch, pred, loss_function)
                    total_loss += batch_loss.item()
                    
                    self.backprop(X_batch, y_batch, loss_function, learning_rate)
                
                total_loss = total_loss / n_batches  # Average loss over batches
            
            epoch_losses.append(total_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
                
        return epoch_losses  # Return the loss history

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network to make predictions"""
        return self.forwardprop(X)

