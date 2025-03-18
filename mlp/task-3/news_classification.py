import torch
import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
from news_article import NeuralNetwork
import itertools

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Common English stopwords - hardcoded to avoid NLTK dependency
ENGLISH_STOPWORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't",
    'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have',
    "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's",
    'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll",
    "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself',
    "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not',
    'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 
    'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll",
    "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's",
    'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's",
    'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those',
    'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we',
    "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when',
    "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why',
    "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're",
    "you've", 'your', 'yours', 'yourself', 'yourselves'
])

class ArticleDataProcessor:
    """Class to handle all data preprocessing operations"""
    
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vocabulary = None
        self.idf_values = None
        self.label_dict = None
    
    def sanitize_text(self, text):
        """Clean text by removing special characters and converting to lowercase"""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text
    
    def get_tokens(self, text):
        """Split text into tokens and remove stopwords"""
        words = text.split()
        return [w for w in words if w not in ENGLISH_STOPWORDS and len(w) > 1]
    
    def extract_top_terms(self, documents):
        """Build vocabulary with most frequent terms"""
        word_counts = Counter()
        for doc in documents:
            tokens = self.get_tokens(doc)
            word_counts.update(tokens)
        
        # Get top terms
        top_terms = word_counts.most_common(self.max_features)
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(top_terms)}
        return self.vocabulary
    
    def calculate_idf(self, documents):
        """Calculate inverse document frequency"""
        doc_count = len(documents)
        term_doc_count = np.zeros(len(self.vocabulary))
        
        for doc in documents:
            tokens = set(self.get_tokens(doc))  # Use set to count each term once per document
            for token in tokens:
                if token in self.vocabulary:
                    term_doc_count[self.vocabulary[token]] += 1
        
        # Calculate IDF with smoothing
        self.idf_values = np.log((doc_count + 1) / (term_doc_count + 1)) + 1
        return self.idf_values
    
    def vectorize_document(self, document):
        """Convert a single document to TF-IDF vector"""
        if self.vocabulary is None or self.idf_values is None:
            raise ValueError("Vocabulary and IDF values must be computed first")
        
        # Initialize vector
        vector = np.zeros(len(self.vocabulary))
        
        # Get tokens and their counts
        tokens = self.get_tokens(document)
        if not tokens:
            return vector
            
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        # Calculate TF-IDF
        for token, count in token_counts.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf = count / total_tokens
                vector[idx] = tf * self.idf_values[idx]
                
        return vector
    
    def vectorize_documents(self, documents):
        """Convert a list of documents to TF-IDF matrix"""
        if self.vocabulary is None or self.idf_values is None:
            raise ValueError("Vocabulary and IDF values must be computed first")
            
        # Initialize matrix
        X = np.zeros((len(documents), len(self.vocabulary)))
        
        # Process each document
        for i, doc in enumerate(documents):
            X[i] = self.vectorize_document(doc)
            
        return X
    
    def process_labels(self, labels):
        """Convert multi-label categories to binary matrix"""
        if self.label_dict is None:
            # First time processing - create label dictionary
            unique_labels = set()
            for label_str in labels:
                # Split comma-separated labels
                label_list = [l.strip() for l in str(label_str).split(',') if l.strip()]
                unique_labels.update(label_list)
            
            # Create label mapping
            self.label_dict = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        
        # Initialize binary matrix
        Y = np.zeros((len(labels), len(self.label_dict)))
        
        # Fill binary matrix
        for i, label_str in enumerate(labels):
            label_list = [l.strip() for l in str(label_str).split(',') if l.strip()]
            for label in label_list:
                if label in self.label_dict:
                    Y[i, self.label_dict[label]] = 1
                    
        return Y
    
    def prepare_data(self, train_file, test_file=None, val_split=0.2):
        """Prepare data for training and evaluation"""
        # Read training data
        print("Reading training data...")
        train_df = pd.read_csv(train_file)
        train_df.dropna(subset=['document', 'category'], inplace=True)
        
        # Clean text
        print("Cleaning text...")
        train_df['document'] = train_df['document'].apply(self.sanitize_text)
        
        # Extract features
        print("Extracting features...")
        documents = train_df['document'].tolist()
        self.extract_top_terms(documents)
        self.calculate_idf(documents)
        
        # Convert to TF-IDF
        print("Converting to TF-IDF...")
        X = self.vectorize_documents(documents)
        
        # Process labels
        print("Processing labels...")
        Y = self.process_labels(train_df['category'].tolist())
        
        # Split into train and validation
        print("Splitting into train and validation...")
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=val_split, random_state=42
        )
        
        # If test file provided, process it
        if test_file:
            print("Processing test data...")
            test_df = pd.read_csv(test_file)
            test_df.dropna(subset=['document', 'category'], inplace=True)
            test_df['document'] = test_df['document'].apply(self.sanitize_text)
            
            X_test = self.vectorize_documents(test_df['document'].tolist())
            Y_test = self.process_labels(test_df['category'].tolist())
            
            return X_train, Y_train, X_val, Y_val, X_test, Y_test
        
        return X_train, Y_train, X_val, Y_val, None, None

class NewsClassifier:
    """Class to handle model creation, training, and evaluation"""
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_model(self, hidden_sizes, activation):
        """Create neural network with specified architecture"""
        model = NeuralNetwork(device=self.device)
        
        # Add layers
        current_size = self.input_size
        
        for hidden_size in hidden_sizes:
            model.add_layer(current_size, hidden_size, activation)
            current_size = hidden_size
            
        # Output layer with sigmoid activation
        model.add_layer(current_size, self.output_size, 'sigmoid')
        
        return model
    
    def train_model(self, model, X_train, Y_train, X_val, Y_val, 
                   learning_rate=0.01, epochs=100, batch_size=32, 
                   optimizer='Mini_Batch', verbose=True):
        """Train model and return metrics"""
        # Prepare data for our neural network (transpose to match nn interface)
        X_train_t = torch.tensor(X_train.T, dtype=torch.float32, device=self.device)
        Y_train_t = torch.tensor(Y_train.T, dtype=torch.float32, device=self.device)
        X_val_t = torch.tensor(X_val.T, dtype=torch.float32, device=self.device)
        Y_val_t = torch.tensor(Y_val.T, dtype=torch.float32, device=self.device)
        
        # Initialize lists to track metrics
        train_losses = []
        val_losses = []
        val_exact_matches = []
        
        # Train the model
        start_time = time.time()
        
        # Get initial validation metrics
        val_metrics = self.evaluate_model(model, X_val, Y_val)
        val_losses.append(val_metrics['bce_loss'])
        val_exact_matches.append(val_metrics['exact_match'])
        
        # Get model's train method result (epoch losses)
        epoch_losses = model.train(
            X_train_t, Y_train_t,
            epochs=epochs,
            learning_rate=learning_rate,
            loss_function='bce',
            optimizer=optimizer,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Record training losses
        train_losses = epoch_losses
        
        # Get final validation metrics for each epoch
        for epoch in range(1, epochs):
            if epoch % 10 == 0 or epoch == epochs-1:
                val_metrics = self.evaluate_model(model, X_val, Y_val)
                val_losses.append(val_metrics['bce_loss'])
                val_exact_matches.append(val_metrics['exact_match'])
        
        train_time = time.time() - start_time
        
        # Get final validation metrics
        final_val_metrics = self.evaluate_model(model, X_val, Y_val)
        
        # Create a dictionary to store all metrics
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_exact_matches': val_exact_matches,
            'final_val_metrics': final_val_metrics,
            'train_time': train_time
        }
        
        return model, metrics
    
    def evaluate_model(self, model, X, Y, threshold=0.5):
        """Evaluate model performance"""
        # Prepare data
        X_t = torch.tensor(X.T, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y.T, dtype=torch.float32, device=self.device)
        
        # Get predictions
        with torch.no_grad():
            Y_pred = model.predict(X_t)
            Y_pred_binary = (Y_pred > threshold).float()
        
        # Calculate metrics
        hamming_loss = torch.mean(torch.abs(Y_t - Y_pred_binary)).item()
        exact_match = torch.mean(torch.all(Y_pred_binary == Y_t, dim=0).float()).item()
        
        # Calculate BCE loss
        bce_loss = -torch.mean(Y_t * torch.log(Y_pred + 1e-7) + 
                              (1 - Y_t) * torch.log(1 - Y_pred + 1e-7)).item()
        
        return {
            'hamming_loss': hamming_loss,
            'exact_match': exact_match,
            'bce_loss': bce_loss
        }
    
    def run_hyperparameter_search(self, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                                 param_grid, save_plots=True):
        """Run hyperparameter search"""
        results = []
        best_hamming_loss = float('inf')
        best_config = None
        
        # Store activation-optimizer performance
        act_opt_performance = {}
        
        # Print overall info
        print(f"Running hyperparameter search with {len(param_grid['learning_rate']) * len(param_grid['epochs']) * len(param_grid['hidden_sizes']) * len(param_grid['activation']) * len(param_grid['optimizer'])} configurations")
        
        # Run all combinations
        for lr, epochs, arch, act, opt in itertools.product(
            param_grid['learning_rate'],
            param_grid['epochs'],
            param_grid['hidden_sizes'],
            param_grid['activation'],
            param_grid['optimizer']
        ):
            config_name = f"lr={lr}_epochs={epochs}_arch={arch}_act={act}_opt={opt}"
            print(f"\nTraining with {config_name}")
            
            # Create and train model
            model = self.create_model(arch, act)
            model, metrics = self.train_model(
                model, X_train, Y_train, X_val, Y_val,
                learning_rate=lr, epochs=epochs, optimizer=opt
            )
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(model, X_test, Y_test) if X_test is not None else None
            
            # Print validation and test metrics
            print(f"Validation metrics: Hamming Loss={metrics['final_val_metrics']['hamming_loss']:.4f}, "
                  f"Exact Match={metrics['final_val_metrics']['exact_match']:.4f}, "
                  f"BCE Loss={metrics['final_val_metrics']['bce_loss']:.4f}")
            
            if test_metrics:
                print(f"Test metrics: Hamming Loss={test_metrics['hamming_loss']:.4f}, "
                      f"Exact Match={test_metrics['exact_match']:.4f}, "
                      f"BCE Loss={test_metrics['bce_loss']:.4f}")
            
            print(f"Training time: {metrics['train_time']:.2f} seconds")
            
            # Store configuration results
            config_result = {
                'learning_rate': lr,
                'epochs': epochs,
                'architecture': arch,
                'activation': act,
                'optimizer': opt,
                'val_metrics': metrics['final_val_metrics'],
                'test_metrics': test_metrics,
                'train_losses': metrics['train_losses'],
                'val_losses': metrics['val_losses'],
                'val_exact_matches': metrics['val_exact_matches'],
                'train_time': metrics['train_time']
            }
            results.append(config_result)
            
            # Track best configuration
            if metrics['final_val_metrics']['hamming_loss'] < best_hamming_loss:
                best_hamming_loss = metrics['final_val_metrics']['hamming_loss']
                best_config = config_result
            
            # Store in activation-optimizer mapping
            act_opt_key = (act, opt)
            if act_opt_key not in act_opt_performance or metrics['final_val_metrics']['hamming_loss'] < act_opt_performance[act_opt_key]['hamming_loss']:
                act_opt_performance[act_opt_key] = {
                    'hamming_loss': metrics['final_val_metrics']['hamming_loss'],
                    'exact_match': metrics['final_val_metrics']['exact_match'],
                    'config': config_result
                }
            
            # Plot learning curve if requested
            if save_plots:
                plt.figure(figsize=(10, 6))
                
                # Plot available epochs
                epochs_trained = len(metrics['train_losses'])
                x_train = list(range(1, epochs_trained + 1))
                
                # Plot training loss
                train_line, = plt.plot(x_train, metrics['train_losses'], label='Training Loss', 
                                      color='blue', linewidth=2)
                
                # Plot validation metrics (may have fewer points)
                if metrics['val_losses']:
                    # Create x points for validation (every 10 epochs)
                    x_val = list(range(0, epochs_trained, max(1, epochs_trained // len(metrics['val_losses']))))
                    if len(x_val) < len(metrics['val_losses']):
                        x_val.append(epochs_trained)
                    x_val = x_val[:len(metrics['val_losses'])]
                    
                    val_line, = plt.plot(x_val, metrics['val_losses'], label='Validation Loss', 
                                        marker='o', color='orange', linewidth=2)
                    
                    # Plot validation exact match on secondary axis if available
                    if metrics['val_exact_matches']:
                        ax2 = plt.gca().twinx()
                        exact_line, = ax2.plot(x_val, metrics['val_exact_matches'], label='Exact Match', 
                                              color='green', marker='s', linestyle='--', linewidth=2)
                        ax2.set_ylabel('Exact Match (Accuracy)', fontsize=12)
                        ax2.set_ylim([0, 1])
                        
                        # Create a combined legend
                        lines = [train_line, val_line, exact_line]
                        labels = ['Training Loss (BCE)', 'Validation Loss (BCE)', 'Exact Match (Accuracy)']
                        plt.legend(lines, labels, loc='upper right', fontsize=10)
                    else:
                        plt.legend(loc='upper right', fontsize=10)
                else:
                    plt.legend(loc='upper right', fontsize=10)
                
                plt.title(f'Training & Validation Metrics: {config_name}', fontsize=14)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('BCE Loss', fontsize=12)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'metrics_{act}_{opt}_lr{lr}.png', dpi=300)
                plt.close()
        
        # Sort results by test performance if available, otherwise by validation
        if X_test is not None:
            results.sort(key=lambda x: x['test_metrics']['hamming_loss'] if x['test_metrics'] else float('inf'))
            metric_key = 'test_metrics'
        else:
            results.sort(key=lambda x: x['val_metrics']['hamming_loss'])
            metric_key = 'val_metrics'
        
        # Print results in a tabular format
        print("\n======= MODEL CONFIGURATIONS RANKED BY HAMMING LOSS (BEST TO WORST) =======")
        print(f"{'Rank':<5}{'Learning Rate':<15}{'Architecture':<20}{'Activation':<10}{'Optimizer':<15}{'Hamming Loss':<12}{'Exact Match':<12}{'BCE Loss':<12}")
        print("="*89)
        
        for i, res in enumerate(results):
            metrics = res[metric_key]
            if metrics:
                print(f"{i+1:<5}{res['learning_rate']:<15}{str(res['architecture']):<20}{res['activation']:<10}{res['optimizer']:<15}"
                      f"{metrics['hamming_loss']:.5f}{'':<6}{metrics['exact_match']:.5f}{'':<6}{metrics['bce_loss']:.5f}")
        
        # Print best configuration
        print("\n===== BEST CONFIGURATION =====")
        print(f"Learning Rate: {best_config['learning_rate']}")
        print(f"Architecture: {best_config['architecture']}")
        print(f"Activation Function: {best_config['activation']}")
        print(f"Optimizer: {best_config['optimizer']}")
        print(f"Hamming Loss: {best_config[metric_key]['hamming_loss']:.5f}")
        print(f"Exact Match: {best_config[metric_key]['exact_match']:.5f}")
        print(f"BCE Loss: {best_config[metric_key]['bce_loss']:.5f}")
        
        return results, best_config, act_opt_performance
    
    def train_final_model(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, config):
        """Train final model with best configuration"""
        print(f"\nTraining final model with configuration: "
              f"LR={config['learning_rate']}, Epochs={config['epochs']}, "
              f"Architecture={config['architecture']}, Activation={config['activation']}, "
              f"Optimizer={config['optimizer']}")
        
        # Create model
        model = self.create_model(config['architecture'], config['activation'])
        
        # Train model
        model, metrics = self.train_model(
            model, X_train, Y_train, X_val, Y_val,
            learning_rate=config['learning_rate'],
            epochs=config['epochs'],
            optimizer=config['optimizer'],
            verbose=True
        )
        
        # Plot final learning curve
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        epochs_trained = len(metrics['train_losses'])
        x_train = list(range(1, epochs_trained + 1))
        train_line, = plt.plot(x_train, metrics['train_losses'], label='Training Loss', 
                              color='blue', linewidth=2)
        
        # Plot validation metrics
        if metrics['val_losses']:
            # Create x points for validation (every 10 epochs)
            x_val = list(range(0, epochs_trained, max(1, epochs_trained // len(metrics['val_losses']))))
            if len(x_val) < len(metrics['val_losses']):
                x_val.append(epochs_trained)
            x_val = x_val[:len(metrics['val_losses'])]
            
            val_line, = plt.plot(x_val, metrics['val_losses'], label='Validation Loss', 
                                marker='o', color='orange', linewidth=2)
            
            # Plot validation exact match on secondary axis
            if metrics['val_exact_matches']:
                ax2 = plt.gca().twinx()
                exact_line, = ax2.plot(x_val, metrics['val_exact_matches'], label='Exact Match', 
                                      color='green', marker='s', linestyle='--', linewidth=2)
                ax2.set_ylabel('Exact Match (Accuracy)', fontsize=12)
                ax2.set_ylim([0, 1])
                
                # Create a combined legend
                lines = [train_line, val_line, exact_line]
                labels = ['Training Loss (BCE)', 'Validation Loss (BCE)', 'Exact Match (Accuracy)']
                plt.legend(lines, labels, loc='upper right', fontsize=10)
            else:
                plt.legend(loc='upper right', fontsize=10)
        else:
            plt.legend(loc='upper right', fontsize=10)
        
        plt.title('Final Model Training & Validation Metrics', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('BCE Loss', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('final_model_metrics.png', dpi=300)
        plt.close()
        
        # Evaluate on test set
        test_metrics = self.evaluate_model(model, X_test, Y_test)
        
        print(f"Final Test Metrics:")
        print(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}")
        print(f"  Exact Match: {test_metrics['exact_match']:.4f}")
        print(f"  BCE Loss: {test_metrics['bce_loss']:.4f}")
        
        return model, test_metrics

def main():
    # Define file paths
    train_file = "news-article/train.csv"
    test_file = "news-article/test.csv"
    
    # Initialize data processor
    data_processor = ArticleDataProcessor(max_features=5000)
    
    # Process data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data_processor.prepare_data(
        train_file, test_file, val_split=0.2
    )
    
    # Print data shapes
    print(f"Data shapes:")
    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"X_val: {X_val.shape}, Y_val: {Y_val.shape}")
    print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")
    
    # Initialize classifier
    classifier = NewsClassifier(X_train.shape[1], Y_train.shape[1])
    
    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.01, 0.05],
        'epochs': [50, 100],
        'hidden_sizes': [[256, 128], [512, 256, 128]],
        'activation': ['relu', 'tanh', 'sigmoid'],
        'optimizer': ['Batch', 'Mini_Batch', 'SGD']
    }
    
    # Run hyperparameter search
    results, best_config, act_opt_performance = classifier.run_hyperparameter_search(
        X_train, Y_train, X_val, Y_val, X_test, Y_test, param_grid
    )
    
    # Train final model with best configuration
    final_model, final_metrics = classifier.train_final_model(
        X_train, Y_train, X_val, Y_val, X_test, Y_test, best_config
    )

if __name__ == "__main__":
    main() 