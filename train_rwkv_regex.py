import json
import time  # Added for timing epochs

import matplotlib.pyplot as plt  # Added for plotting
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score  # Added f1_score
from torch.utils.data import DataLoader, Dataset

from config import MAX_LEN, MODEL_HYPERPARAMETERS  # Added import
# Import the model (assuming rwkv_model.py is in the same directory)
from rwkv_model import RWKV7_Model_Classifier

# --- Configuration ---
# Removed D_MODEL, N_LAYER, HEAD_SIZE, FFN_HIDDEN_MULTIPLIER, LORA_DIM_W, LORA_DIM_A, LORA_DIM_V, LORA_DIM_G
# VOCAB_SIZE will be loaded from dataset, but should be 4 for {'<pad>':0, 'a':1, 'b':2, 'c':3}

BATCH_SIZE = 1536
LEARNING_RATE = 5e-4
NUM_EPOCHS = 100         # Maximum number of epochs
EARLY_STOPPING_TARGET_ACC = 1.0  # Aim for perfect accuracy
EARLY_STOPPING_PATIENCE_COUNT = 3  # Stop if val_acc == 1.0 for this many consecutive epochs
NO_IMPROVEMENT_PATIENCE = 15  # More patience for complex patterns
VAL_ACC_THRESHOLD = 0.9999  # Very high accuracy threshold for early termination
EARLY_STOPPING_TRAIN_ACC_MIN = 0.99  # Minimum training accuracy to consider early stopping

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_FILE = 'regex_dataset.json'

# Global VOCAB, will be loaded from dataset file
VOCAB = {}

# --- Dataset Class ---
class RegexDataset(Dataset):
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs # list of (sequence_as_ints, label)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        seq, label = self.data_pairs[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# --- Custom Collate Function for DataLoader ---
def collate_fn(batch):
    sequences, labels = zip(*batch)
    
    # Determine max length in the current batch
    # Handle truly empty sequences by giving them a length of 1 for padding purposes
    current_max_len = 0
    for s in sequences:
        if len(s) > current_max_len:
            current_max_len = len(s) # Fixed: assign len(s)
            
    if current_max_len == 0: # All sequences in batch are empty
        current_max_len = 1 # Pad to length 1 with <pad>

    padded_sequences = []
    for s in sequences:
        seq_len = len(s)
        if seq_len == 0:            
            padded_sequences.append(torch.full((current_max_len,), VOCAB.get('<pad>', 0), dtype=torch.long)) # Pad with <pad> ID
        else:
            padding_needed = current_max_len - seq_len
            # Ensure sequence is not longer than MAX_LEN before padding
            # This logic assumes sequences are already truncated if necessary before collate_fn
            # or that current_max_len will not exceed a model-defined MAX_LEN
            padded_seq = torch.tensor(s[:MAX_LEN] + [VOCAB.get('<pad>', 0)] * padding_needed, dtype=torch.long) 
            padded_sequences.append(padded_seq)
                
    return torch.stack(padded_sequences), torch.stack(labels).unsqueeze(1)


# --- Main Training and Evaluation Function ---
def train_and_evaluate():
    global VOCAB # Allow modification of global VOCAB
    print(f"Using device: {DEVICE}")

    # 1. Load Dataset
    try:
        with open(DATASET_FILE, 'r') as f:
            dataset_obj = json.load(f)
        train_data_pairs = dataset_obj['train_data']
        val_data_pairs = dataset_obj['val_data']
        VOCAB = dataset_obj['vocab'] # Load vocab
        vocab_size = len(VOCAB)
        print(f"Dataset loaded. Vocab size: {vocab_size}. Train samples: {len(train_data_pairs)}, Val samples: {len(val_data_pairs)}")
    except FileNotFoundError:
        print(f"Error: {DATASET_FILE} not found. Please run dataset_generator.py first.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    train_dataset = RegexDataset(train_data_pairs)
    val_dataset = RegexDataset(val_data_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 2. Initialize Model
    model = RWKV7_Model_Classifier(
        d_model=MODEL_HYPERPARAMETERS["D_MODEL"],
        n_layer=MODEL_HYPERPARAMETERS["N_LAYER"],
        vocab_size=vocab_size, # Use loaded vocab size
        head_size=MODEL_HYPERPARAMETERS["HEAD_SIZE"],
        ffn_hidden_multiplier=MODEL_HYPERPARAMETERS["FFN_HIDDEN_MULTIPLIER"],
        lora_dim_w=MODEL_HYPERPARAMETERS["LORA_DIM_W"],
        lora_dim_a=MODEL_HYPERPARAMETERS["LORA_DIM_A"],
        lora_dim_v=MODEL_HYPERPARAMETERS["LORA_DIM_V"],
        lora_dim_g=MODEL_HYPERPARAMETERS["LORA_DIM_G"]
    ).to(DEVICE)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {trainable_params} trainable parameters.")

    # 3. Loss Function and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. Training and Validation Loop
    best_val_accuracy = 0.0
    epochs_since_last_improvement = 0
    consecutive_perfect_epochs = 0
    consecutive_near_perfect_epochs = 0

    # History for plotting
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    train_f1_history = [] # Added for F1 score
    val_f1_history = []   # Added for F1 score

    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_targets = []

        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # RWKV states are managed per call if states_list_prev=None
            logits, _ = model(sequences, states_list_prev=None) 
            
            loss = criterion(logits, labels)
            loss.backward()
            # Optional: Gradient clipping if needed
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Store predictions and targets for accuracy calculation
            preds = torch.sigmoid(logits).round().detach()
            all_train_preds.append(preds.cpu())
            all_train_targets.append(labels.cpu())

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = accuracy_score(torch.cat(all_train_targets).numpy(), torch.cat(all_train_preds).numpy())
        train_f1 = f1_score(torch.cat(all_train_targets).numpy(), torch.cat(all_train_preds).numpy(), zero_division=0) # Added F1 score calculation

        # Validation phase
        model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_targets = []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                logits, _ = model(sequences, states_list_prev=None)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                
                preds = torch.sigmoid(logits).round().detach()
                all_val_preds.append(preds.cpu())
                all_val_targets.append(labels.cpu())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(torch.cat(all_val_targets).numpy(), torch.cat(all_val_preds).numpy())
        val_f1 = f1_score(torch.cat(all_val_targets).numpy(), torch.cat(all_val_preds).numpy(), zero_division=0) # Added F1 score calculation
        
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {epoch_duration:.2f}s | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Train F1: {train_f1:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")

        # Store history for plotting
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_accuracy)
        train_f1_history.append(train_f1) # Store train F1
        val_loss_history.append(avg_val_loss)
        val_acc_history.append(val_accuracy)
        val_f1_history.append(val_f1) # Store val F1

        # Early Stopping Logic - Enhanced Version
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_since_last_improvement = 0
            # Uncomment to save best model
            torch.save(model.state_dict(), 'best_rwkv_regex_model.pth')
            print(f"    New best validation accuracy: {best_val_accuracy:.4f}")
        else:
            epochs_since_last_improvement += 1

        # Perfect accuracy condition (exact match with regular language)
        if val_accuracy == 1.0 and train_accuracy >= EARLY_STOPPING_TRAIN_ACC_MIN:
            consecutive_perfect_epochs += 1
            if consecutive_perfect_epochs >= EARLY_STOPPING_PATIENCE_COUNT:
                print(f"Perfect accuracy (100%) achieved for {EARLY_STOPPING_PATIENCE_COUNT} "
                      f"consecutive epochs. RWKV has learned the regular language pattern perfectly.")
                break
        else:
            consecutive_perfect_epochs = 0  # Reset counter if accuracy drops
            
        # Near-perfect accuracy condition
        if val_accuracy >= VAL_ACC_THRESHOLD and train_accuracy >= EARLY_STOPPING_TRAIN_ACC_MIN:
            consecutive_near_perfect_epochs += 1
            if consecutive_near_perfect_epochs >= EARLY_STOPPING_PATIENCE_COUNT + 2:  # More patience for near-perfect
                print(f"Near-perfect accuracy (>{VAL_ACC_THRESHOLD*100}%) maintained for "
                      f"{EARLY_STOPPING_PATIENCE_COUNT+2} consecutive epochs.")
                break
        else:
            consecutive_near_perfect_epochs = 0
            
        # No improvement condition
        if epochs_since_last_improvement >= NO_IMPROVEMENT_PATIENCE:
            print(f"No improvement in validation accuracy for {NO_IMPROVEMENT_PATIENCE} epochs. "
                  f"Best accuracy achieved: {best_val_accuracy:.4f}")
            break
            
    print("\nTraining finished.")
    print(f"Best validation accuracy achieved: {best_val_accuracy:.4f}")

    # Plotting the results
    if train_loss_history: # Check if any history was recorded
        num_epochs_plotted = len(train_loss_history)
        epochs_x_axis = range(1, num_epochs_plotted + 1)

        plt.figure(figsize=(21, 6)) # Adjusted figure size for three subplots

        # Subplot 1: Loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs_x_axis, train_loss_history, label='Train Loss', marker='o')
        plt.plot(epochs_x_axis, val_loss_history, label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True)
        # Show integer ticks for epochs if not too many, otherwise let matplotlib decide
        if num_epochs_plotted <= 20: # Heuristic for readability
             plt.xticks(epochs_x_axis)


        # Subplot 2: Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(epochs_x_axis, train_acc_history, label='Train Accuracy', marker='o')
        plt.plot(epochs_x_axis, val_acc_history, label='Validation Accuracy', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.legend()
        plt.grid(True)
        if num_epochs_plotted <= 20: # Heuristic for readability
            plt.xticks(epochs_x_axis)

        # Subplot 3: F1 Score
        plt.subplot(1, 3, 3)
        plt.plot(epochs_x_axis, train_f1_history, label='Train F1 Score', marker='o')
        plt.plot(epochs_x_axis, val_f1_history, label='Validation F1 Score', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training & Validation F1 Score')
        plt.legend()
        plt.grid(True)
        if num_epochs_plotted <= 20: # Heuristic for readability
            plt.xticks(epochs_x_axis)

        plt.tight_layout() # Adjust layout to make space for suptitle
        plt.show()
    else:
        print("No training epochs were completed, skipping plotting.")

if __name__ == "__main__":
    train_and_evaluate()
