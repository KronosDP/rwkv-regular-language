import json

import torch

from config import (DATASET_FILE_CONFIG, MAX_LEN, MODEL_CHECKPOINT_PATH_CONFIG,
                    MODEL_HYPERPARAMETERS, PAD_TOKEN_CONFIG)
# Import the model definition
from rwkv_model import RWKV7_Model_Classifier
from utils import get_language_label  # MODIFIED - Import from utils

# --- Configuration ---
D_MODEL = MODEL_HYPERPARAMETERS["D_MODEL"]
N_LAYER = MODEL_HYPERPARAMETERS["N_LAYER"]
HEAD_SIZE = MODEL_HYPERPARAMETERS["HEAD_SIZE"]
FFN_HIDDEN_MULTIPLIER = MODEL_HYPERPARAMETERS["FFN_HIDDEN_MULTIPLIER"]
LORA_DIM_W = MODEL_HYPERPARAMETERS["LORA_DIM_W"]
LORA_DIM_A = MODEL_HYPERPARAMETERS["LORA_DIM_A"]
LORA_DIM_V = MODEL_HYPERPARAMETERS["LORA_DIM_V"]
LORA_DIM_G = MODEL_HYPERPARAMETERS["LORA_DIM_G"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Modified
DATASET_FILE = DATASET_FILE_CONFIG # Modified
MODEL_CHECKPOINT_PATH = MODEL_CHECKPOINT_PATH_CONFIG # Modified

# Global VOCAB, will be loaded from dataset file
VOCAB = {}
TARGET_SUBSTRING = "" # Will be loaded
PAD_TOKEN = PAD_TOKEN_CONFIG # Modified

# --- Language Checking Functions (for verification, same as in dataset_generator.py) ---

def preprocess_string(input_str, vocab_map, max_len_from_dataset=None):
    """
    Tokenizes and pads a single input string.
    If max_len_from_dataset is provided, it will pad to that length.
    Otherwise, it just tokenizes. For batch inference, consistent length is important.
    For single string inference, the model can handle variable length, but
    the collate_fn in training pads. Let's assume we don't need strict padding here
    unless a specific length is required by a non-dynamic model part.
    The current model handles dynamic length inputs.
    Empty strings are padded to length 1 with <pad> to match training.
    """
    if not input_str: # Handle empty string
        tokenized_input = [vocab_map.get(PAD_TOKEN, 0)]
    else:
        tokenized_input = [vocab_map.get(char, vocab_map.get(PAD_TOKEN, 0)) for char in input_str]

    # If max_len_from_dataset is given, pad or truncate
    if max_len_from_dataset is not None:
        if len(tokenized_input) < max_len_from_dataset:
            padding = [vocab_map.get(PAD_TOKEN, 0)] * (max_len_from_dataset - len(tokenized_input))
            tokenized_input.extend(padding)
        elif len(tokenized_input) > max_len_from_dataset:
            tokenized_input = tokenized_input[:max_len_from_dataset]
            
    return torch.tensor([tokenized_input], dtype=torch.long).to(DEVICE) # Batch size of 1

def _load_config_and_vocab():
    """Loads vocabulary and target substring from the dataset file."""
    global VOCAB, TARGET_SUBSTRING
    try:
        with open(DATASET_FILE, 'r') as f:
            dataset_obj = json.load(f)
        VOCAB = dataset_obj['vocab']
        vocab_size = len(VOCAB)
        TARGET_SUBSTRING = dataset_obj.get('target_substring', "abbccc")
        return vocab_size
    except FileNotFoundError:
        print(f"Error: {DATASET_FILE} not found. Please run dataset_generator.py first.")
        return None
    except Exception as e:
        print(f"Error loading dataset config: {e}")
        return None

def _initialize_model(vocab_size):
    """Initializes the RWKV7 model."""
    model = RWKV7_Model_Classifier(
        d_model=D_MODEL,
        n_layer=N_LAYER,
        vocab_size=vocab_size,
        head_size=HEAD_SIZE,
        ffn_hidden_multiplier=FFN_HIDDEN_MULTIPLIER,
        lora_dim_w=LORA_DIM_W, lora_dim_a=LORA_DIM_A,
        lora_dim_v=LORA_DIM_V, lora_dim_g=LORA_DIM_G
    ).to(DEVICE)
    return model

def _load_model_weights(model):
    """Loads trained model weights."""
    try:
        model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE))
        model.eval() # Set model to evaluation mode
        print(f"Model weights loaded from {MODEL_CHECKPOINT_PATH}")
        return True
    except FileNotFoundError:
        print(f"Error: Model checkpoint {MODEL_CHECKPOINT_PATH} not found. Make sure you have a trained model.")
        return False
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return False

def _inference_loop(model):
    """Handles the user input and model inference loop."""
    print("\n--- RWKV-7 Regex Inference ---")
    print("Enter a string to test, or type 'quit' to exit.")
    
    while True:
        try:
            input_str = input("Input string: ").strip()
        except EOFError:
            print("\nExiting...")
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if input_str.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if any(char not in VOCAB and char != PAD_TOKEN for char in input_str): # Check for unknown characters
            print(f"Warning: String contains characters not in the vocabulary: {list(VOCAB.keys())}. These will be treated as '{PAD_TOKEN}'.")

        # Preprocess the input string
        processed_input = preprocess_string(input_str, VOCAB, MAX_LEN) 

        with torch.no_grad():
            # Initial states for RWKV are None for a new sequence
            logits, _ = model(processed_input, states_list_prev=None) 
            probability = torch.sigmoid(logits.squeeze()) # Get probability for the single class
            prediction = (probability > 0.5).int().item()

        true_label = get_language_label(input_str, ('a', 'b'), TARGET_SUBSTRING) # MODIFIED - Use imported function
        
        print(f"  Input: '{input_str}'")
        print(f"  Model Prediction: {prediction} (Probability: {probability.item():.4f})")
        print(f"  True Label (for verification): {true_label}")
        if prediction == true_label:
            print("  Result: Correct!")
        else:
            print("  Result: Incorrect.")
        print("-" * 30)

def run_inference():
    print(f"Using device: {DEVICE}")

    # 1. Load Vocabulary and dataset config
    vocab_size = _load_config_and_vocab()
    if vocab_size is None:
        return

    # 2. Initialize Model
    model = _initialize_model(vocab_size)

    # 3. Load Trained Model Weights
    if not _load_model_weights(model):
        return

    # 4. Inference Loop
    _inference_loop(model)

if __name__ == "__main__":
    run_inference()
