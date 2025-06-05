import json

import torch
from tqdm import tqdm  # Added import

from config import MODEL_HYPERPARAMETERS  # Added import
from dataset_generator import VOCAB
from rwkv_model import RWKV7_Model_Classifier
from utils import check_ab_star, check_contains_substring, get_language_label

# --- Configuration ---
MODEL_PATH = "best_rwkv_regex_model.pth"
VALIDATION_FILE = "validation.txt"
DATASET_INFO_FILE = "regex_dataset.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALPHABET_CHARS_AB = ['a', 'b'] # This can be removed if VOCAB is used directly or passed
TARGET_SUBSTRING = "abbccc" # This should ideally be loaded or passed if it can vary

# --- Model Loading and Inference ---
def load_model_for_inference(model_path, vocab_size, d_model, n_layer, head_size, ffn_hidden_multiplier, 
                             lora_dim_w, lora_dim_a, lora_dim_v, lora_dim_g):
    model = RWKV7_Model_Classifier(
        d_model=d_model, n_layer=n_layer, vocab_size=vocab_size,
        head_size=head_size, ffn_hidden_multiplier=ffn_hidden_multiplier,
        lora_dim_w=lora_dim_w, lora_dim_a=lora_dim_a,
        lora_dim_v=lora_dim_v, lora_dim_g=lora_dim_g
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}.")
        return None
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        return None
    model.to(DEVICE)
    model.eval()
    return model

def string_to_tensor(s, vocab, max_len):
    token_ids = [vocab.get(char, vocab.get('<unk>', 0)) for char in s]
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    padded_token_ids = token_ids + [vocab.get('<pad>', 0)] * (max_len - len(token_ids))
    return torch.tensor(padded_token_ids, dtype=torch.long).unsqueeze(0)

def predict(model, text_string, vocab, max_len_for_model_input):
    if model is None: return None, None
    with torch.no_grad():
        input_tensor = string_to_tensor(text_string, vocab, max_len_for_model_input).to(DEVICE)
        # model() likely returns (logits, states), we only need logits
        output = model(input_tensor, states_list_prev=None) 
        logits = output[0] if isinstance(output, tuple) else output # Get the first element if it's a tuple
        prob = torch.sigmoid(logits).item()
        prediction = 1 if prob > 0.5 else 0
    return prediction, prob

# --- Helper Functions for Main Logic ---
def _load_config_and_vocab():
    current_vocab = VOCAB
    vocab_size = len(current_vocab)
    if not current_vocab or '<pad>' not in current_vocab:
        print("Error: VOCAB not loaded or missing '<pad>'.")
        return None, None, None
    print(f"Using vocabulary: {current_vocab}")

    try:
        with open(DATASET_INFO_FILE, 'r') as f:
            dataset_info = json.load(f)
            model_input_max_len = dataset_info.get('max_len')
            if model_input_max_len is None:
                print(f"Error: 'max_len' not found in {DATASET_INFO_FILE}.")
                return None, None, None
            print(f"Loaded 'max_len' from {DATASET_INFO_FILE}: {model_input_max_len}")
    except FileNotFoundError:
        print(f"Error: {DATASET_INFO_FILE} not found.")
        return None, None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {DATASET_INFO_FILE}.")
        return None, None, None
    return current_vocab, vocab_size, model_input_max_len

def _process_single_string(model, s, current_vocab, model_input_max_len):
    true_label = get_language_label(s, ('a','b'), TARGET_SUBSTRING) # MODIFIED
    predicted_label, _ = predict(model, s, current_vocab, model_input_max_len) # probability is unused
    if predicted_label is None:
        return None, None # Error in prediction
    return predicted_label == true_label, true_label, predicted_label

def _update_and_log_category_counts(s, is_correct, counts):
    is_ab_star_type = check_ab_star(s, ('a','b')) # MODIFIED
    is_contains_type = check_contains_substring(s, TARGET_SUBSTRING) and not is_ab_star_type # MODIFIED
    is_neither_type = not is_ab_star_type and not check_contains_substring(s, TARGET_SUBSTRING) # MODIFIED

    if is_ab_star_type:
        counts['ab_star_total'] += 1
        if is_correct: counts['ab_star_correct'] += 1
    elif is_contains_type:
        counts['contains_abbccc_total'] += 1
        if is_correct: counts['contains_abbccc_correct'] += 1
    elif is_neither_type:
        counts['neither_total'] += 1
        if is_correct: counts['neither_correct'] += 1

def _print_final_accuracies(counts, total_predictions, correct_predictions):
    if total_predictions > 0:
        overall_accuracy = (correct_predictions / total_predictions) * 100
        print(f"Overall Model Accuracy: {correct_predictions}/{total_predictions} = {overall_accuracy:.2f}%")
        _print_category_accuracy("(ab)* strings", counts['ab_star_correct'], counts['ab_star_total'])
        _print_category_accuracy("'contains abbccc' strings", counts['contains_abbccc_correct'], counts['contains_abbccc_total'])
        _print_category_accuracy("'neither' strings", counts['neither_correct'], counts['neither_total'])
    else:
        print("No predictions were made.")

def _print_category_accuracy(category_name, correct, total):
    if total > 0:
        acc = (correct / total) * 100
        print(f"  Accuracy for {category_name}: {correct}/{total} = {acc:.2f}%")
    else:
        print(f"  No {category_name} were tested.")

def _evaluate_model(model, test_strings, current_vocab, model_input_max_len):
    correct_predictions = 0
    total_predictions = 0
    counts = {
        'ab_star_total': 0, 'ab_star_correct': 0,
        'contains_abbccc_total': 0, 'contains_abbccc_correct': 0,
        'neither_total': 0, 'neither_correct': 0
    }
    print("\nStarting validation...")
    for s in tqdm(test_strings, desc="Processing strings"): # MODIFIED
        result = _process_single_string(model, s, current_vocab, model_input_max_len)
        if result is None or result[0] is None: # Indicates an error during prediction
            # tqdm will show progress, so specific skip message might be less critical or could be logged differently
            # print(f"Skipping string due to prediction error.") # Optional: keep if detailed per-string error is needed
            continue
        
        is_correct, _, _ = result
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        _update_and_log_category_counts(s, is_correct, counts)
        
        # Removed manual progress printing:
        # if (i + 1) % 50 == 0 or (i + 1) == len(test_strings):
        #     print(f"  Processed {i+1}/{len(test_strings)} strings...")
    print("\nValidation Complete!")
    _print_final_accuracies(counts, total_predictions, correct_predictions)


def main():
    print(f"Using device: {DEVICE}")
    current_vocab, vocab_size, model_input_max_len = _load_config_and_vocab()
    if current_vocab is None: return

    model_params = MODEL_HYPERPARAMETERS # Use imported hyperparameters
    print(f"Model hyperparameters: {model_params}")
    print(f"Model expected input max_len: {model_input_max_len}")

    model = load_model_for_inference(
        MODEL_PATH, vocab_size,
        model_params["D_MODEL"], model_params["N_LAYER"],
        model_params["HEAD_SIZE"], model_params["FFN_HIDDEN_MULTIPLIER"],
        model_params["LORA_DIM_W"], model_params["LORA_DIM_A"],
        model_params["LORA_DIM_V"], model_params["LORA_DIM_G"]
    )
    if model is None: return

    try:
        with open(VALIDATION_FILE, 'r') as f:
            test_strings = [line.strip() for line in f if line.strip()]
        if not test_strings:
            print(f"No strings found in {VALIDATION_FILE}.")
            return
        print(f"Read {len(test_strings)} strings from {VALIDATION_FILE}")
    except FileNotFoundError:
        print(f"Error: {VALIDATION_FILE} not found. Please generate it first.")
        return

    _evaluate_model(model, test_strings, current_vocab, model_input_max_len)

if __name__ == "__main__":
    main()
