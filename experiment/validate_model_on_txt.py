import json
import os

import torch
from sklearn.metrics import accuracy_score, f1_score  # Added import
from tqdm import tqdm  # Added import

import wandb
from config import MAX_LEN  # Added MODEL_CHECKPOINT_PATH_CONFIG import
from config import MODEL_CHECKPOINT_PATH_CONFIG, MODEL_HYPERPARAMETERS
from dataset_generator import NUM_SAMPLES, VOCAB
from rwkv_model import RWKV7_Model_Classifier
from utils import check_ab_star, check_contains_substring, get_language_label

# --- Configuration ---
MODEL_PATH = MODEL_CHECKPOINT_PATH_CONFIG  # Use path from config
VALIDATION_LENGTHS = [MAX_LEN, MAX_LEN + 10, MAX_LEN + 50, MAX_LEN * 10, MAX_LEN*100, MAX_LEN*1000]
VALIDATION_FILES = [f"validation_len{length}.txt" for length in VALIDATION_LENGTHS]
DATASET_INFO_FILE = "regex_dataset.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model artifact configuration - modify these to use different models
MODEL_ARTIFACT_NAME = None  # e.g., "rwkv_model_train_d64_l2_lr0.001_bs1536:latest"
USE_LOCAL_MODEL = True  # Set to False to use wandb artifact

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
    
    # Store all predictions for sklearn metrics
    y_true = []
    y_pred = []
    
    print("\nStarting validation...")
    for s in tqdm(test_strings, desc="Processing strings"):
        result = _process_single_string(model, s, current_vocab, model_input_max_len)
        if result is None or result[0] is None:
            continue
        
        is_correct, true_label, predicted_label = result
        y_true.append(true_label)
        y_pred.append(predicted_label)
        
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        _update_and_log_category_counts(s, is_correct, counts)
        
    print("\nValidation Complete!")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred) if y_true else 0
    f1 = f1_score(y_true, y_pred) if y_true else 0
    
    _print_final_accuracies(counts, total_predictions, correct_predictions)
    
    # Calculate category accuracies for wandb logging
    ab_star_acc = (counts['ab_star_correct'] / counts['ab_star_total']) if counts['ab_star_total'] > 0 else 0
    contains_acc = (counts['contains_abbccc_correct'] / counts['contains_abbccc_total']) if counts['contains_abbccc_total'] > 0 else 0
    neither_acc = (counts['neither_correct'] / counts['neither_total']) if counts['neither_total'] > 0 else 0
    
    return accuracy, f1, total_predictions, ab_star_acc, contains_acc, neither_acc, counts

def get_current_num_samples():
    """Read NUM_SAMPLES from the current config file"""
    try:
        # Read the config file to get the current NUM_SAMPLES value
        config_path = os.path.join(os.path.dirname(__file__), 'config.py')
        with open(config_path, 'r') as f:
            content = f.read()
        
        import re
        match = re.search(r'"NUM_SAMPLES":\s*(\d+)', content)
        if match:
            return int(match.group(1))
        else:
            # Fallback to default if not found
            print("Warning: NUM_SAMPLES not found in config, using default 1000")
            return 1000
    except Exception as e:
        print(f"Error reading NUM_SAMPLES from config: {e}")
        return 1000
    
def get_current_max_len():
    """Read MAX_LEN from the current config file"""
    try:
        # Read the config file to get the current MAX_LEN value
        config_path = os.path.join(os.path.dirname(__file__), 'config.py')
        with open(config_path, 'r') as f:
            content = f.read()
        
        import re
        match = re.search(r'MAX_LEN\s*=\s*(\d+)', content)
        if match:
            return int(match.group(1))
        else:
            # Fallback to default if not found
            print("Warning: MAX_LEN not found in config, using default 50")
            return 50
    except Exception as e:
        print(f"Error reading MAX_LEN from config: {e}")
        return 50


def _initialize_wandb_run(model_path):
    """Initialize wandb run for validation tracking"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    validation_run_name = f"validation_{model_path.replace('.pth', '')}_{timestamp}"
    
    # Get current NUM_SAMPLES dynamically
    current_num_samples = get_current_num_samples()
    current_max_len = get_current_max_len()
    
    run = wandb.init(project="rwkv-regex-learning", 
                     job_type="validation",
                     name=validation_run_name,
                     tags=["validation", "model_evaluation", f"model_{model_path.replace('.pth', '')}", timestamp],
                     config={
                         "model_path": model_path,
                         "validation_lengths": VALIDATION_LENGTHS,
                         "alphabet_chars_ab": ALPHABET_CHARS_AB,
                         "target_substring": TARGET_SUBSTRING,
                         "model_artifact_name": MODEL_ARTIFACT_NAME,
                         "use_local_model": USE_LOCAL_MODEL,
                         "model_hyperparameters": MODEL_HYPERPARAMETERS,
                         "dataset": {
                             "num_samples": current_num_samples,  # Use dynamic value
                             "max_len": current_max_len
                         }
                     })
    return run

def _setup_model_path(run, model_path):
    """Setup model path, handling both local and artifact models"""
    # Try to link to training run if using local model
    if USE_LOCAL_MODEL and MODEL_ARTIFACT_NAME is None:
        try:
            api = wandb.Api()
            training_runs = api.runs("rwkv-regex-learning", filters={"config.job_type": "training"})
            if training_runs:
                latest_training_run = training_runs[0]
                run.config.update({"linked_training_run": latest_training_run.id})
                print(f"✓ Linked to training run: {latest_training_run.name}")
        except Exception as e:
            print(f"Could not link to training run: {e}")
    
    # Use model artifact if specified
    if not USE_LOCAL_MODEL and MODEL_ARTIFACT_NAME:
        try:
            model_artifact = run.use_artifact(f"rwkv-regex-learning/{MODEL_ARTIFACT_NAME}")
            artifact_dir = model_artifact.download()
            model_path = os.path.join(artifact_dir, model_path)
            print(f"✓ Using model from artifact: {MODEL_ARTIFACT_NAME}")
        except Exception as e:
            print(f"Failed to load model artifact: {e}")
            return None
    
    return model_path

def _process_validation_files(model, current_vocab, model_input_max_len):
    """Process all validation files and return results summary"""
    results_summary = []
    
    for i, (validation_file, validation_length) in enumerate(zip(VALIDATION_FILES, VALIDATION_LENGTHS)):
        print(f"\n{'='*60}")
        print(f"VALIDATION SET {i+1}: {validation_file} (Max Length: {validation_length})")
        print(f"{'='*60}")
        
        try:
            with open(validation_file, 'r') as f:
                test_strings = [line.strip() for line in f if line.strip()]
            if not test_strings:
                print(f"No strings found in {validation_file}.")
                continue
            print(f"Read {len(test_strings)} strings from {validation_file}")
        except FileNotFoundError:
            print(f"Error: {validation_file} not found. Please generate it first.")
            continue

        accuracy, f1, total_samples, ab_star_acc, contains_acc, neither_acc, counts = _evaluate_model(
            model, test_strings, current_vocab, model_input_max_len)
        
        # Calculate generalization metric (how much longer than training MAX_LEN)
        generalization_factor = validation_length / MAX_LEN
        is_generalization = validation_length > MAX_LEN
        
        # Store results
        result = {
            'validation_file': validation_file,
            'max_length': validation_length,
            'accuracy': accuracy,
            'f1_score': f1,
            'total_samples': total_samples,
            'ab_star_accuracy': ab_star_acc,
            'contains_accuracy': contains_acc,
            'neither_accuracy': neither_acc,
            'generalization_factor': generalization_factor,
            'is_generalization': is_generalization
        }
        results_summary.append(result)
        
        # Log to wandb with proper structure for plotting
        log_data = {
            f"validation_len_{validation_length}/accuracy": accuracy,
            f"validation_len_{validation_length}/f1_score": f1,
            f"validation_len_{validation_length}/total_samples": total_samples,
            f"validation_len_{validation_length}/ab_star_accuracy": ab_star_acc,
            f"validation_len_{validation_length}/contains_abbccc_accuracy": contains_acc,
            f"validation_len_{validation_length}/neither_accuracy": neither_acc,
            f"validation_len_{validation_length}/generalization_factor": generalization_factor,
            f"validation_len_{validation_length}/ab_star_total": counts['ab_star_total'],
            f"validation_len_{validation_length}/contains_abbccc_total": counts['contains_abbccc_total'],
            f"validation_len_{validation_length}/neither_total": counts['neither_total']
        }
          # Add data for length-based plotting with proper x-axis
        log_data.update({
            "performance_vs_length/validation_length": validation_length,
            "performance_vs_length/accuracy": accuracy,
            "performance_vs_length/f1_score": f1,
            "performance_vs_length/generalization_factor": generalization_factor,
            "performance_vs_length/ab_star_accuracy": ab_star_acc,
            "performance_vs_length/contains_accuracy": contains_acc,
            "performance_vs_length/neither_accuracy": neither_acc
        })
        
        wandb.log(log_data)
        
        print(f"\nSUMMARY FOR {validation_file}:")
        print(f"  Length: {validation_length} (Generalization Factor: {generalization_factor:.1f}x)")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Total Samples: {total_samples}")
        print(f"  (ab)* Accuracy: {ab_star_acc:.4f} ({ab_star_acc*100:.2f}%)")
        print(f"  Contains 'abbccc' Accuracy: {contains_acc:.4f} ({contains_acc*100:.2f}%)")
        print(f"  Neither Accuracy: {neither_acc:.4f} ({neither_acc*100:.2f}%)")
    
    return results_summary

def _calculate_summary_metrics(results_summary):
    """Calculate comprehensive summary metrics for model comparison"""
    # Separate baseline vs generalization performance
    baseline_results = [r for r in results_summary if r['max_length'] == MAX_LEN]
    generalization_results = [r for r in results_summary if r['max_length'] > MAX_LEN]
    
    # Calculate baseline performance
    baseline_accuracy = baseline_results[0]['accuracy'] if baseline_results else 0
    baseline_f1 = baseline_results[0]['f1_score'] if baseline_results else 0
    
    # Calculate generalization metrics
    if generalization_results:
        avg_generalization_accuracy = sum(r['accuracy'] for r in generalization_results) / len(generalization_results)
        avg_generalization_f1 = sum(r['f1_score'] for r in generalization_results) / len(generalization_results)
        accuracy_degradation = baseline_accuracy - avg_generalization_accuracy
        f1_degradation = baseline_f1 - avg_generalization_f1
        
        longest_result = max(generalization_results, key=lambda x: x['max_length'])
        ultimate_generalization_accuracy = longest_result['accuracy']
        ultimate_generalization_f1 = longest_result['f1_score']
        ultimate_length = longest_result['max_length']
        avg_gen_factor = sum(r['generalization_factor'] for r in generalization_results) / len(generalization_results)
    else:
        avg_generalization_accuracy = baseline_accuracy
        avg_generalization_f1 = baseline_f1
        accuracy_degradation = 0
        f1_degradation = 0
        ultimate_generalization_accuracy = baseline_accuracy
        ultimate_generalization_f1 = baseline_f1
        ultimate_length = MAX_LEN
        avg_gen_factor = 1.0
    
    # Overall metrics
    avg_accuracy = sum(r['accuracy'] for r in results_summary) / len(results_summary)
    avg_f1 = sum(r['f1_score'] for r in results_summary) / len(results_summary)
    total_samples_all = sum(r['total_samples'] for r in results_summary)
    
    # Calculate robustness and efficiency
    accuracy_variance = sum((r['accuracy'] - avg_accuracy) ** 2 for r in results_summary) / len(results_summary)
    robustness_score = 1.0 / (1.0 + accuracy_variance)
    generalization_efficiency = avg_generalization_accuracy / avg_gen_factor if avg_gen_factor > 0 else baseline_accuracy
    
    return {
        'baseline_accuracy': baseline_accuracy,
        'baseline_f1': baseline_f1,
        'avg_generalization_accuracy': avg_generalization_accuracy,
        'avg_generalization_f1': avg_generalization_f1,
        'accuracy_degradation': accuracy_degradation,
        'f1_degradation': f1_degradation,
        'ultimate_generalization_accuracy': ultimate_generalization_accuracy,
        'ultimate_generalization_f1': ultimate_generalization_f1,
        'ultimate_length': ultimate_length,
        'avg_accuracy': avg_accuracy,
        'avg_f1': avg_f1,
        'total_samples_all': total_samples_all,
        'robustness_score': robustness_score,
        'generalization_efficiency': generalization_efficiency,
        'avg_gen_factor': avg_gen_factor,
        'num_generalization_tests': len(generalization_results)
    }

def _print_summary_and_log_metrics(results_summary):
    """Print overall summary and log final metrics to wandb"""
    print("\n" + "="*60)
    print("OVERALL VALIDATION SUMMARY")
    print("="*60)
    print(f"{'Validation File':<30} {'Max Length':<12} {'Accuracy':<10} {'F1 Score':<10} {'Samples':<8}")
    print("-" * 70)
    for result in results_summary:
        print(f"{result['validation_file']:<30} {result['max_length']:<12} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} {result['total_samples']:<8}")
    
    if not results_summary:
        return
    
    metrics = _calculate_summary_metrics(results_summary)
    
    # Create wandb table
    table_data = []
    for result in results_summary:
        table_data.append([
            result['max_length'], result['generalization_factor'], result['accuracy'],
            result['f1_score'], result['ab_star_accuracy'], result['contains_accuracy'],
            result['neither_accuracy'], result['total_samples'],
            "Baseline" if result['max_length'] == MAX_LEN else "Generalization"
        ])
    
    validation_table = wandb.Table(
        columns=["Length", "Gen_Factor", "Accuracy", "F1", "AB_Star_Acc", "Contains_Acc", "Neither_Acc", "Samples", "Type"],
        data=table_data
    )
    
    # Log comprehensive summary metrics
    summary_metrics = {
        # 🎯 PRIMARY COMPARISON METRICS
        "🎯_PRIMARY/GENERALIZATION_SCORE": metrics['avg_generalization_accuracy'],
        "🎯_PRIMARY/BASELINE_ACCURACY": metrics['baseline_accuracy'],
        "🎯_PRIMARY/ULTIMATE_GENERALIZATION": metrics['ultimate_generalization_accuracy'],
        "🎯_PRIMARY/ROBUSTNESS_SCORE": metrics['robustness_score'],
        "🎯_PRIMARY/GENERALIZATION_EFFICIENCY": metrics['generalization_efficiency'],
        
        # 📊 PERFORMANCE BREAKDOWN
        "📊_PERFORMANCE/baseline_accuracy": metrics['baseline_accuracy'],
        "📊_PERFORMANCE/baseline_f1": metrics['baseline_f1'],
        "📊_PERFORMANCE/avg_generalization_accuracy": metrics['avg_generalization_accuracy'],
        "📊_PERFORMANCE/avg_generalization_f1": metrics['avg_generalization_f1'],
        "📊_PERFORMANCE/accuracy_degradation": metrics['accuracy_degradation'],
        "📊_PERFORMANCE/f1_degradation": metrics['f1_degradation'],
        
        # 🚀 GENERALIZATION ANALYSIS
        "🚀_GENERALIZATION/ultimate_accuracy": metrics['ultimate_generalization_accuracy'],
        "🚀_GENERALIZATION/ultimate_f1": metrics['ultimate_generalization_f1'],
        "🚀_GENERALIZATION/ultimate_length": metrics['ultimate_length'],
        "🚀_GENERALIZATION/ultimate_factor": metrics['ultimate_length'] / MAX_LEN,
        "🚀_GENERALIZATION/num_gen_tests": metrics['num_generalization_tests'],
        "🚀_GENERALIZATION/avg_gen_factor": metrics['avg_gen_factor'],
        
        # 📈 OVERALL STATS
        "📈_OVERALL/avg_accuracy_all": metrics['avg_accuracy'],
        "📈_OVERALL/avg_f1_all": metrics['avg_f1'],
        "📈_OVERALL/total_samples": metrics['total_samples_all'],
        "📈_OVERALL/num_validation_sets": len(results_summary),
        "📈_OVERALL/training_max_len": MAX_LEN,
        
        "summary/validation_results_table": validation_table
    }
    
    wandb.log(summary_metrics)
    
    # Print key metrics
    print("\n" + "="*60)
    print("🎯 KEY METRICS FOR MODEL COMPARISON")
    print("="*60)
    print(f"📊 BASELINE PERFORMANCE (Length={MAX_LEN}):")
    print(f"  ├─ Accuracy: {metrics['baseline_accuracy']:.4f} ({metrics['baseline_accuracy']*100:.2f}%)")
    print(f"  └─ F1 Score: {metrics['baseline_f1']:.4f}")
    print("\n🚀 GENERALIZATION PERFORMANCE:")
    print(f"  ├─ Avg Accuracy: {metrics['avg_generalization_accuracy']:.4f} ({metrics['avg_generalization_accuracy']*100:.2f}%)")
    print(f"  ├─ Accuracy Drop: {metrics['accuracy_degradation']:.4f} ({metrics['accuracy_degradation']*100:.2f}% points)")
    print(f"  └─ Generalization Score: {metrics['avg_generalization_accuracy'] * (1 - metrics['accuracy_degradation']):.4f}")
    print(f"\n⭐ ULTIMATE GENERALIZATION (Length={metrics['ultimate_length']}, {metrics['ultimate_length']/MAX_LEN:.1f}x):")
    print(f"  ├─ Accuracy: {metrics['ultimate_generalization_accuracy']:.4f} ({metrics['ultimate_generalization_accuracy']*100:.2f}%)")
    print(f"  └─ F1 Score: {metrics['ultimate_generalization_f1']:.4f}")
    print("\n📈 OVERALL SUMMARY:")
    print(f"  ├─ Tested {len(results_summary)} length configurations")
    print(f"  ├─ {metrics['num_generalization_tests']} generalization tests")
    print(f"  └─ {metrics['total_samples_all']} total validation samples")
    
    print(f"\nValidation completed on {len(results_summary)} datasets.")
    print("💡 Use the '🎯_PRIMARY/GENERALIZATION_SCORE' metric to compare models!")
    for result in results_summary:
        print(f"{result['validation_file']:<30} {result['max_length']:<12} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} {result['total_samples']:<8}")
    
    # Calculate key metrics for model comparison
    summary_metrics = _calculate_summary_metrics(results_summary)
    
    # Log comprehensive summary metrics with clear grouping
    final_log_data = {
        # 🎯 PRIMARY COMPARISON METRICS (use these to rank models)
        "🎯_PRIMARY/GENERALIZATION_SCORE": summary_metrics['avg_generalization_accuracy'],
        "🎯_PRIMARY/BASELINE_ACCURACY": summary_metrics['baseline_accuracy'], 
        "🎯_PRIMARY/ULTIMATE_GENERALIZATION": summary_metrics['ultimate_generalization_accuracy'],
        "🎯_PRIMARY/ROBUSTNESS_SCORE": summary_metrics['robustness_score'],
        "🎯_PRIMARY/GENERALIZATION_EFFICIENCY": summary_metrics['generalization_efficiency'],
        
        # 📊 PERFORMANCE_BREAKDOWN
        "📊_PERFORMANCE/baseline_accuracy": summary_metrics['baseline_accuracy'],
        "📊_PERFORMANCE/baseline_f1": summary_metrics['baseline_f1'],
        "📊_PERFORMANCE/avg_generalization_accuracy": summary_metrics['avg_generalization_accuracy'],
        "📊_PERFORMANCE/avg_generalization_f1": summary_metrics['avg_generalization_f1'],
        "📊_PERFORMANCE/accuracy_degradation": summary_metrics['accuracy_degradation'],
        "📊_PERFORMANCE/f1_degradation": summary_metrics['f1_degradation'],
        
        # 🚀 GENERALIZATION_ANALYSIS
        "🚀_GENERALIZATION/ultimate_accuracy": summary_metrics['ultimate_generalization_accuracy'],
        "🚀_GENERALIZATION/ultimate_f1": summary_metrics['ultimate_generalization_f1'],
        "🚀_GENERALIZATION/ultimate_length": summary_metrics['ultimate_length'],
        "🚀_GENERALIZATION/ultimate_factor": summary_metrics['ultimate_length'] / MAX_LEN,
        "🚀_GENERALIZATION/num_gen_tests": summary_metrics['num_generalization_tests'],
        "🚀_GENERALIZATION/avg_gen_factor": summary_metrics['avg_gen_factor'],
        
        # 📈 OVERALL_STATS
        "📈_OVERALL/avg_accuracy_all": summary_metrics['avg_accuracy'],
        "📈_OVERALL/avg_f1_all": summary_metrics['avg_f1'],
        "📈_OVERALL/total_samples": summary_metrics['total_samples_all'],
        "📈_OVERALL/num_validation_sets": len(results_summary),
        "📈_OVERALL/training_max_len": MAX_LEN
    }
    
    # Create a table for wandb
    table_data = []
    for result in results_summary:
        table_data.append([
            result['max_length'],
            result['generalization_factor'],
            result['accuracy'],
            result['f1_score'],
            result['ab_star_accuracy'],
            result['contains_accuracy'],
            result['neither_accuracy'],
            result['total_samples'],
            "Baseline" if result['max_length'] == MAX_LEN else "Generalization"
        ])
    
    validation_table = wandb.Table(
        columns=["Length", "Gen_Factor", "Accuracy", "F1", "AB_Star_Acc", "Contains_Acc", "Neither_Acc", "Samples", "Type"],
        data=table_data
    )
    
    final_log_data["summary/validation_results_table"] = validation_table
    
    # Create custom plots - removed unused variables and fixed for proper generalization visualization
    wandb.log(final_log_data)
    
    # Print key metrics for easy comparison
    print("\n" + "="*60)
    print("🎯 KEY METRICS FOR MODEL COMPARISON")
    print("="*60)
    print(f"📊 BASELINE PERFORMANCE (Length={MAX_LEN}):")
    print(f"  ├─ Accuracy: {summary_metrics['baseline_accuracy']:.4f} ({summary_metrics['baseline_accuracy']*100:.2f}%)")
    print(f"  └─ F1 Score: {summary_metrics['baseline_f1']:.4f}")
    print("\n🚀 GENERALIZATION PERFORMANCE:")
    print(f"  ├─ Avg Accuracy: {summary_metrics['avg_generalization_accuracy']:.4f} ({summary_metrics['avg_generalization_accuracy']*100:.2f}%)")
    print(f"  ├─ Accuracy Drop: {summary_metrics['accuracy_degradation']:.4f} ({summary_metrics['accuracy_degradation']*100:.2f}% points)")
    print(f"  └─ Generalization Score: {summary_metrics['avg_generalization_accuracy'] * (1 - summary_metrics['accuracy_degradation']):.4f}")
    print(f"\n⭐ ULTIMATE GENERALIZATION (Length={summary_metrics['ultimate_length']}, {summary_metrics['ultimate_length']/MAX_LEN:.1f}x):")
    print(f"  ├─ Accuracy: {summary_metrics['ultimate_generalization_accuracy']:.4f} ({summary_metrics['ultimate_generalization_accuracy']*100:.2f}%)")
    print(f"  └─ F1 Score: {summary_metrics['ultimate_generalization_f1']:.4f}")
    print(f"\n📈 OVERALL SUMMARY:")
    print(f"  ├─ Tested {len(results_summary)} length configurations")
    print(f"  ├─ {summary_metrics['num_generalization_tests']} generalization tests")
    print(f"  └─ {summary_metrics['total_samples_all']} total validation samples")
    
    print(f"\nValidation completed on {len(results_summary)} datasets.")
    print("💡 Use the '🎯_PRIMARY/GENERALIZATION_SCORE' metric to compare models!")

def main():
    print(f"Using device: {DEVICE}")
    current_vocab, vocab_size, model_input_max_len = _load_config_and_vocab()
    if current_vocab is None: 
        return

    model_params = MODEL_HYPERPARAMETERS
    print(f"Model hyperparameters: {model_params}")
    print(f"Model expected input max_len: {model_input_max_len}")
    
    model_path = MODEL_PATH
    run = _initialize_wandb_run(model_path)
    
    model_path = _setup_model_path(run, model_path)
    if model_path is None:
        return

    model = load_model_for_inference(
        model_path, vocab_size,
        model_params["D_MODEL"], model_params["N_LAYER"],
        model_params["HEAD_SIZE"], model_params["FFN_HIDDEN_MULTIPLIER"],
        model_params["LORA_DIM_W"], model_params["LORA_DIM_A"],
        model_params["LORA_DIM_V"], model_params["LORA_DIM_G"]
    )
    if model is None: 
        return

    results_summary = _process_validation_files(model, current_vocab, model_input_max_len)
    _print_summary_and_log_metrics(results_summary)
    
    wandb.finish()


if __name__ == "__main__":
    main()
