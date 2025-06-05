import json
import random

import wandb

# --- Configuration ---w
VOCAB = {'<pad>': 0, 'a': 1, 'b': 2, 'c': 3}
INV_VOCAB = {v: k for k, v in VOCAB.items()}
ALPHABET_CHARS_AB = ['a', 'b']
ALPHABET_CHARS_ABC = ['a', 'b', 'c']
TARGET_SUBSTRING = "abbccc"
NUM_SAMPLES = 3_000
MAX_LEN = 50

# --- Language Checking Functions ---
from utils import (check_ab_star, check_contains_substring,
                   generate_random_string, get_language_label)

# --- Helper Functions for Dataset Generation ---

def _create_ab_star_sequences(max_len):
    """Creates a list of valid (ab)* sequences up to max_len."""
    if max_len < 0: return [""] # Should not happen with typical max_len >= 0
    return [("ab" * k)[:max_len] for k in range((max_len // 2) + 2)] # +2 to include empty and full length

def _gen_ab_star_unique(ab_star_sequences):
    """Picks a random (ab)* sequence from pre-generated list."""
    if not ab_star_sequences: return ""
    return random.choice(ab_star_sequences)

def _gen_contains(max_len):
    """Generates a string containing TARGET_SUBSTRING."""
    if len(TARGET_SUBSTRING) > max_len:
        if max_len < 0: return ""
        prefix_len = random.randint(0, max(0, max_len - len(TARGET_SUBSTRING)))
    else:
        prefix_len = random.randint(0, max_len - len(TARGET_SUBSTRING))

    suffix_len = max(0, max_len - len(TARGET_SUBSTRING) - prefix_len)
    return (generate_random_string(prefix_len, ALPHABET_CHARS_ABC)
            + TARGET_SUBSTRING
            + generate_random_string(suffix_len, ALPHABET_CHARS_ABC))[:max_len]


def _gen_random_string_candidate(max_len, alphabet):
    """Generates a random string from the given alphabet."""
    # Ensure length is at least 0, even if max_len is negative (though not expected)
    return generate_random_string(random.randint(0, max(0,max_len)), alphabet)


# --- Mutation Helpers for (ab)* Near Misses ---
def _mutate_s_list_char_swap_ab(s_list_param):
    s_list = list(s_list_param)
    if len(s_list) < 2: return s_list
    for i in range(len(s_list) - 1):
        if s_list[i] == 'a' and s_list[i+1] == 'b':
            s_list[i], s_list[i+1] = 'b', 'a'
            return s_list
    idx = random.randrange(len(s_list) - 1)
    s_list[idx], s_list[idx+1] = s_list[idx+1], s_list[idx]
    return s_list

def _mutate_s_list_char_change_to_c(s_list_param):
    s_list = list(s_list_param)
    if not s_list: return s_list
    idx = random.randrange(len(s_list))
    s_list[idx] = 'c'
    return s_list

def _mutate_s_list_char_change_within_ab(s_list_param):
    s_list = list(s_list_param)
    if not s_list: return s_list
    idx = random.randrange(len(s_list))
    if s_list[idx] == 'a': s_list[idx] = 'b'
    elif s_list[idx] == 'b': s_list[idx] = 'a'
    return s_list

def _mutate_s_list_insert_c(s_list_param, max_len):
    s_list = list(s_list_param)
    if len(s_list) >= max_len: return s_list # Cannot insert if already at max_len
    idx = random.randint(0, len(s_list))
    s_list.insert(idx, 'c')
    return s_list

def _mutate_s_list_make_odd_len(s_list_param, max_len):
    s_list = list(s_list_param)
    if len(s_list) >= max_len or not s_list : return s_list # Cannot append or s_list is empty
    s_list.append(random.choice(ALPHABET_CHARS_AB))
    return s_list

def _mutate_s_list_remove_to_make_invalid_pair(s_list_param):
    s_list = list(s_list_param)
    if len(s_list) <= 1: return s_list # Cannot remove meaningfully or list too short
    s_list.pop(random.randrange(len(s_list)))
    return s_list

def _apply_ab_star_mutation(s_list_orig, max_len):
    s_list = list(s_list_orig) # Work on a copy

    mutations = {
        'char_swap_ab': lambda sl: _mutate_s_list_char_swap_ab(sl),
        'char_change_to_c': lambda sl: _mutate_s_list_char_change_to_c(sl),
        'char_change_within_ab': lambda sl: _mutate_s_list_char_change_within_ab(sl),
        'insert_c': lambda sl: _mutate_s_list_insert_c(sl, max_len),
        'make_odd_len': lambda sl: _mutate_s_list_make_odd_len(sl, max_len),
        'remove_to_make_invalid_pair': lambda sl: _mutate_s_list_remove_to_make_invalid_pair(sl),
    }
    
    # All mutations can technically be chosen; they handle non-applicability internally.
    mutation_key = random.choice(list(mutations.keys()))
    mutated_s_list = mutations[mutation_key](s_list)
    
    return "".join(mutated_s_list)[:max_len]

def _gen_ab_star_specific_near_miss(max_len):
    """Generates strings that are specifically near misses to (ab)* pattern."""
    # Strategy 1: Mutate a valid (ab)* string or a component
    if random.random() < 0.8 and max_len > 0:
        base_len = random.randint(0, (max_len // 2) + 1)
        s_list = list("ab" * base_len)
        if not s_list and max_len > 0: # if base_len was 0, start with a single char or two to mutate
            s_list = list(random.choice(["a", "b", "aa", "bb", "ba"]))
        
        if not s_list: # Still empty if max_len was 0 or very small initial choice
             return generate_random_string(random.randint(1, max(1, max_len)), ALPHABET_CHARS_ABC)

        s = _apply_ab_star_mutation(s_list, max_len)
    # Strategy 2: Generate strings with only 'a's or only 'b's or mixed a/b not in (ab)* form
    else:
        length = random.randint(1, max(1, max_len)) # Ensure length is at least 1
        if random.random() < 0.33:
            s = 'a' * length
        elif random.random() < 0.5:
            s = 'b' * length
        else: # random a's and b's, less likely to be (ab)*
            s = "".join(random.choice(ALPHABET_CHARS_AB) for _ in range(length))
    return s[:max_len] # Ensure final string respects max_len

def _gen_near_miss_general(max_len):
    """Generates general near-miss strings, including (ba)* patterns."""
    if random.random() < 0.7:
        if max_len == 0: return ""
        length = random.randint(1, max(1, max_len))
        s_chars = ['b' if i % 2 == 0 else 'a' for i in range(length)]
        s = "".join(s_chars)
    else:
        s = generate_random_string(random.randint(1, max(1, max_len)), ALPHABET_CHARS_ABC)
    return s[:max_len]

# --- Sampling Logic ---
def _sample_category(generator_fn, target_count, expected_label, seen_set, allow_duplicates=False, max_attempts_multiplier=20):
    """
    General sampler for categories: call generator_fn to get candidate string,
    check label and uniqueness, and collect until target_count reached.
    """
    collected, attempts = [], 0
    # Ensure target_count is non-negative
    if target_count <= 0:
        return []
        
    max_total_attempts = target_count * max_attempts_multiplier
    
    while len(collected) < target_count and attempts < max_total_attempts:
        attempts += 1
        s = generator_fn()
        if get_language_label(s, tuple(ALPHABET_CHARS_AB), TARGET_SUBSTRING) == expected_label and (allow_duplicates or s not in seen_set):
            collected.append(s)
            if not allow_duplicates:
                seen_set.add(s)
    return collected

# --- Main Dataset Generation Function ---
def generate_dataset(num_samples, max_len):
    half = num_samples // 2
    pos_target, neg_target = half, num_samples - half

    pos_breakdown = {
        'ab_star': int(num_samples * 0.20),
        'contains': int(num_samples * 0.15),
    }
    neg_breakdown = {
        'near_miss_general': int(num_samples * 0.15),
        'near_miss_ab_star_specific': int(num_samples * 0.10),
    }

    seen = set()
    positives, negatives = [], []

    ab_star_sequences = _create_ab_star_sequences(max_len)

    # Positive sampling
    positives.extend(_sample_category(lambda: _gen_ab_star_unique(ab_star_sequences),
                                      pos_breakdown['ab_star'], 1, seen, allow_duplicates=True))
    positives.extend(_sample_category(lambda: _gen_contains(max_len),
                                      pos_breakdown['contains'], 1, seen))
    
    # Fill remaining positives
    remaining_pos_target = pos_target - len(positives)
    if remaining_pos_target > 0:
        positives.extend(_sample_category(lambda: _gen_random_string_candidate(max_len, ALPHABET_CHARS_ABC),
                                          remaining_pos_target, 1, seen))
    positives = positives[:pos_target] # Ensure exact count

    # Negative sampling
    negatives.extend(_sample_category(lambda: _gen_near_miss_general(max_len),
                                      neg_breakdown['near_miss_general'], 0, seen))
    negatives.extend(_sample_category(lambda: _gen_ab_star_specific_near_miss(max_len),
                                      neg_breakdown['near_miss_ab_star_specific'], 0, seen))

    # Fill remaining negatives
    remaining_neg_target = neg_target - len(negatives)
    if remaining_neg_target > 0:
        negatives.extend(_sample_category(lambda: _gen_random_string_candidate(max_len, ALPHABET_CHARS_ABC),
                                          remaining_neg_target, 0, seen))
    negatives = negatives[:neg_target] # Ensure exact count

    # Combine and shuffle
    dataset = [(s, 1) for s in positives] + [(s, 0) for s in negatives]
    random.shuffle(dataset)

    # Convert to int sequences
    int_data = [([VOCAB[c] for c in s if c in VOCAB], lbl) for s, lbl in dataset]
    return int_data, VOCAB

if __name__ == "__main__":
    
    
    # Create unique dataset version based on parameters
    dataset_version = f"samples_{NUM_SAMPLES}_maxlen_{MAX_LEN}_target_{TARGET_SUBSTRING}"
      # Initialize wandb for dataset generation
    run = wandb.init(project="rwkv-regex-learning", 
                     job_type="dataset_generation",
                     name=f"dataset_{dataset_version}",
                     tags=["dataset", f"samples_{NUM_SAMPLES}", f"maxlen_{MAX_LEN}", f"target_{TARGET_SUBSTRING}", "generation"],
                     config={
                         "num_samples": NUM_SAMPLES,
                         "max_len": MAX_LEN,
                         "target_substring": TARGET_SUBSTRING,
                         "vocab_size": len(VOCAB),
                         "alphabet_chars_ab": ALPHABET_CHARS_AB,
                         "alphabet_chars_abc": ALPHABET_CHARS_ABC,
                         "dataset_version": dataset_version
                     })
    
    print(f"Generating {NUM_SAMPLES} samples up to max_len {MAX_LEN}...")
    data, vocab_map = generate_dataset(NUM_SAMPLES, MAX_LEN)
    
    train_ratio = 0.8
    num_train = int(len(data) * train_ratio)
    
    train_data = data[:num_train]
    val_data = data[num_train:]
    
    dataset_obj = {
        'train_data': train_data,
        'val_data': val_data,
        'vocab': vocab_map,
        'max_len': MAX_LEN,
        'target_substring': TARGET_SUBSTRING
    }
    
    file_path = 'regex_dataset.json'
    with open(file_path, 'w') as f:
        json.dump(dataset_obj, f, indent=4)
        
    print(f"\nGenerated {len(train_data)} training samples and {len(val_data)} validation samples.")
    
    pos_ab_star_only_count = 0
    pos_abbccc_only_count = 0
    pos_both_count = 0
    neg_count = 0
    
    for s_int, l_int in data:
        s_str = "".join([INV_VOCAB.get(t, "") for t in s_int])
        is_ab = check_ab_star(s_str, tuple(ALPHABET_CHARS_AB))
        is_abbccc = check_contains_substring(s_str, TARGET_SUBSTRING)
        
        if l_int == 1:
            if is_ab and is_abbccc: pos_both_count += 1 # This will be 0
            elif is_ab: pos_ab_star_only_count += 1
            elif is_abbccc: pos_abbccc_only_count += 1
        else: neg_count += 1

    total_generated = len(data)
    if total_generated > 0:
        print(f"Actual distribution (approx {total_generated} total samples):")
        print(f"  Positive (ab)* only: {pos_ab_star_only_count} ({(pos_ab_star_only_count/total_generated*100):.1f}%)")
        print(f"  Positive (contains '{TARGET_SUBSTRING}') only: {pos_abbccc_only_count} ({(pos_abbccc_only_count/total_generated*100):.1f}%)")
        print(f"  Positive (both conditions met): {pos_both_count} ({(pos_both_count/total_generated*100):.1f}%) # Note: Expected 0 as TARGET_SUBSTRING ('{TARGET_SUBSTRING}') contains 'c' and (ab)* does not.")
        total_pos = pos_ab_star_only_count + pos_abbccc_only_count + pos_both_count
        print(f"  Total Positive: {total_pos} ({(total_pos/total_generated*100):.1f}%)")
        print(f"  Total Negative: {neg_count} ({(neg_count/total_generated*100):.1f}%)")
    print(f"Vocabulary: {vocab_map}")
    print(f"Max length: {MAX_LEN}")
    print(f"Dataset saved to {file_path}")
    
    # Log dataset statistics to wandb
    wandb.log({
        "dataset/total_samples": len(data),
        "dataset/train_samples": len(train_data),
        "dataset/val_samples": len(val_data),
        "dataset/pos_ab_star_only": pos_ab_star_only_count,
        "dataset/pos_contains_abbccc_only": pos_abbccc_only_count,
        "dataset/pos_both_conditions": pos_both_count,
        "dataset/total_positive": pos_ab_star_only_count + pos_abbccc_only_count + pos_both_count,
        "dataset/total_negative": neg_count,
        "dataset/pos_ab_star_percentage": (pos_ab_star_only_count/total_generated*100) if total_generated > 0 else 0,
        "dataset/pos_abbccc_percentage": (pos_abbccc_only_count/total_generated*100) if total_generated > 0 else 0,
        "dataset/neg_percentage": (neg_count/total_generated*100) if total_generated > 0 else 0
    })
      # Save dataset as wandb artifact with versioned name
    artifact = wandb.Artifact(
        name=f"regex_dataset_{dataset_version}",
        type="dataset",
        description=f"Regex dataset with {NUM_SAMPLES} samples, max_len={MAX_LEN}, target='{TARGET_SUBSTRING}'",
        metadata={
            "num_samples": NUM_SAMPLES,
            "max_len": MAX_LEN,
            "target_substring": TARGET_SUBSTRING,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "vocab_size": len(VOCAB),
            "pos_ab_star_only": pos_ab_star_only_count,
            "pos_contains_abbccc_only": pos_abbccc_only_count,
            "total_positive": pos_ab_star_only_count + pos_abbccc_only_count + pos_both_count,
            "total_negative": neg_count
        }
    )
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)
    
    # Log the artifact name for easy reference in training
    wandb.summary["dataset_artifact_name"] = f"regex_dataset_{dataset_version}"
    wandb.summary["dataset_file_path"] = file_path
    
    print(f"\n✓ Dataset artifact saved as: regex_dataset_{dataset_version}")
    print(f"✓ This dataset can be reused in training by referencing the artifact name")
    
    wandb.finish()
