import random

from config import MAX_LEN  # Added import
from utils import (check_ab_star, check_contains_substring,
                   generate_random_string)

# --- Configuration ---
VOCAB_CHARS = ['a', 'b', 'c']
ALPHABET_CHARS_AB = ['a', 'b'] # For (ab)* check
TARGET_SUBSTRING = "abbccc"
NUM_SAMPLES_PER_CATEGORY = 200 # Number of unique samples to aim for in each category
OUTPUT_FILE = "validation.txt"

# Different validation lengths to test
VALIDATION_LENGTHS = [MAX_LEN, MAX_LEN + 10, MAX_LEN + 50, MAX_LEN * 10, MAX_LEN*100, MAX_LEN*1000]

def generate_ab_star_candidate(max_len):
    """Generates a candidate (ab)* string."""
    if max_len == 0: return ""
    max_pairs = max_len // 2
    num_pairs = random.randint(0, max_pairs) # 0 pairs gives empty string
    return "ab" * num_pairs

def generate_contains_abbccc_candidate(max_len, alphabet_full):
    """Generates a string containing TARGET_SUBSTRING."""
    if len(TARGET_SUBSTRING) > max_len:
        # This case should ideally not be hit if MAX_LEN is reasonable
        return TARGET_SUBSTRING[:max_len] 

    prefix_max_len = max_len - len(TARGET_SUBSTRING)
    prefix_len = random.randint(0, prefix_max_len)
    suffix_len = max_len - len(TARGET_SUBSTRING) - prefix_len
    
    prefix = generate_random_string(prefix_len, alphabet_full)
    suffix = generate_random_string(suffix_len, alphabet_full)
    return prefix + TARGET_SUBSTRING + suffix

def generate_category_strings(category_name, target_count, generator_func, validator_func, all_generated_strings, max_attempts_multiplier=20):
    """Helper function to generate strings for a specific category."""
    category_strings = []
    attempts = 0
    
    while len(category_strings) < target_count and attempts < target_count * max_attempts_multiplier:
        s = generator_func()
        if validator_func(s) and s not in all_generated_strings:
            category_strings.append(s)
            all_generated_strings.add(s)
        attempts += 1
    
    if len(category_strings) < target_count:
        print(f"Warning: Could only generate {len(category_strings)} unique {category_name} strings out of {target_count} desired.")
    
    return category_strings

def generate_all_categories(all_generated_strings, max_attempts_multiplier, validation_max_len):
    """Generate all three categories of strings."""
    # Category 1: (ab)* strings
    category1_ab_star = generate_category_strings(
        category_name="(ab)*",
        target_count=NUM_SAMPLES_PER_CATEGORY,
        generator_func=lambda: generate_ab_star_candidate(random.randint(0, validation_max_len)),
        validator_func=lambda s: check_ab_star(s, tuple(ALPHABET_CHARS_AB)),
        all_generated_strings=all_generated_strings,
        max_attempts_multiplier=max_attempts_multiplier
    )

    # Category 2: Strings containing "abbccc"
    category2_contains_abbccc = generate_category_strings(
        category_name="contains 'abbccc'",
        target_count=NUM_SAMPLES_PER_CATEGORY,
        generator_func=lambda: generate_contains_abbccc_candidate(random.randint(len(TARGET_SUBSTRING), validation_max_len), VOCAB_CHARS),
        validator_func=lambda s: check_contains_substring(s, TARGET_SUBSTRING) and not check_ab_star(s, tuple(ALPHABET_CHARS_AB)),
        all_generated_strings=all_generated_strings,
        max_attempts_multiplier=max_attempts_multiplier
    )
        
    # Category 3: Random gibberish strings (neither (ab)* nor containing "abbccc")
    category3_neither = generate_category_strings(
        category_name="gibberish (neither)",
        target_count=NUM_SAMPLES_PER_CATEGORY,
        generator_func=lambda: generate_random_string(random.randint(0, validation_max_len), VOCAB_CHARS),
        validator_func=lambda s: not check_ab_star(s, tuple(ALPHABET_CHARS_AB)) and not check_contains_substring(s, TARGET_SUBSTRING),
        all_generated_strings=all_generated_strings,
        max_attempts_multiplier=max_attempts_multiplier
    )
    
    return category1_ab_star, category2_contains_abbccc, category3_neither

def write_validation_data(final_list_of_strings, max_len):
    """Write validation data to file."""
    output_path = f"validation_len{max_len}.txt"
    with open(output_path, 'w') as f:
        for s in final_list_of_strings:
            f.write(s + '\n')
    return output_path

def main():
    all_generated_strings = set()
    max_attempts_multiplier = 20

    for i, validation_max_len in enumerate(VALIDATION_LENGTHS):
        print(f"\nGenerating validation data for max_len = {validation_max_len}...")
        
        # Generate all categories for this length
        category1_ab_star, category2_contains_abbccc, category3_neither = generate_all_categories(
            all_generated_strings, max_attempts_multiplier, validation_max_len
        )

        # Combine all strings and shuffle
        final_list_of_strings = category1_ab_star + category2_contains_abbccc + category3_neither
        random.shuffle(final_list_of_strings)        # Write to file with length suffix
        output_path = write_validation_data(final_list_of_strings, validation_max_len)
                
        print(f"Generated {len(final_list_of_strings)} unique strings in total.")
        print("Distribution of generated strings (categories are mutually exclusive by design):")
        print(f"  - Category 1 ((ab)* pattern): {len(category1_ab_star)} strings")
        print(f"  - Category 2 (contains '{TARGET_SUBSTRING}', not (ab)*): {len(category2_contains_abbccc)} strings")
        print(f"  - Category 3 (Random gibberish, neither of above): {len(category3_neither)} strings")
        print(f"Validation data written to {output_path}")

if __name__ == "__main__":
    main()
