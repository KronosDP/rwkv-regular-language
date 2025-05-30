import random

# --- Configuration ---
VOCAB_CHARS = ['a', 'b', 'c']
ALPHABET_CHARS_AB = ['a', 'b'] # For (ab)* check
TARGET_SUBSTRING = "abbccc"
MAX_LEN = 50
NUM_SAMPLES_PER_CATEGORY = 150 # Number of unique samples to aim for in each category
OUTPUT_FILE = "validation.txt"

# --- Language Checking Functions ---
def check_ab_star(s):
    """Checks if a string matches (ab)* pattern."""
    if not s: return True # Empty string is (ab)*
    if any(char not in ALPHABET_CHARS_AB for char in s): return False
    if len(s) % 2 != 0: return False
    for i in range(0, len(s), 2):
        if s[i] != 'a' or s[i+1] != 'b':
            return False
    return True

def check_contains_abbccc(s):
    """Checks if a string contains TARGET_SUBSTRING."""
    return TARGET_SUBSTRING in s

# --- String Generation Utilities ---
def generate_random_string(length, alphabet):
    """Generates a random string of a given length from an alphabet."""
    if length < 0: length = 0
    return "".join(random.choice(alphabet) for _ in range(length))

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

def main():
    all_generated_strings = set()
    
    category1_ab_star = []
    category2_contains_abbccc = []
    category3_neither = []

    max_attempts_multiplier = 20 # Try harder to find unique strings

    # Category 1: (ab)* strings
    # (ab)* strings cannot contain "abbccc" because "abbccc" has "c" and "bb".
    attempts = 0
    target_cat1 = NUM_SAMPLES_PER_CATEGORY
    while len(category1_ab_star) < target_cat1 and attempts < target_cat1 * max_attempts_multiplier:
        s_len = random.randint(0, MAX_LEN) # Vary length for (ab)*
        s = generate_ab_star_candidate(s_len)
        if check_ab_star(s) and s not in all_generated_strings: # check_ab_star is somewhat redundant here but good for clarity
            category1_ab_star.append(s)
            all_generated_strings.add(s)
        attempts += 1
    if len(category1_ab_star) < target_cat1:
        print(f"Warning: Could only generate {len(category1_ab_star)} unique (ab)* strings out of {target_cat1} desired.")

    # Category 2: Strings containing "abbccc"
    # These will not be (ab)* strings.
    attempts = 0
    target_cat2 = NUM_SAMPLES_PER_CATEGORY
    while len(category2_contains_abbccc) < target_cat2 and attempts < target_cat2 * max_attempts_multiplier:
        s_len = random.randint(len(TARGET_SUBSTRING), MAX_LEN) # Must be long enough for substring
        s = generate_contains_abbccc_candidate(s_len, VOCAB_CHARS)
        # Ensure it actually contains abbccc and is not an (ab)* string (which it shouldn't be)
        # and is not already collected
        if check_contains_abbccc(s) and not check_ab_star(s) and s not in all_generated_strings:
            category2_contains_abbccc.append(s)
            all_generated_strings.add(s)
        attempts += 1
    if len(category2_contains_abbccc) < target_cat2:
        print(f"Warning: Could only generate {len(category2_contains_abbccc)} unique strings containing '{TARGET_SUBSTRING}' out of {target_cat2} desired.")
        
    # Category 3: Random gibberish strings (neither (ab)* nor containing "abbccc")
    attempts = 0
    target_cat3 = NUM_SAMPLES_PER_CATEGORY
    while len(category3_neither) < target_cat3 and attempts < target_cat3 * max_attempts_multiplier:
        length = random.randint(0, MAX_LEN) 
        s = generate_random_string(length, VOCAB_CHARS)
        if not check_ab_star(s) and not check_contains_abbccc(s) and s not in all_generated_strings:
            category3_neither.append(s)
            all_generated_strings.add(s)
        attempts += 1
    if len(category3_neither) < target_cat3:
        print(f"Warning: Could only generate {len(category3_neither)} unique gibberish strings out of {target_cat3} desired.")

    # Combine all strings and shuffle
    final_list_of_strings = category1_ab_star + category2_contains_abbccc + category3_neither
    random.shuffle(final_list_of_strings)

    # Write to file
    output_path = "validation.txt" # Assuming script is run from workspace root
    with open(output_path, 'w') as f:
        for s in final_list_of_strings:
            f.write(s + '\n')
            
    print(f"Generated {len(final_list_of_strings)} unique strings in total.")
    print("Distribution of generated strings (categories are mutually exclusive by design):")
    print(f"  - Category 1 ((ab)* pattern): {len(category1_ab_star)} strings")
    print(f"  - Category 2 (contains '{TARGET_SUBSTRING}', not (ab)*): {len(category2_contains_abbccc)} strings")
    print(f"  - Category 3 (Random gibberish, neither of above): {len(category3_neither)} strings")
    print(f"Validation data written to {output_path}")

if __name__ == "__main__":
    main()
