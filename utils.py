import random


# --- Language Checking Functions ---
def check_ab_star(s, ab_chars_tuple):
    """
    Checks if a string matches (char1 char2)* pattern.
    ab_chars_tuple should be a tuple like ('a', 'b').
    The string s should only contain characters from ab_chars_tuple.
    """
    char1, char2 = ab_chars_tuple
    if not s: return True
    # Check if all characters in s are either char1 or char2
    if any(char not in ab_chars_tuple for char in s): return False
    if len(s) % 2 != 0: return False
    for i in range(0, len(s), 2):
        if not (s[i] == char1 and s[i+1] == char2):
            return False
    return True

def check_contains_substring(s, substring_to_check):
    """Checks if a string contains a specific substring."""
    return substring_to_check in s

def get_language_label(s, ab_chars_tuple_for_ab_star, substring_to_check):
    """
    Labels a string: 1 if it matches (char1 char2)* pattern (using ab_chars_tuple_for_ab_star)
    OR if it contains substring_to_check. Otherwise, returns 0.
    """
    is_ab_star_match = check_ab_star(s, ab_chars_tuple_for_ab_star)
    contains_substring_match = check_contains_substring(s, substring_to_check)
    return 1 if is_ab_star_match or contains_substring_match else 0

# --- String Generation Utilities ---
def generate_random_string(length, alphabet_for_random_string):
    """Generates a random string of a given length from an alphabet."""
    if length <= 0: return ""
    return "".join(random.choice(alphabet_for_random_string) for _ in range(length))
