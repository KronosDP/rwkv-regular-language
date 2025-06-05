import os


def load_fsm(filepath):
    """
    Loads an FSM definition from a file.

    The file format is expected to be:
    - Lines of transitions: source_state target_state input_symbol output_symbol
    - Followed by a line with the initial_state
    - Followed by lines each containing one accepting_state

    Args:
        filepath (str): The path to the FSM definition file.

    Returns:
        tuple: (initial_state, accepting_states, transitions)
               transitions is a dict: {(source, input_sym): (target, output_sym)}
    
    Raises:
        FileNotFoundError: If the filepath does not exist.
        ValueError: If the file format is incorrect or FSM definition is incomplete.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file '{filepath}' was not found.")

    transitions = {}
    initial_state = None
    accepting_states = set()
    
    lines = []
    with open(filepath, 'r') as f:
        # Read all non-empty lines, stripping whitespace
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("FSM file is empty.")

    # Determine where state definitions (initial, accepting) start
    # This is assumed to be the first line that doesn't have 4 parts (transition format)
    first_state_def_line_index = -1
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 4:
            if len(parts) == 1: # Potential start of state definitions
                first_state_def_line_index = i
                break
            else:
                # This line is neither a valid transition nor a single-token state definition
                raise ValueError(
                    f"Malformed line {i+1} in FSM file: '{line}'. "
                    "Expected 4 parts for a transition or 1 part for a state definition."
                )
    
    if first_state_def_line_index == -1:
        # This implies all lines were formatted as transitions,
        # meaning initial/accepting states are missing.
        raise ValueError(
            "FSM definition incomplete: Initial and/or accepting states not found at the end of the file."
        )

    # Parse transitions (lines before first_state_def_line_index)
    for i in range(first_state_def_line_index):
        line = lines[i]
        parts = line.split()
        # This check is somewhat redundant due to the loop finding first_state_def_line_index,
        # but good for safety if logic changes.
        if len(parts) == 4:
            source, target, input_sym, output_sym = parts
            if (source, input_sym) in transitions:
                # Handle non-determinism or overwrite: current policy is to overwrite with a warning.
                # Depending on FSM type, non-determinism might need different handling.
                print(f"Warning: Duplicate transition for ({source}, {input_sym}). "
                      f"Overwriting previous: {transitions[(source, input_sym)]} with ({target}, {output_sym}).")
            transitions[(source, input_sym)] = (target, output_sym)
        else:
            # This case should ideally not be reached if first_state_def_line_index is found correctly.
             raise ValueError(f"Unexpected line format in transition section at line {i+1}: '{line}'.")


    # Parse initial state and accepting states
    state_def_lines = lines[first_state_def_line_index:]
    if not state_def_lines: # Should be caught by first_state_def_line_index logic
        raise ValueError("Initial and accepting states definitions are missing.")
    
    # Initial state is the first line in this section
    initial_state_parts = state_def_lines[0].split()
    if len(initial_state_parts) != 1:
        raise ValueError(
            f"Malformed initial state line: '{state_def_lines[0]}'. Expected a single state identifier."
        )
    initial_state = initial_state_parts[0]

    # Accepting states are the subsequent lines in this section
    if len(state_def_lines) > 1:
        for line_idx, line_content in enumerate(state_def_lines[1:]):
            parts = line_content.split()
            if len(parts) != 1:
                original_line_num = first_state_def_line_index + 1 + line_idx + 1
                raise ValueError(
                    f"Malformed accepting state line (approx. original line {original_line_num}): "
                    f"'{line_content}'. Expected a single state identifier."
                )
            accepting_states.add(parts[0])
            
    if initial_state is None: # Should be caught earlier
        raise ValueError("Initial state could not be determined.")
        
    return initial_state, accepting_states, transitions

def simulate_fsm(initial_state, accepting_states, transitions, input_string):
    """
    Simulates the FSM with a given input string.

    Args:
        initial_state (str): The starting state of the FSM.
        accepting_states (set): A set of accepting states.
        transitions (dict): The FSM's transition rules.
        input_string (str): The input string to process.

    Returns:
        tuple: (accepted (bool), output_sequence (list of str))
               'accepted' is True if the FSM ends in an accepting state, False otherwise.
               'output_sequence' is the list of output symbols generated.
    """
    current_state = initial_state
    output_sequence = []

    # Handle empty input string
    if not input_string:
        if current_state in accepting_states:
            return True, output_sequence
        else:
            return False, output_sequence

    for symbol in input_string:
        if (current_state, symbol) in transitions:
            next_state, output_sym = transitions[(current_state, symbol)]
            current_state = next_state
            output_sequence.append(output_sym)
        else:
            # No transition found for the current state and input symbol
            # The FSM gets stuck and rejects the input.
            return False, output_sequence
    
    # After processing all symbols, check if the current state is an accepting state
    if current_state in accepting_states:
        return True, output_sequence
    else:
        return False, output_sequence

def is_ab_star_pattern(s):
    """Check if string matches (ab)* pattern"""
    if len(s) % 2 != 0:
        return False
    for i in range(0, len(s), 2):
        if i + 1 >= len(s) or s[i] != 'a' or s[i + 1] != 'b':
            return False
    return True

def run_validation_tests(initial_state, accepting_states, transitions, validation_file="validation.txt"):
    """
    Run test cases from validation file and calculate pass percentage.
    
    Args:
        initial_state (str): The starting state of the FSM.
        accepting_states (set): A set of accepting states.
        transitions (dict): The FSM's transition rules.
        validation_file (str): Path to the validation test file.
    
    Returns:
        tuple: (total_tests, passed_tests, pass_percentage)
    """
    if not os.path.exists(validation_file):
        print(f"Validation file '{validation_file}' not found.")
        return 0, 0, 0.0
    
    test_cases = []
    with open(validation_file, 'r') as f:
        test_cases = [line.strip() for line in f if line.strip()]
    
    if not test_cases:
        print("No test cases found in validation file.")
        return 0, 0, 0.0
    
    total_tests = len(test_cases)
    passed_tests = 0
    failed_cases = []
    correct_rejections = 0
    incorrect_rejections = 0
    
    print(f"\nRunning {total_tests} test cases from {validation_file}...")
    print("=" * 50)
    for i, test_input in enumerate(test_cases, 1):
        accepted, output_sequence = simulate_fsm(initial_state, accepting_states, transitions, test_input)
        
        if accepted:
            passed_tests += 1
            print(f"Test {i:3d}: ACCEPTED - '{test_input}'")
        else:
            failed_cases.append((i, test_input))
            # Check if the rejection is correct
            should_accept = is_ab_star_pattern(test_input) or "abbccc" in test_input
            if should_accept:
                incorrect_rejections += 1
                print(f"Test {i:3d}: REJECTED - '{test_input}' [INCORRECT - should accept]")
            else:
                correct_rejections += 1
                print(f"Test {i:3d}: REJECTED - '{test_input}' [CORRECT]")
    
    pass_percentage = (passed_tests / total_tests) * 100
    fsm_accuracy = ((passed_tests + correct_rejections) / total_tests) * 100
    
    print("=" * 50)
    print(f"Test Results Summary:")
    print(f"  Total tests: {total_tests}")
    print(f"  Accepted: {passed_tests}")
    print(f"  Correctly rejected: {correct_rejections}")
    print(f"  Incorrectly rejected: {incorrect_rejections}")
    print(f"  Accept rate: {pass_percentage:.2f}%")
    print(f"  FSM accuracy: {fsm_accuracy:.2f}%")
    
    if incorrect_rejections > 0:
        print(f"\nIncorrectly rejected test cases (should have been accepted):")
        count = 0
        for test_num, test_input in failed_cases:
            should_accept = is_ab_star_pattern(test_input) or "abbccc" in test_input
            if should_accept and count < 10:
                pattern_type = "(ab)*" if is_ab_star_pattern(test_input) else "contains 'abbccc'"
                print(f"  Test {test_num}: '{test_input[:50]}{'...' if len(test_input) > 50 else ''}' [{pattern_type}]")
                count += 1
        if count == 10 and incorrect_rejections > 10:
            print(f"  ... and {incorrect_rejections - 10} more")
    
    return total_tests, passed_tests, pass_percentage

def main():
    """
    Main function to load FSM and process user inputs.
    """

    filepath = "tes.txt"
    
    try:
        initial_state, accepting_states, transitions = load_fsm(filepath)
        print("\nFSM loaded successfully.")
        print(f"  Initial state: {initial_state}")
        print(f"  Accepting states: {accepting_states}")
        print(f"  Transitions loaded: {len(transitions)} rules")

        # Run validation tests first
        run_validation_tests(initial_state, accepting_states, transitions)

        # Interactive mode
        print("\n" + "=" * 50)
        print("Interactive Mode - Enter test strings manually")
        print("=" * 50)
        
        while True:
            user_input_str = input("\nEnter an input string (e.g., 'abc', or type 'quit' to exit): ")
            if user_input_str.lower() == 'quit':
                print("Exiting simulator.")
                break
            
            # Assuming input symbols are single characters. If not, adjust input processing.
            # For this example, we treat the input string as a sequence of characters.
            
            accepted, output_sequence = simulate_fsm(initial_state, accepting_states, transitions, user_input_str)
            
            output_str = "".join(output_sequence)
            
            if accepted:
                print(f"Input '{user_input_str}' => ACCEPTED")
                print(f"Output sequence: '{output_str}'")
            else:
                print(f"Input '{user_input_str}' => REJECTED")
                if output_sequence:
                     print(f"Partial output sequence before rejection: '{output_str}'")
                else:
                     # This could mean the very first symbol had no transition from the initial state,
                     # or the input string was empty and the initial state was not accepting.
                     if user_input_str: # Only print this if there was an attempt to process symbols
                        print("Processing stopped due to no valid transition or FSM structure.")


    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error loading or processing FSM: {e}")
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
