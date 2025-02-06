import numpy as np

def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.

    Args:
        point1: A numerical value representing the first point.
        point2: A numerical value representing the second point.

    Returns:
        The Euclidean distance between point1 and point2.
    """
    return np.sqrt(np.sum((point1 - point2)**2))

def dynamic_time_wrapping(seq1, seq2, distance_metric=euclidean_distance, return_path=False):
    """
    Calculates the Dynamic Time Warping (DTW) distance between two sequences
    and optionally returns the optimal warping path (traceback).

    Args:
        seq1: The first sequence (list or numpy array of numerical values).
        seq2: The second sequence (list or numpy array of numerical values).
        distance_metric: A function that calculates the distance between two points.
                         Defaults to euclidean_distance.
        return_path: A boolean flag. If True, the function returns the optimal
                     warping path in addition to the DTW distance.
                     Defaults to False.

    Returns:
        If return_path is False:
            - The DTW distance between seq1 and seq2.
        If return_path is True:
            - A tuple containing:
                - The DTW distance between seq1 and seq2.
                - The optimal warping path as a list of index pairs (tuples).
    """
    n = len(seq1)
    m = len(seq2)

    # Initialize the cost matrix and path matrix
    cost_matrix = np.zeros((n + 1, m + 1))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf
    path_matrix = np.zeros((n + 1, m + 1), dtype=int)  # 0: start, 1: up, 2: super diagonal, 3: diagonal

    # Fill in the cost matrix and path matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = distance_metric(seq1[i - 1], seq2[j - 1])
            insertion_cost = cost_matrix[i, j - 1] # Level
            if i - 2 < 0:
                shrink_cost = np.inf
            else:
                shrink_cost = cost_matrix[i - 2, j - 1] # Super Diagonal
            match_cost = cost_matrix[i - 1, j - 1] # Dia

            min_cost = min(insertion_cost, shrink_cost, match_cost)
            cost_matrix[i, j] = cost + min_cost

            if min_cost == insertion_cost:
                path_matrix[i, j] = 1  # Level (Insertion)
            elif min_cost == shrink_cost:
                path_matrix[i, j] = 2  # Super diagonal
            else: # min_cost == match_cost (or could be equal to multiple, diagonal preferred if tied - standard DTW)
                path_matrix[i, j] = 3  # Diagonal (Match)

    if not return_path:
        return cost_matrix[n, m]
    else:
        # Traceback to find the optimal path
        path = []
        i = n
        j = m
        while i > 0 or j > 0:
            path.append((i - 1, j - 1)) # Adjust indices to be 0-based for sequences
            print(path[-1])
            direction = path_matrix[i, j]
            if direction == 1:          # Level (Insertion)
                j = j - 1
            elif direction == 2:        # Super diagonal
                i = i - 2
                j = j - 1
            elif direction == 3:        # Diagonal (Match)
                i = i - 1
                j = j - 1
            else:                       # direction == 0 (Start) - should not happen normally, but for safety
                break

        formatted_path = np.array([[i[0] for i in reversed(path)], [i[1] for i in reversed(path)]])

        return cost_matrix[n, m], cost_matrix, formatted_path # Reverse path to be from start to end


def dynamic_time_wrapping_fast(sequences: np.ndarray, seq2, distance_metric=euclidean_distance, return_path=False):
    """
    Calculates the Dynamic Time Warping (DTW) distance between two sequences
    and optionally returns the optimal warping path (traceback).

    Args:
        seq1: The first sequence (list or numpy array of numerical values).
        seq2: The second sequence (list or numpy array of numerical values).
        distance_metric: A function that calculates the distance between two points.
                         Defaults to euclidean_distance.
        return_path: A boolean flag. If True, the function returns the optimal
                     warping path in addition to the DTW distance.
                     Defaults to False.

    Returns:
        If return_path is False:
            - The DTW distance between seq1 and seq2.
        If return_path is True:
            - A tuple containing:
                - The DTW distance between seq1 and seq2.
                - The optimal warping path as a list of index pairs (tuples).
    """
    n = len(seq1)
    m = len(seq2)

    # Initialize the cost matrix and path matrix
    cost_matrix = np.zeros((n + 1, m + 1))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf
    path_matrix = np.zeros((n + 1, m + 1), dtype=int)  # 0: start, 1: up, 2: super diagonal, 3: diagonal

    # Fill in the cost matrix and path matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = distance_metric(seq1[i - 1], seq2[j - 1])
            insertion_cost = cost_matrix[i - 1, j] # Level
            match_cost = cost_matrix[i - 1, j - 1] # Dia
            skip_cost = cost_matrix[i - 2, j - 1] # Super Diagonal

            min_cost = min(insertion_cost, skip_cost, match_cost)
            cost_matrix[i, j] = cost + min_cost

            if min_cost == insertion_cost:
                path_matrix[i, j] = 1  # Up (Insertion)
            elif min_cost == skip_cost:
                path_matrix[i, j] = 2  # Super diagonal
            else: # min_cost == match_cost (or could be equal to multiple, diagonal preferred if tied - standard DTW)
                path_matrix[i, j] = 3  # Diagonal (Match)

    if not return_path:
        return cost_matrix[n, m]
    else:
        # Traceback to find the optimal path
        path = []
        i = n
        j = m
        while i > 0 or j > 0:
            path.append((i - 1, j - 1)) # Adjust indices to be 0-based for sequences
            direction = path_matrix[i, j]
            if direction == 1:          # Up (Insertion)
                i = i - 1
            elif direction == 2:        # Super diagonal
                j = j - 1
                i = i - 2
            elif direction == 3:        # Diagonal (Match)
                i = i - 1
                j = j - 1
            else:                       # direction == 0 (Start) - should not happen normally, but for safety
                break

        return cost_matrix[n, m], path[::-1] # Reverse path to be from start to end

def find_shortest_dtw(sequence_stack, sample_sequence, distance_metric=euclidean_distance, return_path=False):
    """
    Compares a stack of sequences to a sample sequence using DTW and finds the
    shortest DTW distance among them. Optionally returns the traceback for the shortest path.

    Args:
        sequence_stack: A list of sequences.
        sample_sequence: The sample sequence.
        distance_metric: Distance metric function.
        return_path: If True, returns the traceback for the shortest path as well.

    Returns:
        If return_path is False:
            - A tuple: (shortest_distance, index_of_shortest)
        If return_path is True:
            - A tuple: (shortest_distance, index_of_shortest, shortest_path)
    """
    shortest_distance = float('inf')
    index_of_shortest = None
    shortest_path = None

    if not sequence_stack:
        return shortest_distance, index_of_shortest, shortest_path if return_path else index_of_shortest

    for index, seq in enumerate(sequence_stack):
        if return_path:
            distance, path = dynamic_time_wrapping(seq, sample_sequence, distance_metric, return_path=True)
        else:
            distance = dynamic_time_wrapping(seq, sample_sequence, distance_metric, return_path=False)

        if distance < shortest_distance:
            shortest_distance = distance
            index_of_shortest = index
            if return_path:
                shortest_path = path

    if return_path:
        return shortest_distance, index_of_shortest, shortest_path
    else:
        return shortest_distance, index_of_shortest


# --- Example Usage with Traceback ---
if __name__ == '__main__':
    sequence1 = [1, 2, 3, 4, 5]
    sequence2 = [1, 2, 2, 4, 5, 6]
    sample_sequence = [1, 2, 2, 3, 5]
    sequence_stack = [sequence1, sequence2]

    # Example with traceback from dtw_distance directly
    distance, path = dynamic_time_wrapping(sequence1, sample_sequence, return_path=True)
    print(f"DTW Distance between sequence1 and sample_sequence: {distance}")
    print(f"Optimal Warping Path (sequence1 vs sample_sequence): {path}")
    # Path is a list of (index_seq1, index_sample_sequence) pairs

    # Example with shortest distance and traceback from find_shortest_dtw
    shortest_dist, shortest_index, shortest_path_stack = find_shortest_dtw(
        sequence_stack, sample_sequence, return_path=True
    )

    print("\nSequence Stack:")
    for i, seq in enumerate(sequence_stack):
        print(f"  Sequence {i+1}: {seq}")
    print(f"Sample Sequence: {sample_sequence}")

    print(f"\nShortest DTW Distance (from stack): {shortest_dist}")
    print(f"Index of Sequence with Shortest Distance (in stack, 0-indexed): {shortest_index}")
    print(f"Sequence with Shortest Distance: {sequence_stack[shortest_index]}")
    print(f"Optimal Warping Path (shortest sequence vs sample): {shortest_path_stack}")