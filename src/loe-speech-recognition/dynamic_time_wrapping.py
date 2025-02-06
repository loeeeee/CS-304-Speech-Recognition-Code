import numpy as np

def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)

def dynamic_time_warping(sequence1, sequence2, distance_function, warping_window=np.inf, step_weight=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    Args:
        sequence1 (array): N1*M array
        sequence2 (array): N2*M array
        distance_function (func): Distance function used as cost measure
        warping_window (int, optional): Window size limiting the maximal distance 
                                         between indices of matched entries |i,j|. 
                                         Defaults to inf (no window).
        step_weight (float, optional): Weight applied on off-diagonal moves 
                                       of the path. As step_weight gets larger, 
                                       the warping path is increasingly biased 
                                       towards the diagonal. Defaults to 1.0.

    Returns:
        tuple: A tuple containing the minimum DTW distance, the local cost matrix, 
               the accumulated cost matrix, and the warping path.
    """
    assert len(sequence1)
    assert len(sequence2)
    assert np.isinf(warping_window) or (warping_window >= abs(len(sequence1) - len(sequence2)))
    assert step_weight > 0

    len_seq1, len_seq2 = len(sequence1), len(sequence2)

    if not np.isinf(warping_window):
        extended_cost_matrix = np.full((len_seq1 + 1, len_seq2 + 1), np.inf)
        for i in range(1, len_seq1 + 1):
            extended_cost_matrix[i, max(1, i - warping_window):min(len_seq2 + 1, i + warping_window + 1)] = 0
        extended_cost_matrix[0, 0] = 0
    else:
        extended_cost_matrix = np.zeros((len_seq1 + 1, len_seq2 + 1))
        extended_cost_matrix[0, 1:] = np.inf
        extended_cost_matrix[1:, 0] = np.inf

    cost_matrix = extended_cost_matrix[1:, 1:]  # View
    for i in range(len_seq1):
        for j in range(len_seq2):
            if (np.isinf(warping_window) or (max(0, i - warping_window) <= j <= min(len_seq2, i + warping_window))):
                cost_matrix[i, j] = distance_function(sequence1[i], sequence2[j])

    accumulated_cost_matrix = cost_matrix.copy()
    j_range = range(len_seq2)
    for i in range(len_seq1):
        if not np.isinf(warping_window):
            j_range = range(max(0, i - warping_window), min(len_seq2, i + warping_window + 1))
        for j in j_range:
            min_values = [extended_cost_matrix[i, j]]
            for k in range(1, min(len_seq1, len_seq2) + 1):  # Limit k to shortest sequence length
                i_k = min(i + k, len_seq1)
                j_k = min(j + k, len_seq2)
                min_values += [extended_cost_matrix[i_k, j] * step_weight, extended_cost_matrix[i, j_k] * step_weight]
            accumulated_cost_matrix[i, j] += min(min_values)

    if len(sequence1) == 1:
        path = np.zeros(len(sequence2)), range(len(sequence2))
    elif len(sequence2) == 1:
        path = range(len(sequence1)), np.zeros(len(sequence1))
    else:
        path = _traceback(extended_cost_matrix)  # Assuming _traceback is defined

    return accumulated_cost_matrix[-1, -1], cost_matrix, accumulated_cost_matrix, path

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

# Placeholder that has the same functionality
## Could be paralleled by numba
def dynamic_time_warping_efficiently(templates: np.ndarray, sample, labels,distance_function=euclidean_distance, warping_window=np.inf, step_weight=1.0):
    dtw_distance_min = 0x3f3f3f3f
    results = None
    for i, label in zip(range(templates.shape[-1]), labels):
        dtw_distance, cost_matrix, accumulated_cost_matrix, path = dynamic_time_warping(templates[...,i], sample, distance_function, warping_window, step_weight)
        if dtw_distance < dtw_distance_min:
            dtw_distance_min = dtw_distance
            results = dtw_distance, cost_matrix, accumulated_cost_matrix, path, label
    if results:
        return results
    else:
        raise Exception

if __name__ == "__main__":
    # Example usage (replace with your actual sequences and traceback function)
    seq1 = np.array([1, 2, 3, 4, 5])
    seq2 = np.array([2, 3, 4, 5, 6])

    dtw_distance, cost_matrix, accumulated_cost_matrix, path = dynamic_time_warping(seq1, seq2, euclidean_distance)

    print("DTW Distance:", dtw_distance)
    print("Cost Matrix:\n", cost_matrix) # These matrices can be very large
    print("Accumulated Cost Matrix:\n", accumulated_cost_matrix)
    print("Path:", path)