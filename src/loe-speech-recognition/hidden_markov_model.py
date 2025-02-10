import numpy as np
import os
import librosa
from scipy.stats import multivariate_normal

from mfcc import compute_mfcc

NUM_STATES = 5
FEATURE_DIM = 39 
NUM_DIGITS = 10 
digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
NUM_TRAIN_RECORDS_PER_DIGIT = 10 
NUM_TEST_RECORDS_PER_DIGIT = 10 
MAX_ITERATIONS_SEGM_KMEANS = 50 

class HMM_SingleGaussian:
    def __init__(self, num_states, feature_dim):
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.transition_probs = None # (num_states, num_states)
        self.means = None          # (num_states, feature_dim) Means of Gaussian emissions
        self.covariances = None    # (num_states, feature_dim, feature_dim) Covariances of Gaussian emissions

    def initialize_params(self, training_sequences):
        self.transition_probs = np.zeros((self.num_states, self.num_states))
        self.means = np.zeros((self.num_states, self.feature_dim))
        self.covariances = np.array([np.eye(self.feature_dim) for _ in range(self.num_states)]) # Initialize covariances to identity

        total_data_points_per_state = [0] * self.num_states
        sum_data_points_per_state = [np.zeros(self.feature_dim) for _ in range(self.num_states)]
        state_assignments = [] 

        for seq in training_sequences:
            seq_len = len(seq)
            segment_len = seq_len // self.num_states
            current_state_assignments = []

            for state in range(self.num_states):
                start_index = state * segment_len
                end_index = (state + 1) * segment_len if state < self.num_states - 1 else seq_len
                segment_data = seq[start_index:end_index]
                if len(segment_data) > 0: 
                    self.means[state] += np.mean(segment_data, axis=0)
                    total_data_points_per_state[state] += len(segment_data)
                    sum_data_points_per_state[state] = segment_data
                current_state_assignments.extend([state] * len(segment_data))
            state_assignments.append(current_state_assignments)

        for state in range(self.num_states):
            if total_data_points_per_state[state] > 0:
                self.means[state] /= total_data_points_per_state[state]

        for state in range(self.num_states):
             if total_data_points_per_state[state] > 0 and len(sum_data_points_per_state[state]) > 1: 
                self.covariances[state] = np.cov(sum_data_points_per_state[state].T)
                if np.linalg.det(self.covariances[state]) <= 0:
                    self.covariances[state] += 0.01 * np.eye(self.feature_dim)
             else:
                 self.covariances[state] = np.eye(self.feature_dim)

        for i in range(self.num_states):
            if i < self.num_states - 1:
                self.transition_probs[i, i] = 0.5
                self.transition_probs[i, i + 1] = 0.5
            else:
                self.transition_probs[i, i] = 1.0 # Last state self-loop


    def segmental_kmeans_train(self, training_sequences, max_iterations=MAX_ITERATIONS_SEGM_KMEANS):
        """
        Trains the HMM using segmental K-means.
        """
        self.initialize_params(training_sequences)

        for iteration in range(max_iterations):
            print(f"Segmental K-means Iteration: {iteration + 1}")
            new_means = np.zeros_like(self.means)
            new_covariances = np.array([np.zeros_like(self.covariances[0]) for _ in range(self.num_states)])
            new_transition_counts = np.zeros_like(self.transition_probs)
            state_counts = np.zeros(self.num_states)
            total_data_points_per_state = np.zeros(self.num_states)
            state_assignments_all_seqs = []


            for seq in training_sequences:
                viterbi_path = self.viterbi(seq) 
                state_assignments_all_seqs.append(viterbi_path)

                for t in range(len(seq)):
                    state = viterbi_path[t]
                    new_means[state] += seq[t]
                    state_counts[state] += 1
                    total_data_points_per_state[state] += 1
                    if t > 0:
                        prev_state = viterbi_path[t-1]
                        new_transition_counts[prev_state, state] += 1


            # M-step: Re-estimate parameters
            for state in range(self.num_states):
                if state_counts[state] > 0:
                    new_means[state] /= state_counts[state]

            for seq_idx, seq in enumerate(training_sequences):
                viterbi_path = state_assignments_all_seqs[seq_idx]
                for t in range(len(seq)):
                    state = viterbi_path[t]
                    diff = seq[t] - new_means[state]
                    new_covariances[state] += np.outer(diff, diff) 

            for state in range(self.num_states):
                if state_counts[state] > self.feature_dim + 1:
                    new_covariances[state] /= state_counts[state]
                    # Regularization/smoothing of covariance
                    new_covariances[state] += 0.01 * np.eye(self.feature_dim)
                else:
                    new_covariances[state] = self.covariances[state].copy()

            # Re-estimate transition probabilities
            for i in range(self.num_states):
                row_sum = np.sum(new_transition_counts[i, :])
                if row_sum > 0:
                    self.transition_probs[i, :] = new_transition_counts[i, :] / row_sum
                else:
                    # If no transitions from state i, keep previous transition probabilities (or re-initialize)
                    self.transition_probs[i, :] = self.transition_probs[i,:].copy()


            # Check for convergence
            prev_means = self.means.copy()
            self.means = new_means
            self.covariances = new_covariances


            mean_diff = np.sum(np.abs(self.means - prev_means)) # Simple convergence check
            if mean_diff < 1e-3: # Arbitrary threshold for convergence
                print("Converged.")
                break


    def viterbi(self, observation_sequence):
        """
        Viterbi algorithm to find the best state sequence for a given observation sequence.
        """
        T = len(observation_sequence)
        N = self.num_states

        log_delta = np.zeros((T, N)) # log-probabilities of the most likely path ending in state j at time t
        psi = np.zeros((T, N), dtype=int)   

        # Initialization (t=0)
        for s in range(N):
            log_delta[0, s] = multivariate_normal.logpdf(observation_sequence[0], self.means[s], self.covariances[s], allow_singular=True) # Assuming equal prior probabilities for starting states

        # Recursion (t=1 to T-1)
        for t in range(1, T):
            for j in range(N):
                max_log_prob = -float('inf')
                best_prev_state = 0
                for i in range(N):
                    log_prob = log_delta[t-1, i] + np.log(self.transition_probs[i, j])
                    if log_prob > max_log_prob:
                        max_log_prob = log_prob
                        best_prev_state = i
                log_delta[t, j] = max_log_prob + multivariate_normal.logpdf(observation_sequence[t], self.means[j], self.covariances[j], allow_singular=True)
                psi[t, j] = best_prev_state

        # Termination
        best_path_prob = np.max(log_delta[T-1, :])
        last_state = np.argmax(log_delta[T-1, :])

        # Backtracking
        viterbi_path = [0] * T
        viterbi_path[T-1] = last_state
        for t in range(T-2, -1, -1):
            viterbi_path[t] = psi[t+1, viterbi_path[t+1]]

        return viterbi_path


    def log_likelihood(self, observation_sequence):
        """
        Calculates the log-likelihood of an observation sequence given the HMM using the forward algorithm.
        """
        T = len(observation_sequence)
        N = self.num_states
        log_alpha = np.zeros((T, N))

        # Initialization (t=0)
        for s in range(N):
            log_alpha[0, s] = multivariate_normal.logpdf(observation_sequence[0], self.means[s], self.covariances[s], allow_singular=True) # Assuming equal prior probabilities for starting states

        # Forward recursion (t=1 to T-1)
        for t in range(1, T):
            for j in range(N):
                log_sum_exp = -float('inf') # Initialize for log-sum-exp trick
                for i in range(N):
                    log_sum_exp = np.logaddexp(log_sum_exp, log_alpha[t-1, i] + np.log(self.transition_probs[i, j]))
                log_alpha[t, j] = log_sum_exp + multivariate_normal.logpdf(observation_sequence[t], self.means[j], self.covariances[j], allow_singular=True)

        # Termination: Sum (in log-domain) over final states
        log_prob = -float('inf')
        for s in range(N):
            log_prob = np.logaddexp(log_prob, log_alpha[T-1, s])

        return log_prob


def load_digit_data(data_dir):
    """Loads WAV files from the specified directory, using filenames as labels."""
    data = []
    labels = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".wav"):
            filepath = os.path.join(data_dir, filename)
            try:
                y, sr = librosa.load(filepath, sr=None)
                data.append(y)
                label = filename[0]
                labels.append(label)
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
                continue
    return data, labels


train_directory = "./recordings/voices/digits/train" 
train_audio_data, train_labels = load_digit_data(train_directory)
test_directory = "./recordings/voices/digits/test"
test_audio_data, test_labels = load_digit_data(test_directory)


training_features = []
test_features = []
digits_str = [str(i) for i in range(10)] # Digit labels as strings

print("Computing training features...")
for digit_label_str in digits_str:
    digit_train_audio = [train_audio_data[i] for i, label in enumerate(train_labels) if label == digit_label_str] # Get audio for each digit
    digit_features = []
    for audio in digit_train_audio:
        features = compute_mfcc(audio, sr=16000).T 
        digit_features.append(features)
    training_features.append(digit_features)

print("Computing test features...")
for digit_label_str in digits_str:
    digit_test_audio = [test_audio_data[i] for i, label in enumerate(test_labels) if label == digit_label_str]
    digit_features = []
    for audio in digit_test_audio:
        features = compute_mfcc(audio, sr=16000).T
        digit_features.append(features)
    test_features.append(digit_features)


hmms = {}
print("Training HMMs...")
for digit_index, digit_name in enumerate(digits):
    print(f"Training HMM for digit: {digit_name}")
    digit_training_data = training_features[digit_index][:5] # Use first 5 recordings for training as per problem 2 instruction for training recordings.
    hmm = HMM_SingleGaussian(num_states=NUM_STATES, feature_dim=FEATURE_DIM)
    hmm.segmental_kmeans_train(digit_training_data)
    hmms[digit_name] = hmm
print("HMM training complete.")

correct_predictions = 0
total_predictions = 0

print("Performing Recognition...")
for digit_index, digit_name in enumerate(digits):
    print(f"Predicting {digit_name}")
    for test_sequence in test_features[digit_index][:5]: 
        log_likelihoods = {}
        for model_digit_name in digits:
            model = hmms[model_digit_name]
            log_likelihoods[model_digit_name] = model.log_likelihood(test_sequence)test_features

        predicted_digit_name = max(log_likelihoods, key=log_likelihoods.get) # Digit with highest likelihood
        print(predicted_digit_name)
        if predicted_digit_name == digit_name:
            correct_predictions += 1
        total_predictions += 1

accuracy = correct_predictions / total_predictions
print(f"Recognition Accuracy: {accuracy * 100:.2f}%")