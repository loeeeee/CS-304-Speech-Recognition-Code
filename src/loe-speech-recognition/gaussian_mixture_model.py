import numpy as np
import os
import librosa
from scipy.stats import multivariate_normal

from mfcc import compute_mfcc

# Assume you have functions from previous assignments to:
# 1. `compute_features(audio_data, sample_rate)`: Computes 39-dimensional features (MFCCs/cepstra, delta, double delta) from audio data and sample rate.
#    This function should also handle mean subtraction and variance normalization as required.

# --- Hyperparameters ---
NUM_STATES = 5 # Number of states in each HMM
FEATURE_DIM = 39 # Dimension of feature vectors
NUM_DIGITS = 10 # Digits 0-9
digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
NUM_TRAIN_RECORDS_PER_DIGIT = 10 # Based on the assignment description
NUM_TEST_RECORDS_PER_DIGIT = 10 # Based on the assignment description
MAX_ITERATIONS_SEGM_KMEANS = 50 # Maximum iterations for segmental K-means
NUM_MIXTURES = 4 # Number of Gaussian mixtures per state for Problem 3

# --- HMM Class for Gaussian Mixture Model (GMM) per State ---
class HMM_GMM:
    def __init__(self, num_states, feature_dim, num_mixtures):
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.num_mixtures = num_mixtures
        self.transition_probs = None      # (num_states, num_states) - Transition probabilities
        self.mixture_weights = None     # (num_states, num_mixtures) - Mixture weights for each state
        self.means = None               # (num_states, num_mixtures, feature_dim) - Means of Gaussian mixtures
        self.covariances = None         # (num_states, num_mixtures, feature_dim, feature_dim) - Covariances of Gaussian mixtures

    def initialize_params(self, training_sequences):
        """Initializes HMM parameters with GMM emissions using simple segmentation."""
        self.transition_probs = np.zeros((self.num_states, self.num_states))
        self.mixture_weights = np.ones((self.num_states, self.num_mixtures)) / self.num_mixtures # Uniform mixture weights initially
        self.means = np.zeros((self.num_states, self.num_mixtures, self.feature_dim))
        self.covariances = np.array([[[np.eye(self.feature_dim) for _ in range(self.num_mixtures)] for _ in range(self.num_states)]]) # Initialize covariances to identity

        # Initialize means by randomly picking data points from segments
        for state in range(self.num_states):
            state_data_points = []
            for seq in training_sequences:
                seq_len = len(seq)
                segment_len = seq_len // self.num_states
                start_index = state * segment_len
                end_index = (state + 1) * segment_len if state < self.num_states - 1 else seq_len
                state_data_points.extend(seq[start_index:end_index])

            if len(state_data_points) > 0:
                if len(state_data_points) < self.num_mixtures: # Handle cases with fewer data points than mixtures
                    indices = np.random.choice(len(state_data_points), size=len(state_data_points), replace=False)
                    selected_data = np.array(state_data_points)[indices]
                    for mix_idx in range(len(selected_data)):
                        self.means[state, mix_idx] = selected_data[mix_idx]
                    for mix_idx in range(len(selected_data), self.num_mixtures): # Fill remaining mixtures with mean of available data if any
                        self.means[state, mix_idx] = np.mean(state_data_points, axis=0) if state_data_points else np.zeros(self.feature_dim)
                else:
                    indices = np.random.choice(len(state_data_points), size=self.num_mixtures, replace=False)
                    self.means[state, :] = np.array(state_data_points)[indices]
            else:
                for mix_idx in range(self.num_mixtures):
                    self.means[state, mix_idx] = np.zeros(self.feature_dim) # Default to zero if no data


        # Initialize transition probabilities (left-to-right, self-loop)
        for i in range(self.num_states):
            if i < self.num_states - 1:
                self.transition_probs[i, i] = 0.5
                self.transition_probs[i, i + 1] = 0.5
            else:
                self.transition_probs[i, i] = 1.0 # Last state self-loop


    def segmental_kmeans_train(self, training_sequences, max_iterations=MAX_ITERATIONS_SEGM_KMEANS):
        """Trains GMM-HMM using segmental K-means."""
        self.initialize_params(training_sequences)

        for iteration in range(max_iterations):
            print(f"Segmental K-means Iteration: {iteration + 1}")
            new_mixture_weights = np.zeros_like(self.mixture_weights)
            new_means = np.zeros_like(self.means)
            new_covariances = np.zeros_like(self.covariances)
            new_transition_counts = np.zeros_like(self.transition_probs)
            state_counts = np.zeros(self.num_states)
            mixture_counts = np.zeros((self.num_states, self.num_mixtures)) # Count data points for each mixture
            state_assignments_all_seqs = []

            for seq in training_sequences:
                viterbi_path, mixture_assignments = self.viterbi(seq) # Viterbi now returns mixture assignments too
                state_assignments_all_seqs.append(viterbi_path)

                for t in range(len(seq)):
                    state = viterbi_path[t]
                    mixture_idx = mixture_assignments[t] # Get mixture assignment for this time frame
                    new_means[state, mixture_idx] += seq[t]
                    mixture_counts[state, mixture_idx] += 1 # Count for mixture
                    state_counts[state] += 1 # Total count for state
                    if t > 0:
                        prev_state = viterbi_path[t-1]
                        new_transition_counts[prev_state, state] += 1

            # M-step: Re-estimate parameters
            for state in range(self.num_states):
                for mix_idx in range(self.num_mixtures):
                    if mixture_counts[state, mix_idx] > 0:
                        new_means[state, mix_idx] /= mixture_counts[state, mix_idx]
                        new_mixture_weights[state, mix_idx] = mixture_counts[state, mix_idx] # Mixture weights are proportional to counts

            for state in range(self.num_states):
                if np.sum(new_mixture_weights[state, :]) > 0:
                    new_mixture_weights[state, :] /= np.sum(new_mixture_weights[state, :]) # Normalize mixture weights
                else:
                    new_mixture_weights[state, :] = self.mixture_weights[state, :].copy() # Keep old weights if no data for this state

            for seq_idx, seq in enumerate(training_sequences):
                viterbi_path, mixture_assignments = self.viterbi(seq)
                for t in range(len(seq)):
                    state = viterbi_path[t]
                    mixture_idx = mixture_assignments[t]
                    diff = seq[t] - new_means[state, mixture_idx]
                    new_covariances[state, mixture_idx] += np.outer(diff, diff)

            for state in range(self.num_states):
                for mix_idx in range(self.num_mixtures):
                    if mixture_counts[state, mix_idx] > self.feature_dim + 1:
                        new_covariances[state, mix_idx] /= mixture_counts[state, mix_idx]
                        new_covariances[state, mix_idx] += 0.01 * np.eye(self.feature_dim) # Regularization
                    else:
                        new_covariances[state, mix_idx] = self.covariances[state, mix_idx].copy() # Keep old covariance if not enough data

            # Re-estimate transition probabilities
            for i in range(self.num_states):
                row_sum = np.sum(new_transition_counts[i, :])
                if row_sum > 0:
                    self.transition_probs[i, :] = new_transition_counts[i, :] / row_sum
                else:
                    self.transition_probs[i, :] = self.transition_probs[i,:].copy()


            # Convergence check (simplified)
            prev_means = self.means.copy()
            self.mixture_weights = new_mixture_weights
            self.means = new_means
            self.covariances = new_covariances

            mean_diff = np.sum(np.abs(self.means - prev_means))
            if mean_diff < 1e-3:
                print("Converged.")
                break


    def emission_prob(self, observation, state_index):
        """Calculates emission probability for a given observation and state with GMM."""
        prob = 0
        for mix_idx in range(self.num_mixtures):
            prob += self.mixture_weights[state_index, mix_idx] * multivariate_normal.pdf(observation, self.means[state_index, mix_idx], self.covariances[state_index, mix_idx])
        return prob

    def log_emission_prob(self, observation, state_index):
        """Calculates log emission probability for GMM to avoid underflow."""
        log_prob = -float('inf')
        for mix_idx in range(self.num_mixtures):
            log_prob = np.logaddexp(log_prob, np.log(self.mixture_weights[state_index, mix_idx]) + multivariate_normal.logpdf(observation, self.means[state_index, mix_idx], self.covariances[state_index, mix_idx]))
        return log_prob


    def viterbi(self, observation_sequence):
        """Viterbi algorithm for GMM-HMM, returns best state sequence and mixture assignments."""
        T = len(observation_sequence)
        N = self.num_states

        log_delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)
        mixture_assignments = np.zeros((T,N), dtype=int) # To store mixture component assignments


        # Initialization (t=0)
        for s in range(N):
            log_delta[0, s] = self.log_emission_prob(observation_sequence[0], s)

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
                log_delta[t, j] = max_log_prob + self.log_emission_prob(observation_sequence[t], j)
                psi[t, j] = best_prev_state


        # Termination
        best_path_prob = np.max(log_delta[T-1, :])
        last_state = np.argmax(log_delta[T-1, :])

        # Backtracking - and also determine mixture assignments along the best path
        viterbi_path = [0] * T
        viterbi_path[T-1] = last_state
        current_state = last_state

        mixture_path_assignments = [0] * T # For storing mixture assignments in the best path

        # Determine mixture assignment for the last frame (frame T-1) for the last state
        emission_probs_last_frame = []
        for mix_idx in range(self.num_mixtures):
             emission_probs_last_frame.append(np.log(self.mixture_weights[last_state, mix_idx]) + multivariate_normal.logpdf(observation_sequence[T-1], self.means[last_state, mix_idx], self.covariances[last_state, mix_idx]))
        best_mixture_last_frame = np.argmax(emission_probs_last_frame)
        mixture_path_assignments[T-1] = best_mixture_last_frame


        for t in range(T-2, -1, -1):
            viterbi_path[t] = psi[t+1, viterbi_path[t+1]]
            current_state = viterbi_path[t]

            # Determine mixture assignment for frame t and current_state
            emission_probs_current_frame = []
            for mix_idx in range(self.num_mixtures):
                 emission_probs_current_frame.append(np.log(self.mixture_weights[current_state, mix_idx]) + multivariate_normal.logpdf(observation_sequence[t], self.means[current_state, mix_idx], self.covariances[current_state, mix_idx]))
            best_mixture_current_frame = np.argmax(emission_probs_current_frame)
            mixture_path_assignments[t] = best_mixture_current_frame


        return viterbi_path, mixture_path_assignments


    def log_likelihood(self, observation_sequence):
        """Calculates log-likelihood for GMM-HMM using forward algorithm."""
        T = len(observation_sequence)
        N = self.num_states
        log_alpha = np.zeros((T, N))

        # Initialization (t=0)
        for s in range(N):
            log_alpha[0, s] = self.log_emission_prob(observation_sequence[0], s)

        # Forward recursion (t=1 to T-1)
        for t in range(1, T):
            for j in range(N):
                log_sum_exp = -float('inf')
                for i in range(N):
                    log_sum_exp = np.logaddexp(log_sum_exp, log_alpha[t-1, i] + np.log(self.transition_probs[i, j]))
                log_alpha[t, j] = log_sum_exp + self.log_emission_prob(observation_sequence[t], j)

        # Termination
        log_prob = -float('inf')
        for s in range(N):
            log_prob = np.logaddexp(log_prob, log_alpha[T-1, s])
        return log_prob


# --- Data Loading Function (Provided by User - Same as before) ---
def load_digit_data(data_dir):
    """Loads WAV files from the specified directory, using filenames as labels."""
    data = []
    labels = []
    for filename in sorted(os.listdir(data_dir)): # sorted to ensure consistent order
        if filename.endswith(".wav"):
            filepath = os.path.join(data_dir, filename)
            try:
                y, sr = librosa.load(filepath, sr=None) # sr=None to preserve original sample rate
                data.append(y)  # Extend the list, more memory efficient than appending arrays repeatedly.
                label = filename[0] # Extract the first digit as the label
                labels.append(label)
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
                continue # Skip to the next file if there's an error
    return data, labels


# --- Load Data using the provided function (Same as before) ---
train_directory = "./recordings/voices/digits/train" # Adjust path if necessary
train_audio_data, train_labels = load_digit_data(train_directory)
test_directory = "./recordings/voices/digits/test" # Adjust path if necessary
test_audio_data, test_labels = load_digit_data(test_directory)


# --- Feature Extraction (using loaded audio data - Same as before) ---
training_features = []
test_features = []
digits_str = [str(i) for i in range(10)] # Digit labels as strings

print("Computing training features...")
for digit_label_str in digits_str:
    digit_train_audio = [train_audio_data[i] for i, label in enumerate(train_labels) if label == digit_label_str]
    digit_features = []
    for audio in digit_train_audio:
        features = compute_mfcc(audio, sr=16000).T # Important: Implement this function
        digit_features.append(features)
    training_features.append(digit_features)

print("Computing test features...")
for digit_label_str in digits_str:
    digit_test_audio = [test_audio_data[i] for i, label in enumerate(test_labels) if label == digit_label_str]
    digit_features = []
    for audio in digit_test_audio:
        features = compute_mfcc(audio, sr=16000).T # Important: Implement this function
        digit_features.append(features)
    test_features.append(digit_features)


# --- Training GMM-HMMs (Problem 3) ---
gm_hmms = {}
print("Training GMM-HMMs...")
for digit_index, digit_name in enumerate(digits):
    print(f"Training GMM-HMM for digit: {digit_name}")
    digit_training_data = training_features[digit_index][:5] # Using first 5 training recordings
    gmm_hmm = HMM_GMM(num_states=NUM_STATES, feature_dim=FEATURE_DIM, num_mixtures=NUM_MIXTURES) # Initialize GMM-HMM
    gmm_hmm.segmental_kmeans_train(digit_training_data)
    gm_hmms[digit_name] = gmm_hmm
print("GMM-HMM training complete.")

# --- Recognition and Accuracy Calculation (Problem 3) ---
correct_predictions_gmm = 0
total_predictions_gmm = 0

print("Performing Recognition with GMM-HMMs...")
for digit_index, digit_name in enumerate(digits):
    for test_sequence in test_features[digit_index][:5]: # Using first 5 test recordings
        log_likelihoods_gmm = {}
        for model_digit_name in digits:
            model = gm_hmms[model_digit_name]
            log_likelihoods_gmm[model_digit_name] = model.log_likelihood(test_sequence)

        predicted_digit_name_gmm = max(log_likelihoods_gmm, key=log_likelihoods_gmm.get)

        if predicted_digit_name_gmm == digit_name:
            correct_predictions_gmm += 1
        total_predictions_gmm += 1

accuracy_gmm = correct_predictions_gmm / total_predictions_gmm
print(f"GMM-HMM Recognition Accuracy: {accuracy_gmm * 100:.2f}%")