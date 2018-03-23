################################################################################
# COMPGI10 Bioinformatics Mini-Project - Alexander Gilbert (14057716)

# DESCRIPTION: Protein subcellular location prediction models
#              Choose a model to train by entering the appropriate index number
#              in the my_choice variable; this index number corresponds to a
#              model in the model_choices list.

model_choices = ['Logistic Regression',
                 'Fully-Connected Neural Network',
                 'Convolutional Neural Network',
                 'Convolutional Recurrent Neural Network',
                 'Merged Network']

my_choice = 4 # <<< Choose an index 0-4, corresponding to a model in model_choices

print('Model chosen for training: ', model_choices[my_choice])

################################################################################


#########################
#   PRELIMINARY SETUP   #
#########################
# 1. Import relevant libraries
# 2. Define amino acids and their properties.
# 3. Define functions for creation of hand-crafted features
#    and representation of raw sequences.


# Import relevant libraries.
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from imblearn.over_sampling import RandomOverSampler
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LSTM, GRU, Bidirectional, \
    Merge, Average, Concatenate, Activation, Conv1D, Flatten, MaxPooling1D, TimeDistributed, Add,\
    Embedding, PReLU, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import SGD, Adam, RMSprop

# Define amino acid alphabet (IUPAC).
alphabet = np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                     'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])

# Define amino acid average isotopic molecular weights.
alphabet_mw = np.array([71.0788, 103.1388, 115.0886, 129.1155, 147.1766,
                        57.0519, 137.1411, 113.1594, 128.1741, 113.1594,
                        131.1926, 114.1038, 97.1167, 128.1307, 156.1875,
                        87.0782, 101.1051, 99.1326, 186.2132, 163.1760])

# Define amino acid hydrophobicity (pH 2).
alphabet_hydro = np.array([47, 52, -18, 8, 92, 0, -42, 100, -37, 100,
                           74, -41, -46, -18, -26, -7, 13, 79, 84, 49])

# Define BLOSUM62.
alphabet_blosum62 = np.array([[4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2],
                              [0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],
                              [-2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3],
                              [-1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -2, 0, -1, 2, 0, 0, -1, -2, -3, -2],
                              [-2, -2, -3, -3, 6, -3, -1, 0, -3, 0, 0, -3, -4, -3, -3, -2, -2, -1, 1, 3],
                              [0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -3, 0, -2, -2, -2, 0, -2, -3, -2, -3],
                              [-2, -3, -1, 0, -1, -2, 8, -3, -1, -3, -2, 1, -2, 0, 0, -1, -2, -3, -2, 2],
                              [-1, -1, -3, -3, 0, -4, -3, 4, -3, 2, 1, -3, -3, -3, -3, -2, -1, 3, -3, -1],
                              [-1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -1, 0, -1, 1, 2, 0, -1, -2, -3, -2],
                              [-1, -1, -4, -3, 0, -4, -3, 2, -2, 4, 2, -3, -3, -2, -2, -2, -1, 1, -2, -1],
                              [-1, -1, -3, -2, 0, -3, -2, 1, -1, 2, 5, -2, -2, 0, -1, -1, -1, 1, -1, -1],
                              [-2, -3, 1, 0, -3, 0, 1, -3, 0, -3, -2, 6, -2, 0, 0, 1, 0, -3, -4, -2],
                              [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2, 7, -1, -2, -1, -1, -2, -4, -3],
                              [-1, -3, 0, 2, -3, -2, 0, -3, 1, -2, 0, 0, -1, 5, 1, 0, -1, -2, -2, -1],
                              [-1, -3, -2, 0, -3, -2, 0, -3, 2, -2, -1, 0, -2, 1, 5, -1, -1, -3, -3, -2],
                              [1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 1, -1, 0, -1, 4, 1, -2, -3, -2],
                              [0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 1, 5, 0, -2, -2],
                              [0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, -2, -2, -3, -2, 0, 4, -3, -1],
                              [-3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, 2],
                              [-2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1, 2, 7]])

# Define HSDM.
alphabet_hsdm = np.array([[5.5, 0.45, -2.38, -0.47, -2.42, 0.63, -3.01, -1.72, -1.22, -1.09, 0.16, -1.77, -1.11, -2.16, -2.24, 1.27, 0.6, 0.16, -2.61, -4.22],
                          [0.45, 19.05, -6.98, -4.7, 1.72, -5.7, -5.95, -0.13, -6.65, -0.82, 3.5, -6.53, -6.7, -2.47, -6.29, 1.08, -1.89, 1.32, -3.01, -0.44],
                          [-2.38, -6.98, 11.01, 2.41, -5.06, -3.91, 0.32, -6.18, 1.53, -7.41, -7.88, 4.07, 0.81, 1.1, -0.33, 2.34, -1.36, -6.1, -5.63, -3.85],
                          [-0.47, -4.7, 2.41, 8.43, -4.44, -1.8, -1.29, -5.89, 3.08, -5.62, -3.94, -0.39, -.043, 3.16, 2.83, 0.42, -0.61, -4.23, -6.28, -4.5],
                          [-2.42, 1.72, -5.06, -4.44, 9.14, -7.46, 0.25, 2.3, -6.19, 3.9, 2.66, -6.22, -2.96, -5.54, -4.36, -5.03, -4.0, 0.52, 6.49, 5.38],
                          [0.63, -5.7, -3.91, -1.8, -7.46, 11.64, -1.24, -8.58, -1.82, -6.55, -5.29, 1.16, -1.79, -0.24, -3.39, 0.63, -2.24, -5.32, -4.77, -4.34],
                          [-3.01, -5.95, 0.32, -1.29, 0.25, -1.24, 15.72, -4.44, -0.17, -2.49, -3.66, 1.77, -3.55, -2.24, 0.7, -2.38, -1.14, -1.63, -5.71, 1.17],
                          [-1.72, -0.13, -6.18, -5.89, 2.3, -8.58, -4.44, 6.74, -4.82, 3.86, 2.94, -5.78, -4.04, -3.26, -3.93, -4.67, -3.03, 5.23, -0.26, -0.08],
                          [-1.22, -6.65, 1.53, 3.08, -6.19, -1.82, -0.17, -4.82, 8.23, -5.91, -5.47, 1.64, -1.21, 3.24, 3.89, -0.27, -0.37, -3.57, -5.45, -4.03],
                          [-1.09, -0.82, -7.41, -5.62, 3.9, -6.55, -2.49, 3.86, -5.91, 6.38, 4.32, -5.64, -2.88, -4.56, -2.83, -6.22, -2.4, 2.28, -0.58, 1.81],
                          [0.16, 3.5, -7.88, -3.94, 2.66, -5.29, -3.66, 2.94, -5.47, 4.32, 10.21, -4.67, -2.02, -1.76, -1.43, -3.92, -5.18, 1.18, 4.28, -4.95],
                          [-1.77, -6.53, 4.07, -0.39, -6.22, 1.16, 1.77, -5.78, 1.64, -5.64, -4.67, 10.0, -3.23, 1.42, 0.24, 1.54, 1.14, -5.65, -6.29, -0.93],
                          [-1.11, -6.7, 0.81, -0.43, -2.96, -1.79, -3.55, -4.04, -1.21, -2.88, -2.02, -3.23, 13.32, 1.3, 1.31, -1.28, 2.44, -2.31, -11.46, -7.41],
                          [-2.16, -2.47, 1.1, 3.16, -5.54, -0.24, -2.24, -3.26, 3.24, -4.56, -1.76, 1.42, 1.3, 7.85, -0.74, 2.59, 1.08, -4.97, -4.3, -1.73],
                          [-2.24, -6.29, -0.33, 2.83, -4.36, -3.39, 0.7, -3.93, 3.89, -2.83, -1.43, 0.24, 1.31, -0.74, 8.59, -0.5, 0.34, -3.8, 1.02, -1.01],
                          [1.27, 1.08, 2.34, 0.42, -5.03, 0.63, -2.38, -4.67, -0.27, -6.22, -3.92, 1.54, -1.28, 2.59, -0.5, 6.35, 3.09, -2.69, -4.44, -4.17],
                          [0.6, -1.89, -1.36, -0.61, -4.0, -2.24, -1.14, -3.03, -0.37, -2.4, -5.18, 1.14, 2.44, 1.08, 0.34, 3.09, 6.33, -0.23, -3.55, -2.92],
                          [0.16, 1.32, -6.1, -4.23, 0.52, -5.32, -1.63, 5.23, -3.57, 2.28, 1.18, -5.65, -2.31, -4.97, -3.8, -2.69, -0.23, 5.28, -2.13, 0.66],
                          [-2.61, -3.01, -5.63, -6.28, 6.49, -4.77, -5.71, -0.26, -5.45, -0.58, 4.28, -6.29, -11.46, -4.3, 1.02, -4.44, -3.55, -2.13, 18.08, 6.79],
                          [-4.22, -0.44, -3.85, -4.5, 5.38, -4.34, 1.17, -0.08, -4.03, 1.81, -4.95, -0.93, -7.41, -1.73, -1.01, -4.17, -2.92, 0.66, 6.79, 10.92]])

# Define utility functions that are used for creation of
#   hand-crafted features and sequence representation:

# Count the number of each amino acid in a sequence.
def protein_aa_count(sequence):
    count = np.zeros(20)
    for idx, letter in enumerate(alphabet):
        count[idx] = sequence.count(letter)
    return count

# Count the number of each dipeptide in a sequence.
def protein_dp_count(sequence):
    count = np.zeros(400)
    idx = 0
    for letter_1 in alphabet:
        for letter_2 in alphabet:
            num_dp = start = 0
            search = True
            while search == True:
                start = sequence.find(letter_1 + letter_2, start) + 1
                if start > 0:
                    num_dp += 1
                else:
                    search = False
                    count[idx] = num_dp
                    idx += 1
    return count

# Convert an amino acid to one-hot representation.
def letter_one_hot(aa):
    one_hot = np.zeros(20)
    for idx, letter in enumerate(alphabet):
        if aa == letter:
            one_hot[idx] = 1
            return one_hot

# Convert an entire protein to one-hot representation.
def protein_one_hot(sequence):
    one_hot_seq = np.zeros((len(sequence), 20))
    for idx, aa in enumerate(sequence):
        one_hot_seq[idx, :] = letter_one_hot(aa)
    return one_hot_seq

# Convert an amino acid to BLOSUM62 representation.
def letter_blosum62(aa):
    for idx, letter in enumerate(alphabet):
        if aa == letter:
            return alphabet_blosum62[idx, :]

# Convert an entire protein to BLOSUM62 representation.
def protein_blosum62(sequence):
    blosum62_seq = np.zeros((len(sequence), 20))
    for idx, aa in enumerate(sequence):
        blosum62_seq[idx, :] = letter_blosum62(aa)
    return blosum62_seq

# Convert an amino acid to HSDM representation.
def letter_hsdm(aa):
    for idx, letter in enumerate(alphabet):
        if aa == letter:
            return alphabet_hsdm[idx, :]

# Convert an entire protein to HSDM representation.
def protein_hsdm(sequence):
    hsdm_seq = np.zeros((len(sequence), 20))
    for idx, aa in enumerate(sequence):
        hsdm_seq[idx, :] = letter_hsdm(aa)
    return hsdm_seq

# Convert an amino acid to hydrophobicity representation.
def letter_hydro(aa):
    for idx, letter in enumerate(alphabet):
        if aa == letter:
            hydro = alphabet_hydro[idx]
            return hydro

# Convert an entire protein to hydrophobicity representation.
def protein_hydro(sequence):
    hydro_seq = np.zeros((len(sequence)))
    for idx, aa in enumerate(sequence):
        hydro_seq[idx] = letter_hydro(aa)
    return hydro_seq

# Define the functions for creation of hand-crafted features:

# Hand-crafted feature (dim: 1): sequence length
def protein_length(sequence):
    return len(sequence)

# Hand-crafted feature (dim: 20): amino acid composition
def protein_aa_comp(sequence):
    return protein_aa_count(sequence)/protein_length(sequence)

# Hand-crafted feature (dim: 400): dipeptide composition
def protein_dp_comp(sequence):
    return protein_dp_count(sequence)/(protein_length(sequence) - 1)

# Hand-crafted feature (dim: 1): molecular weight
def protein_mw(sequence):
    return np.sum(protein_aa_count(sequence) * alphabet_mw) + 18.01528

# Hand-crafted feature (dim: 1): isoelectric point
def protein_pi(sequence):
    return ProteinAnalysis(sequence).isoelectric_point()

# Hand-crafted feature (dim: 1): aromaticity
def protein_aromaticity(sequence):
    return ProteinAnalysis(sequence).aromaticity()

# Hand-crafted feature (dim: 1): instability
def protein_instability(sequence):
    return ProteinAnalysis(sequence).instability_index()

# Hand-crafted feature (dim: 1): gravy
def protein_gravy(sequence):
    return ProteinAnalysis(sequence).gravy()

# Hand-crafted feature (dim: 3): secondary structure (helix, turn, sheet)
def protein_secondary_structure(sequence):
    return ProteinAnalysis(sequence).secondary_structure_fraction()


#########################
#  DATA PREPROCESSING   #
#########################
# 1. Set up data parameters and arrays.
# 2. Import data, create hand-crafted features, create sequence representations
# 3. Shuffle the data.
# FIVE-FOLD CROSS-VALIDATION LOOP BEGINS AT THIS POINT
# 4. Generate training/test split.
# 5. Oversample training and test sets.
# 6. Shuffle training set.
# 7. Standardise training and test sets.

np.random.seed(42)

# Set data point parameters.
num_examples = 9222
num_labels = 4

# Set up label array.
Y_data = np.zeros((num_examples, num_labels)) # one-hot
Y_data_decimal = np.zeros((num_examples)) # integer (for oversampling function)

# Set up input array for hand-crafted features.
num_features = 67 # number of hand-crafted features
X_data = np.zeros((num_examples, num_features))

# Set up input array for sequence representation.
num_seq_steps = 140 # number of amino acids used for raw sequence representation
num_seq_features = 61 # number of features for raw sequence representation
X_data_seq = np.zeros((num_examples, num_seq_steps, num_seq_features))


# Import data, create hand-crafted features, and create sequence representations.
print('Importing data, creating features, and creating sequence represenations...')
example = 0
decimal_label = 0
for label, data_file in enumerate(['cyto', 'mito', 'nucleus', 'secreted']):
    fasta_sequences = SeqIO.parse(open(data_file + '.fasta'), 'fasta')
    decimal_label += 1
    for fasta_sequence in fasta_sequences:
        protein_sequence = str(fasta_sequence.seq)

        # Remove non-specific AA codes (very few are actually present in this dataset)
        protein_sequence = protein_sequence.replace('B', '')
        protein_sequence = protein_sequence.replace('J', '')
        protein_sequence = protein_sequence.replace('O', '')
        protein_sequence = protein_sequence.replace('U', '')
        protein_sequence = protein_sequence.replace('X', '')
        protein_sequence = protein_sequence.replace('Z', '')

        # Hand-crafted features 1-20: global amino acid composition
        X_data[example, 0:20] = protein_aa_comp(protein_sequence)

        # Hand-crafted features 21-60: local amino acid composition
        #                              first 50 and last 50 amino acids
        if len(protein_sequence) < 100:
            if len(protein_sequence) % 2 == 0:
                X_data[example, 20:40] = protein_aa_comp(protein_sequence[:(len(protein_sequence)) // 2])
                X_data[example, 40:60] = protein_aa_comp(protein_sequence[(len(protein_sequence)) // 2:])
            else:
                X_data[example, 20:40] = protein_aa_comp(protein_sequence[:(len(protein_sequence)) // 2])
                X_data[example, 40:60] = protein_aa_comp(protein_sequence[(len(protein_sequence)) // 2 + 1:])
        else:
            X_data[example, 20:40] = protein_aa_comp(protein_sequence[:50])
            X_data[example, 40:60] = protein_aa_comp(protein_sequence[-50:])

        # Hand-crafted feature 61: molecular weight
        X_data[example, 60] = protein_mw(protein_sequence)

        # Hand-crafted features 62: isoelectric point
        X_data[example, 61] = protein_pi(protein_sequence)

        # Hand-crafted features 63: aromaticity
        X_data[example, 62] = protein_aromaticity(protein_sequence)

        # Hand-crafted features 64: instability
        X_data[example, 63] = protein_instability(protein_sequence)

        # Hand-crafted features 65: gravy
        X_data[example, 64] = protein_gravy(protein_sequence)

        # Hand-crafted features 66: composition of amino acids in turn
        X_data[example, 65] = protein_secondary_structure(protein_sequence)[1]

        # Hand-crafted features 67: composition of amino acids in sheet
        X_data[example, 66] = protein_secondary_structure(protein_sequence)[2]

        # Raw sequence representation:
        #
        # If protein is less than 140 amino acids long:
        if len(protein_sequence) < num_seq_steps:
            # Encode sequence into one-hots.
            example_one_hot = protein_one_hot(protein_sequence)
            # Encode sequence into BLOSUM62.
            example_blosum62 = protein_blosum62(protein_sequence)
            # Encode sequence into HSDM.
            example_hsdm = protein_hsdm(protein_sequence)
            # Encode sequence into hydrophobicity.
            example_hydro = protein_hydro(protein_sequence)
            # Generate necessary padding.
            padding_left = np.zeros(((num_seq_steps - len(protein_sequence))//2, num_seq_features))
            if len(protein_sequence) % 2 == 0:
                padding_right = np.zeros(((num_seq_steps - len(protein_sequence))//2, num_seq_features))
            else:
                padding_right = np.zeros((((num_seq_steps - len(protein_sequence))//2)+1, num_seq_features))
            # Combine sequence and padding.
            example_combined = np.concatenate((example_one_hot, example_blosum62, example_hsdm, example_hydro[:, np.newaxis]), axis = 1)
            example_combined = np.concatenate((padding_left, example_combined, padding_right), axis=0)
            # Sequence representation: 20 one-hot features + 20 BLOSUM62 features + 1 hydrophobicity feature
            X_data_seq[example, :, :] = example_combined

        # Else if protein at least 140 amino acids long:
        else:
            # Sequence representation: 20 one-hot features (nineteen 0s and one 1)
            X_data_seq[example, :, 0:20] = protein_one_hot(
                protein_sequence[:(num_seq_steps//2)] + protein_sequence[-(num_seq_steps//2):])
            # Sequence representation: 20 BLOSUM62 features
            X_data_seq[example, :, 20:40] = protein_blosum62(
                protein_sequence[:(num_seq_steps//2)] + protein_sequence[-(num_seq_steps//2):])
            # Sequence representation: 20 HSDM features
            X_data_seq[example, :, 40:60] = protein_hsdm(
                protein_sequence[:(num_seq_steps//2)] + protein_sequence[-(num_seq_steps//2):])
            # Sequence representation: 1 hydrophobicity feature
            X_data_seq[example, :, 60] = protein_hydro(
                protein_sequence[:(num_seq_steps//2)] + protein_sequence[-(num_seq_steps//2):])

        # Label the example.
        Y_data[example, label] = 1
        Y_data_decimal[example] = decimal_label

        # Go to next example in the loop.
        example += 1

# Ensure no anomalies in the form of NANs are present.
X_data = np.nan_to_num(X_data)
X_data_seq = np.nan_to_num(X_data_seq)

# Shuffle the data.
print('Shuffling dataset...')
random_ordering = np.arange(num_examples)
np.random.shuffle(random_ordering)

X_data = X_data[random_ordering, :]
X_data_seq = X_data_seq[random_ordering, :, :]

Y_data = Y_data[random_ordering]
Y_data_decimal = Y_data_decimal[random_ordering]


# FIVE-FOLD CROSS-VALIDATION LOOP BEGINS AT THIS POINT

# Cross-validation training/test split indices (inclusive):
# Training ;    Test   ; Training
#  0-7377  ; 7378-9221 ;    -
#  0-5533  ; 5534-7377 ; 7378-9221
#  0-3689  ; 3690-5533 ; 5534-9221
#  0-1845  ; 1846-3689 ; 3690-9221
#    -     ;   0-1845  ; 1846-9221

# Set up arrays to store training/test accuracies at end of training each fold.
score_train = np.zeros((5,2))
score_test = np.zeros((5,2))
print('Beginning cross-fold train/test split and training...')
for fold, [a,b,c,d] in enumerate([[0,7378,9222,7378], [0,5534,7378,9222], [0,3690,5534,9222],
                                [0, 1846, 3690, 9222], [1846,0,1846,9222]]):

    # Generate training set split.
    X_train = X_data[np.r_[a:b, c:d], :]
    X_train_seq = X_data_seq[np.r_[a:b, c:d], :, :]
    Y_train = Y_data[np.r_[a:b, c:d], :]
    Y_train_decimal = Y_data_decimal[np.r_[a:b, c:d]]

    # Generate test set split.
    X_test = X_data[b:c, :]
    X_test_seq = X_data_seq[b:c, :, :]
    Y_test = Y_data[b:c, :]
    Y_test_decimal = Y_data_decimal[b:c]

    # Oversample training set.
    # NOTE: the ROS function from imblearn requires the labels in integer form,
    #       not one-hot; this is why we have created the integer form earlier on.
    ROS = RandomOverSampler()
    idx_X_train_res = np.array([np.arange(np.shape(X_train)[0])]).T
    idx_X_train_res, Y_train_decimal_res = ROS.fit_sample(idx_X_train_res, Y_train_decimal)
    X_train_seq_res = X_train_seq[idx_X_train_res[:,0], :, :]
    X_train_res = X_train[idx_X_train_res[:,0], :]
    X_train_seq = 0 #save some memory
    X_train = 0 #save some memory

    # Oversample test set.
    idx_X_test_res = np.array([np.arange(np.shape(X_test)[0])]).T
    idx_X_test_res, Y_test_decimal_res = ROS.fit_sample(idx_X_test_res, Y_test_decimal)
    X_test_seq_res = X_test_seq[idx_X_test_res[:,0], :, :]
    X_test_res = X_test[idx_X_test_res[:,0], :]
    X_test_seq = 0 #save some memory
    X_test = 0 #save some memory

    # Shuffle the training set again.
    random_ordering = np.arange(np.shape(X_train_res)[0])
    np.random.shuffle(random_ordering)

    X_train_res = X_train_res[random_ordering, :]
    X_train_seq_res = X_train_seq_res[random_ordering, :, :]

    Y_train_decimal_res = Y_train_decimal_res[random_ordering]

    # Now that oversampling with imblearn ROS function is finished, we convert
    # the labels to one-hots for use in the rest of the code.
    Y_train_res = np.zeros((np.shape(X_train_res)[0], num_labels))
    Y_train_res[np.arange(np.shape(X_train_res)[0]),
                Y_train_decimal_res.astype('int')-1] = 1

    Y_test_res = np.zeros((np.shape(X_test_res)[0], num_labels))
    Y_test_res[np.arange(np.shape(X_test_res)[0]),
               Y_test_decimal_res.astype('int')-1] = 1

    # Standardise the hand-crafted feature training set and test set.
    #   NOTE: the training statistics (mean; standard deviation) are used for the
    #         test set as well, so as to prevent information leakage.
    X_train_res_mean = np.nanmean(X_train_res, 0)
    X_train_res_std = np.nanstd(X_train_res, 0)

    for j in range(np.shape(X_train_res_mean)[0]):
        if X_train_res_std[j] == 0:
            X_train_res[:, j] = 0
            X_test_res[:, j] = 0
        else:
            X_train_res[:,j] = (X_train_res[:, j] - X_train_res_mean[j])/X_train_res_std[j]
            X_test_res[:, j] = (X_test_res[:, j] - X_train_res_mean[j])/X_train_res_std[j]

    # Standardise the sequence representation training set and test set.
    #   NOTE: the training statistics (mean; standard deviation) are used for the
    #         test set as well, so as to prevent information leakage.
    X_train_seq_res_mean = np.nanmean(X_train_seq_res, 0)
    X_train_seq_res_std = np.nanstd(X_train_seq_res, 0)

    for i in range(np.shape(X_train_seq_res_mean)[0]):
        for j in range(np.shape(X_train_seq_res_mean)[1]):
            if X_train_seq_res_std[i, j] == 0:
                X_train_seq_res[:,i, j] = 0
                X_test_seq_res[:, i, j] = 0
            else:
                X_train_seq_res[:, i, j] = (X_train_seq_res[:, i, j] - X_train_seq_res_mean[i, j])/X_train_seq_res_std[i, j]
                X_test_seq_res[:, i, j] = (X_test_seq_res[:, i, j] - X_train_seq_res_mean[i, j])/X_train_seq_res_std[i, j]



    #########################
    #      MODEL SETUP      #
    #########################

    if model_choices[my_choice] == 'Logistic Regression':
        network_training_inputs = [X_train_res]
        network_testing_inputs = [X_test_res]

        # Set training hyper-parameters.
        epochs = 15
        batch_size = 64
        learn_rate = 0.001
        drop_prob = 0
        optimiser = Adam(lr=learn_rate)

        model = Sequential()

        model.add(BatchNormalization(input_shape=(num_features,)))
        model.add(Dense(4, kernel_initializer='glorot_uniform', activation='softmax'))

    if model_choices[my_choice] == 'Fully-Connected Neural Network':
        network_training_inputs = [X_train_res]
        network_testing_inputs = [X_test_res]

        # Set training hyper-parameters.
        epochs = 15
        batch_size = 64
        learn_rate = 0.001
        drop_prob = 0.5
        optimiser = Adam(lr=learn_rate)

        model = Sequential()

        model.add(BatchNormalization(input_shape=(num_features,)))
        model.add(Dense(50, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dropout(drop_prob))

        model.add(BatchNormalization())
        model.add(Dense(4, kernel_initializer='glorot_uniform', activation='softmax'))


    if model_choices[my_choice] == 'Convolutional Neural Network':
        network_training_inputs = [X_train_seq_res]
        network_testing_inputs = [X_test_seq_res]

        # Set training hyper-parameters.
        epochs = 15
        batch_size = 64
        learn_rate = 0.001
        drop_prob = 0.5
        optimiser = Adam(lr=learn_rate)

        model = Sequential()

        model.add(BatchNormalization(input_shape=(num_seq_steps, num_seq_features)))
        model.add(Conv1D(50, 5, kernel_initializer='he_uniform', activation='relu',
                           input_shape=(num_seq_steps, num_seq_features)))
        model.add(MaxPooling1D(2))
        model.add(Dropout(drop_prob))

        model.add(BatchNormalization())
        model.add(Conv1D(50, 11, kernel_initializer='he_uniform', activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(drop_prob))

        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(200, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dropout(0.5))

        model.add(BatchNormalization())
        model.add(Dense(4, kernel_initializer='glorot_uniform', activation='softmax'))

    if model_choices[my_choice] == 'Convolutional Recurrent Neural Network':
        network_training_inputs = [X_train_seq_res]
        network_testing_inputs = [X_test_seq_res]

        # Set training hyper-parameters.
        epochs = 15
        batch_size = 64
        learn_rate = 0.001
        drop_prob = 0.5
        optimiser = Adam(lr=learn_rate)

        model = Sequential()

        model.add(BatchNormalization(input_shape=(num_seq_steps, num_seq_features)))
        model.add(Conv1D(50, 5, kernel_initializer='he_uniform', activation='relu',
                           input_shape=(num_seq_steps, num_seq_features)))
        model.add(MaxPooling1D(2))
        model.add(Dropout(drop_prob))

        model.add(BatchNormalization())
        model.add(Conv1D(50, 11, kernel_initializer='he_uniform', activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(drop_prob))

        model.add(BatchNormalization())
        model.add(Bidirectional(GRU(150, kernel_initializer='he_uniform', activation='relu')))
        model.add(Dropout(drop_prob))

        model.add(BatchNormalization())
        model.add(Dense(250, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dropout(0.5))

        model.add(BatchNormalization())
        model.add(Dense(4, kernel_initializer='glorot_uniform', activation='softmax'))

    if model_choices[my_choice] == 'Merged Network':
        network_training_inputs = [X_train_seq_res, X_train_res]
        network_testing_inputs = [X_test_seq_res, X_test_res]

        # Set training hyper-parameters.
        epochs = 22
        batch_size = 64
        learn_rate = 0.001
        drop_prob = 0.75
        optimiser = Adam(lr=learn_rate)

        branch1 = Sequential()

        branch1.add(BatchNormalization(input_shape=(num_seq_steps, num_seq_features)))
        branch1.add(Conv1D(50, 5, kernel_initializer='he_uniform', activation='relu',
                           input_shape=(num_seq_steps, num_seq_features)))
        branch1.add(MaxPooling1D(2))
        branch1.add(Dropout(drop_prob))

        branch1.add(BatchNormalization())
        branch1.add(Conv1D(50, 11, kernel_initializer='he_uniform', activation='relu'))
        branch1.add(MaxPooling1D(2))
        branch1.add(Dropout(drop_prob))

        branch1.add(BatchNormalization())
        branch1.add(Bidirectional(GRU(150, kernel_initializer='he_uniform', activation='relu')))
        branch1.add(Dropout(drop_prob))

        branch2 = Sequential()
        branch2.add(Activation('linear', input_shape=(num_features,)))

        model = Sequential()
        model.add(Merge([branch1, branch2], mode='concat', concat_axis=1))

        model.add(BatchNormalization())
        model.add(Dense(300, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dropout(0.5))

        model.add(BatchNormalization())
        model.add(Dense(4, kernel_initializer='glorot_uniform', activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimiser,
                  metrics=['accuracy'])

    #########################
    #        TRAINING       #
    #########################

    history = model.fit(network_training_inputs, Y_train_res,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True,
                        verbose=1,
                        validation_data=(network_testing_inputs, Y_test_res))

    #########################
    #       EVALUATION      #
    #########################

    score_train[fold] = history.history['acc'][-1]
    score_test[fold] = history.history['val_acc'][-1]
    print('Training accuracy:', np.round(score_train[fold, 1]*100, decimals=2),'%')
    print('Test accuracy:', np.round(score_test[fold, 1]*100, decimals=2),'%')

cross_val = np.mean(score_test[:,1], 0)
print('Cross-validation accuracy: ', cross_val)