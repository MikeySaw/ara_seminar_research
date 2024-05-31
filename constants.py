# Author : Debanjali Biswas

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys 

"""
Constants and File Paths

"""

# Model Constants

HIDDEN_DIM1 = 1024  # Scorer hidden dimension 1
HIDDEN_DIM2 = 1024  # Scorer hidden dimension 2
OUTPUT_DIM = 1  # Classifier output_dim
DROPOUT0 = 0.5 # Dropout-rate between encoding and hidden layer 1
DROPOUT1 = 0.5 # Dropout-rate between hidden layer 1 and hidden layer 2
DROPOUT2 = 0.5 # Dropout-rate after hidden layer 2
LR = 0.00001  # Learning rate for Adam optimizer
MAX_EPOCHS = 10  # Training Epochs
CUDA_DEVICE = "cuda:0"  # GPU 
PATIENCE = 250
OPTIMIZER = "Adam" # one of the following: "Adam" (used in the original model; uses LR), "DefaultAdam" (uses default lr), "SGD" (uses default lr and exponential lr scheduler), "RMSprop" (uses default lr), "Adagrad" (uses default lr)
#####################################


# Model file paths and names

destination_folder1 = "./results1"  # Destination folder for the trained model and metrics for extended model
destination_folder2 = "./results2"  # Destination folder for the trained model and metrics for base model
destination_folder3 = "./results3"  # Destination folder for the trained model and metrics for cosine similarity baseline model
destination_folder4 = "./results4"  # Destination folder for the heuristics for naive baseline model
prediction_file = "prediction.tsv"  # Testing results


#####################################


# Data file paths and names

folder = "./data"  # Dataset folder
test_folder = "./data" # Folder with the gold data for evaluation
recipe_folder_name = "recipes"  # Folder containing recipes
alignment_file = "alignments.tsv"  # Alignment file
