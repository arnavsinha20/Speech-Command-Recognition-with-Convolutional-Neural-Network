🎙️ Speech Recognition CNN Model
This project implements a Convolutional Neural Network (CNN) for a simple speech command recognition task using TensorFlow and the mini_speech_commands dataset.

The goal is to classify short audio clips into one of eight distinct speech commands.

✨ Project Highlights
Task: Multi-class classification of spoken words.

Model: Sequential CNN with two Conv2D layers and Max Pooling.

Features: Mel-Frequency Cepstral Coefficients (MFCCs) extracted from raw audio.

Target Commands (8 Classes): 'stop', 'go', 'left', 'yes', 'no', 'right', 'down', and 'up'.

💾 Dataset and Preprocessing
Dataset
This model uses the mini_speech_commands dataset, a small, readily available subset of the Google Speech Commands Dataset.

Total Samples: 8,000 audio files.

Split: 80% Train (6,400), 10% Validation (800), 10% Test (800).

Feature Extraction
MFCCs are used as the input features for the CNN. This converts the raw audio signal into a fixed-size spectral representation suitable for image-like processing.

Parameter

Value

Feature Type

MFCCs

Sampling Rate (sr)

16000 Hz

Number of MFCCs

40

Input Shape

(40, 98, 1) - (n_mfcc, max_time_steps, channels)

🧠 Model Architecture
The CNN architecture is defined below. The input shape expects the padded MFCC features.

Layer Type

Filters/Units

Kernel Size/Dropout Rate

Activation

Input

N/A

(40, 98, 1) Shape

N/A

Conv2D

64

(3, 3)

relu

MaxPooling2D

N/A

(2, 2)

N/A

Dropout

N/A

0.3

N/A

Conv2D

64

(3, 3)

relu

MaxPooling2D

N/A

(2, 2)

N/A

Flatten

N/A

N/A

N/A

Dense

128

N/A

relu

Dropout

N/A

0.5

N/A

Dense (Output)

8

N/A

softmax

Compilation Details:

Optimizer: adam

Loss: sparse_categorical_crossentropy

🛠️ Setup and Installation
Prerequisites
You need Python 3 and the following dependencies.

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
.\venv\Scripts\activate   # On Windows

Installation
Install all required packages from the requirements.txt file (provided separately). The key dependencies are TensorFlow and librosa.

pip install -r requirements.txt

(The installed TensorFlow version is 2.19.0 as used in the original notebook.)

Running the Code
Clone the repository (if applicable).

Ensure the notebook file (SpeechRecognitionCNNModel.ipynb) is present.

Launch the Jupyter Notebook server:

jupyter notebook SpeechRecognitionCNNModel.ipynb

Run all cells in the notebook. The script will automatically handle:

Downloading and extracting the dataset.

Pre-processing and feature extraction.

Training the CNN model.