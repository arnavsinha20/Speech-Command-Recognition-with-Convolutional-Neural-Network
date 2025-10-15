# 🎙️ Speech Recognition CNN Model

This project implements a **Convolutional Neural Network (CNN)** for a simple speech command recognition task using **TensorFlow** and the `mini_speech_commands` dataset.

The goal is to classify short audio clips into one of **eight distinct speech commands**.

---

## ✨ Project Highlights

- **Task:** Multi-class classification of spoken words  
- **Model:** Sequential CNN with two Conv2D layers and Max Pooling  
- **Features:** Mel-Frequency Cepstral Coefficients (MFCCs) extracted from raw audio  
- **Target Commands (8 Classes):** `stop`, `go`, `left`, `yes`, `no`, `right`, `down`, `up`  

---

## 💾 Dataset and Preprocessing

### Dataset
- **Dataset Used:** `mini_speech_commands` (subset of Google Speech Commands Dataset)  
- **Total Samples:** 8,000 audio files  
- **Split:**  
  - 80% Train (6,400 samples)  
  - 10% Validation (800 samples)  
  - 10% Test (800 samples)  

### Feature Extraction
MFCCs are used as input features for the CNN. This converts raw audio signals into a fixed-size spectral representation suitable for image-like processing.

| Parameter           | Value                  |
|--------------------|-----------------------|
| Feature Type        | MFCCs                 |
| Sampling Rate (sr)  | 16,000 Hz             |
| Number of MFCCs     | 40                    |
| Input Shape         | (40, 98, 1) `(n_mfcc, max_time_steps, channels)` |

---

## 🧠 Model Architecture

The CNN architecture expects **padded MFCC features** as input.

| Layer Type          | Filters/Units | Kernel Size/Dropout Rate | Activation |
|-------------------|---------------|-------------------------|-----------|
| Input               | N/A           | (40, 98, 1)             | N/A       |
| Conv2D              | 64            | (3, 3)                  | relu      |
| MaxPooling2D        | N/A           | (2, 2)                  | N/A       |
| Dropout             | N/A           | 0.3                     | N/A       |
| Conv2D              | 64            | (3, 3)                  | relu      |
| MaxPooling2D        | N/A           | (2, 2)                  | N/A       |
| Flatten             | N/A           | N/A                     | N/A       |
| Dense               | 128           | N/A                     | relu      |
| Dropout             | N/A           | 0.5                     | N/A       |
| Dense (Output)      | 8             | N/A                     | softmax   |

**Compilation Details:**
- **Optimizer:** `adam`  
- **Loss:** `sparse_categorical_crossentropy`  

---

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.x  
- Virtual environment (optional but recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
.\venv\Scripts\activate
