# Comparative Sentiment Classification using RNN and LSTM (IMDB Dataset)

This project compares the performance of RNN and LSTM models for binary sentiment classification on the IMDB movie review dataset. The workflow includes data preprocessing, tokenization, padding, model building, training, evaluation, and error analysis.

---

## 📌 Objectives

- Build sentiment classification models using RNN and LSTM architectures.
- Compare model performance based on accuracy, loss, and gradient stability.
- Analyze the impact of preprocessing and sequence length on model behavior.
- Evaluate model strengths and weaknesses when handling long-text dependencies.

---

## 📂 Dataset

- Dataset: IMDB Movie Review Sentiment Dataset
- Total samples: 50,000 reviews
- Classes: Positive / Negative
- Balanced dataset (50:50)

---

## 🧹 Preprocessing

Steps applied during preprocessing:

- Remove duplicate reviews  
- Lowercasing  
- Remove URLs  
- Remove HTML tags  
- Remove extra whitespace  

The cleaned dataset is then used for tokenization.

---

## ✂️ Tokenization and Padding

- Vocabulary size: 20,000 most frequent words  
- OOV token used  
- Reviews are padded and truncated to a fixed sequence length of 260 tokens  
- Train/Validation/Test split: 80/10/10  

---

## 🧠 Model Architectures

Both models follow a similar pipeline:

### RNN Model
- Embedding layer  
- Bidirectional RNN  
- Dense output layer (sigmoid)  

### LSTM Model
- Embedding layer  
- Bidirectional LSTM  
- Dense output layer (sigmoid)  

Key hyperparameters:
EMBEDDING_DIM = 128  
HIDDEN_DIM = 64  
DROPOUT = 0.5  
LEARNING_RATE = 0.001  
EPOCHS = 50  

---

## 📈 Model Performance Summary

### RNN
- Validation accuracy around 78%  
- Clear signs of overfitting  
- Unstable gradients in earlier layers  
- Struggles with longer sequences  

### LSTM
- Validation accuracy around 88–90%  
- More stable validation loss  
- Gradient behavior more stable  
- Better handling of long-term dependencies  

LSTM outperforms RNN overall in this sentiment classification task.

---

## 🔍 Error and Gradient Analysis

- RNN shows more fluctuating gradients and higher error on long reviews.
- LSTM retains context better, leading to higher accuracy and fewer long-sequence misclassifications.

---

## 🛠️ Tech Stack

- Python 3  
- NumPy, Pandas  
- TensorFlow / Keras or PyTorch  
- Matplotlib, Seaborn  
