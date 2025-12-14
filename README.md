# IMDb Movie Review Sentiment Analysis

This project performs sentiment analysis on the IMDb movie reviews dataset using multiple feature engineering strategies and machine learning / deep learning models.  
The goal is to systematically compare traditional methods, neural networks, sequence models, and transfer learning approaches on the same dataset.

---
## Team Members

| Name         | NetID  |
|--------------|--------|
| Shuai He     | sh8349 |
| Max Zhang    | yz12020 |
| Shiyue Zhu   | sz4911 |
| Ningning Han | nh2942 |

---

## Project Overview

- **Task**: Binary sentiment classification (Positive / Negative)
- **Dataset**: IMDb Movie Reviews (50,000 reviews, balanced)
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - AUC (ROC)

We experiment with **five different models**, ranging from classical machine learning to deep learning and transfer learning, to understand the trade-offs between simplicity, performance, and representation power.

---

## Requirements

- Python 
- NumPy 
- Pandas 
- Scikit-learn 
- TensorFlow 
- Matplotlib 
- Seaborn 

---

## Dataset

- Source: `tensorflow.keras.datasets.imdb`
- Total samples: **50,000**
  - Training set: 25,000
  - Test set: 25,000
- Vocabulary size: **20,000**
- Labels:
  - `0`: Negative
  - `1`: Positive

---

## Feature Engineering

We design and compare three types of features:

### 1. TF-IDF Features
- Unigrams and bigrams
- Max features: 3,000
- Captures word importance across documents

### 2. Statistical Features
- Review length (number of words)
- Vocabulary richness (unique word ratio)
- Average word length
- Combined with Word2Vec features for tree-based models

### 3. Word Embeddings
- **Word2Vec** (trained on IMDb corpus)
- **GloVe (100d)** pre-trained embeddings for transfer learning
- Review-level representation obtained by:
  - Averaging embeddings (for NN)
  - Sequence modeling (for LSTM)

---

## Models

### Model 1: Logistic Regression (TF-IDF)
- Classical linear baseline
- Strong performance with sparse features
- Class-balanced training

### Model 2: Random Forest (Statistical + Word2Vec)
- Nonlinear ensemble model
- Uses handcrafted statistical features + dense embeddings

### Model 3: Neural Network (Word2Vec)
- Multi-layer fully connected network
- Batch Normalization + Dropout
- L2 regularization

### Model 4: LSTM (Sequence Model)
- Learns word order and contextual dependencies
- Trained from scratch with trainable embeddings

### Model 5: Transfer Learning (GloVe + LSTM)
- Pre-trained GloVe embeddings
- Fine-tuned embedding layer
- Best semantic representation capability

---

## Final Results

| Model                        | Accuracy | Precision | Recall | F1-score | AUC   |
|-----------------------------|----------|-----------|--------|----------|-------|
| **LogReg_TFIDF**            | **0.8782** | 0.8740 | 0.8839 | **0.8789** | **0.9510** |
| LSTM_GloVe                  | 0.8562 | **0.8962** | 0.8057 | 0.8485 | 0.9364 |
| NN_Word2Vec_Upgraded        | 0.8501 | 0.8319 | 0.8774 | 0.8541 | 0.9283 |
| LSTM                        | 0.8341 | 0.8059 | **0.8803** | 0.8414 | 0.9132 |
| RandomForest_Stat_W2V       | 0.8201 | 0.8175 | 0.8242 | 0.8208 | 0.9016 |

---

## Results Analysis

- **Logistic Regression with TF-IDF** achieves the **best overall performance**, demonstrating that strong linear baselines remain highly competitive for sentiment analysis.
- **Transfer Learning with GloVe + LSTM** improves semantic understanding but does not surpass TF-IDF, likely due to limited sequence length and dataset size.
- **Neural Network with Word2Vec** benefits from dense representations but slightly underperforms TF-IDF-based methods.
- **LSTM models** capture word order and context, showing higher recall but lower precision compared to linear models.
- **Random Forest** performs reasonably well but struggles with high-dimensional dense features.

Overall, the experiments highlight the trade-off between model complexity and performance, and show that simpler models can outperform deep architectures on well-structured text classification tasks.

---

## Experimental Setup

- Optimizer: Adam
- Loss function: Binary Cross-Entropy
- Regularization: Dropout, L2
- Early stopping based on validation loss
- Random seed fixed for reproducibility

---

