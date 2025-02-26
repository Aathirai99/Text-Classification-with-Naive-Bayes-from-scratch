# Naïve Bayes Text Classification for Sentiment Analysis

This notebook implements a Naïve Bayes classifier from scratch for sentiment analysis on the IMDB movie review dataset.

## Table of Contents
1. [Introduction](#introduction)
2. [Naïve Bayes Classification Process](#naïve-bayes-classification-process)
3. [Implementation](#implementation)
4. [Results and Evaluation](#results-and-evaluation)

## Introduction

Naïve Bayes is a probabilistic model based on Bayes' Theorem, with the key assumption that words in a document are conditionally independent given the class. This implementation applies Naïve Bayes to classify movie reviews as positive or negative.

### Bayes' Theorem

The core of the classifier is based on Bayes' Theorem:

P(Y|X) = (P(X|Y) * P(Y)) / P(X)

For text classification, we model the likelihood as:

P(Y|w1, w2, ..., wn) ∝ P(Y) * ∏(i=1 to n) P(wi|Y)

## Naïve Bayes Classification Process

1. **Loading Data**: IMDB dataset with movie reviews and sentiment labels.
2. **Preprocessing**: Remove HTML tags, URLs, non-alphanumeric characters, convert to lowercase, remove stopwords.
3. **Label Encoding**: Convert sentiment labels (positive → 1, negative → 0).
4. **Calculate Prior Probabilities**: Compute P(Y) for each class.
5. **Build Vocabulary & Word Counts**: Create a vocabulary and count word frequencies in positive and negative reviews.
6. **Naïve Bayes Classification**: Implement the classifier using Laplace smoothing and log probabilities.
7. **Model Evaluation**: Split data, run classifier, compute confusion matrix and F1 score.

## Implementation

The notebook uses Python with pandas and numpy libraries. Key steps include:

- Data loading and exploration
- Text preprocessing
- Vocabulary building
- Probability calculations
- Classification algorithm
- Performance evaluation

## Results and Evaluation

- F1 Score: 0.52
- Confusion Matrix: <br>
 [[ 119 4808] <br>
 [1577 3496]]


## Conclusion

This notebook was used to learn **Naïve Bayes from scratch** and did not focus on improving model performance.  
#### **Ways to Improve Model Performance:**  
- **Better Text Preprocessing:** Removing stopwords, stemming, and lemmatization.    
- **Handling Imbalanced Data:** Using **class weighting** or **oversampling/undersampling** methods.  
- **Feature Engineering:** Extracting **n-grams** (bigrams, trigrams) to capture word relationships.  
- **Hyperparameter Tuning:** Adjusting **Laplace smoothing** and feature selection methods.  
- **Using Word Embeddings:** Incorporating **word vectors (e.g., Word2Vec, GloVe)** for better representation.  

---

Last updated: Tuesday, February 25, 2025
