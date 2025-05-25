# Stock Price Sentiment Analysis: Logistic Regression and LSTM Approach

## Overview

This project analyzes the sentiment of stock-related text data using both classical machine learning and deep learning techniques. It is based on a public dataset containing news headlines labeled with sentiment (positive, neutral, negative). The primary goal is to predict sentiment from news text which can potentially help in stock movement analysis.

## Dataset

The dataset used is from Kaggle:  
**Stock Market Sentiment Analysis Dataset**  
It contains stock-related news articles or headlines along with corresponding sentiment labels:  
- `positive`  
- `neutral`  
- `negative`

The file used: `stock_sentiment.csv`

### Data Columns Used:
- `text`: The news headline or article snippet.
- `sentiment`: Target label indicating sentiment class.

## Methodology

The project follows the steps below:

### 1. Data Preprocessing

- Removed URLs, mentions, hashtags, punctuation, and numeric digits.
- Converted text to lowercase.
- Removed missing values.
- Cleaned text was stored in a new column `clean_text`.

### 2. Feature Extraction

- For Logistic Regression: Used `TfidfVectorizer` (TF-IDF) to convert cleaned text into numerical vectors.
- For LSTM: Used tokenization, padded sequences, and embedded text via Keras `Embedding` layer.

### 3. Model 1 - Logistic Regression

- A baseline model using TF-IDF and Logistic Regression.
- The data was split into 80% training and 20% test.
- Evaluated using accuracy, precision, recall, F1-score, and confusion matrix.

**Results (Logistic Regression):**
- Accuracy: ~83%
- The model performed well on distinguishing between neutral and negative sentiment, with some confusion between positive and neutral.

### 4. Model 2 - LSTM (Deep Learning)

- Used Keras/TensorFlow to build a Sequential LSTM model:
  - Embedding layer
  - LSTM layer (128 units)
  - Dense output layer with softmax activation
- Text was tokenized and padded to ensure consistent input shapes.
- Used sparse categorical crossentropy loss and Adam optimizer.
- Trained over 5 epochs.

**Results (LSTM):**
- Accuracy: ~86% on test set
- LSTM outperformed the logistic regression model by better capturing long-range dependencies in the text.
- It showed better performance especially on longer or more complex phrases.

## Evaluation Metrics Used

- Accuracy Score
- Classification Report (Precision, Recall, F1)
- Confusion Matrix (Visualized via heatmap)

## Summary

| Model               | Accuracy | Notes                                      |
|--------------------|----------|--------------------------------------------|
| Logistic Regression| ~83%     | Fast, interpretable, good baseline         |
| LSTM (Deep Learning)| ~86%     | Better for longer text, more expressive    |

## Future Work

- Integrate FinBERT (financial-specific BERT model) for better contextual understanding.
- Combine sentiment predictions with historical stock price data to study predictive power.
- Use attention-based transformers for more advanced analysis.

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib, seaborn
- TensorFlow / Keras

## References

- Dataset: https://www.kaggle.com/datasets/sbhatti/stock-market-sentiment-analysis-dataset
- Keras documentation: https://keras.io
- scikit-learn documentation: https://scikit-learn.org

