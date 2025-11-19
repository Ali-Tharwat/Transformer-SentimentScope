# SentimentScope: Custom Transformer for Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-4.0-yellow?logo=huggingface&logoColor=white)
![License](https://img.shields.io/badge/License-CC_BY--NC--ND_3.0-lightgrey)

> 🎓 **Educational Context:** 
> 
> This project was developed as the capstone project for the **Programming Transformers with PyTorch** course, part of the **Future AWS AI Scientist Nanodegree** program at **Udacity** ![Udacity](https://img.shields.io/badge/Udacity-grey?style=for-the-badge&logo=udacity&logoColor=15B8E6).

## 📌 Project Overview
**SentimentScope** is a deep learning project that implements a **Decoder-only Transformer architecture from scratch** to perform binary sentiment analysis on the IMDB movie reviews dataset.

Unlike standard implementations that rely on pre-built classification heads, this project constructs the Transformer blocks, attention mechanisms, and classification layers manually in PyTorch. It effectively adapts a generative architecture for a discriminative task using global mean pooling.

## 🚀 Key Features
* **Custom Architecture:** A purely PyTorch-based implementation of a Decoder Transformer (`DemoGPT`) modified for binary classification.
* **Global Mean Pooling:** Aggregates sequence embeddings across the time dimension to generate a single sentiment vector.
* **Efficient Tokenization:** Integrates the `bert-base-uncased` tokenizer for robust subword handling.
* **Custom Data Pipeline:** Implements optimized `Dataset` and `DataLoader` classes for efficient batching and shuffling.

## 🛠️ Tech Stack
* ![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) **Python** – Core programming language.

The project is built using the following core libraries:

* ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) **PyTorch** (`torch`) – Deep learning framework for building the Transformer architecture and training loops.
* ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black) **Transformers** (`transformers`) – Used specifically for the pre-trained BERT tokenizer.
* ![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white) **Pandas** (`pandas`) – Data manipulation and DataFrame management.
* ![Numpy](https://img.shields.io/badge/Numpy-7790B5?style=for-the-badge&logo=numpy&logoColor=white) **NumPy** (`numpy`) – Numerical operations and array handling.
* ![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?style=for-the-badge&logo=matplotlib) **Matplotlib** (`matplotlib`) – Visualization of data distributions and training metrics.
## 🏗️ Model Architecture
The model (`DemoGPT`) consists of the following components:
1.  **Embeddings:** Learnable Token and Positional embeddings ($d_{model} = 128$).
2.  **Transformer Blocks:** Stacked layers containing Multi-Head Self-Attention and Feed-Forward Networks with GELU activation.
3.  **Pooling Layer:** A mean pooling operation that condenses the sequence output $(Batch, Seq, Dim)$ into $(Batch, Dim)$.
4.  **Classifier:** A linear layer mapping the pooled representation to binary logits.

## 📊 Dataset
The project utilizes the [Large Movie Review Dataset (IMDB)](https://ai.stanford.edu/~amaas/data/sentiment/).
* **Input:** 50,000 highly polar movie reviews.
* **Split:** 25,000 Training (90% Train / 10% Validation) / 25,000 Test.
* **Preprocessing:** Text cleaning and subword tokenization with a max sequence length of 128.

## 📈 Results
The model was trained for 5 epochs using the **AdamW** optimizer and **CrossEntropyLoss**.

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **77.40%** |
| Validation Accuracy | 79.08% |

*The model successfully distinguishes between positive and negative reviews with high accuracy, exceeding the project baseline of 75%.*

## 💻 Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Ali-Tharwat/Transformer-SentimentScope.git](https://github.com/Ali-Tharwat/Transformer-SentimentScope.git)
    cd Transformer-SentimentScope
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Project**
    Open the notebook to execute the training pipeline:
    ```bash
    jupyter notebook SentimentScope.ipynb
    ```

## ⚖️ License
The starter code and dataset for this project are provided by Udacity and are subject to their license terms.

* **Starter Code:** Copyright © Udacity, Inc.
* **License:** Creative Commons Attribution-NonCommercial-NoDerivs 3.0 License (CC BY-NC-ND 3.0).
* **Student Implementation:** The modifications and solution code implemented by **Ali Tharwat** are for educational purposes.

## 👤 Author
**Ali Tharwat** [![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin-white&logoColor=fff)](https://linkedin.com/in/ali-tharwat) [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ali-Tharwat)