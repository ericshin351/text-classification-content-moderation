# Text Classification and Model Evaluation for Content Moderation

## 1. Project Overview

This project explores the application of Natural Language Processing (NLP) and machine learning techniques to automated content moderation.

The main objective is to build and compare different text classification models that identify toxic content in online comments.

The project will investigate three approaches with increasing levels of complexity:

1. **TF-IDF + Logistic Regression** — A traditional machine learning baseline.
2. **Word2Vec-based Classifier** — A classifier using distributed word representations.
3. **BERT-based Classifier** — A transformer-based approach for contextual text representation.

The models will be evaluated using consistent evaluation metrics to examine their classification performance and limitations.

## 2. Research Objectives

- Explore and preprocess textual data for toxicity classification.
- Perform exploratory data analysis (EDA) to understand the dataset.
- Implement a traditional machine learning baseline.
- Investigate word embedding techniques for text classification.
- Fine-tune a pretrained BERT model for toxicity detection.
- Compare model performance using appropriate evaluation metrics.
- Discuss the strengths and limitations of different approaches.

## 3. Dataset

The planned dataset is the **Jigsaw Toxic Comment Classification** dataset.

It contains online comments intended for toxicity classification research.

Dataset source: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

The dataset has not yet been downloaded or processed. Data preparation and exploratory analysis will be documented in subsequent stages.

## 4. Methodology

### Model 1: TF-IDF + Logistic Regression

- Text preprocessing and vectorization using TF-IDF.
- Logistic Regression for classification.
- Establish a baseline for subsequent comparisons.

### Model 2: Word2Vec-based Classifier

- Learn or use distributed word representations.
- Convert comments into document-level feature vectors.
- Train a classification model using the resulting representations.

### Model 3: BERT-based Classifier

- Use a pretrained BERT model.
- Fine-tune the model for toxicity classification.
- Evaluate its ability to capture contextual information.

## 5. Model Evaluation

The models will be assessed using appropriate classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC, where appropriate

Because toxicity datasets may have imbalanced class distributions, the analysis will pay particular attention to precision, recall, and F1-score rather than relying on accuracy alone.

All reported results will be obtained from actual experiments conducted in this project.

## 6. Repository Structure

```text
text-classification-content-moderation/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── reports/
│
├── .gitignore
└── README.md
```

## 7. Technologies

- Python 3.12
- JupyterLab
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- NLTK
- Gensim
- PyTorch and Transformers (planned for the BERT stage)

## 8. Environment Setup

Clone the repository:

```bash
git clone https://github.com/ericshin351/text-classification-content-moderation.git
cd text-classification-content-moderation
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the current project's core dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn nltk gensim jupyterlab
```

Launch JupyterLab:

```bash
jupyter lab
```

Additional dependencies for the BERT implementation will be introduced in the relevant project stage.

## 9. Project Status

**Current stage: Initial setup completed.**

- [x] Python environment setup
- [x] Project directory structure created
- [x] Git repository initialized
- [x] Initial project setup pushed to GitHub
- [ ] Dataset acquisition
- [ ] Exploratory Data Analysis
- [ ] TF-IDF + Logistic Regression
- [ ] Word2Vec-based Classification
- [ ] BERT-based Classification
- [ ] Model Evaluation and Comparison
- [ ] Final Report

This project is currently under development. Model implementations, experimental results, and final conclusions will be added as the work progresses.

## 10. Author

**Eric Shin**

GitHub: [@ericshin351](https://github.com/ericshin351)