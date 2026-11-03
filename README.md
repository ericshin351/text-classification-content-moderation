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


This project uses the **Jigsaw Toxic Comment Classification** dataset.

**Dataset source:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

The dataset contains online comments labeled according to whether they contain toxic content.

### Dataset Overview

- Total samples: 159,571
- Features used: `comment_text`
- Target variable: `toxic`
- Missing values in selected columns: 0
- Toxic comments: approximately 9.59%
- Non-toxic comments: approximately 90.41%

### Data Splitting

The dataset was divided into training and testing sets using an 80:20 stratified split.

- Training samples: 127,656
- Testing samples: 31,915
- Random state: 42
- Stratification: Enabled

The same data split was used across all three classification approaches to ensure a consistent basis for model comparison.

The dataset is not included in this repository. Users must obtain it from the original Kaggle source.

## 4. Methodology

This project implements and compares three text classification approaches with different levels of model complexity.

### Model 1: TF-IDF + Logistic Regression

A traditional machine learning approach was implemented as the baseline model.

- Text representation: TF-IDF (Term Frequency-Inverse Document Frequency).
- Maximum number of features: 50,000.
- Classifier: Logistic Regression.
- Solver: `liblinear`.
- Maximum iterations: 1,000.

TF-IDF converts text documents into numerical feature vectors based on word importance. Logistic Regression is then used to classify comments as toxic or non-toxic.

### Model 2: Word2Vec + Logistic Regression

A word embedding-based approach was implemented to investigate distributed word representations.

- Embedding method: Word2Vec.
- Embedding dimension: 100.
- Context window: 5.
- Minimum word frequency: 2.
- Training epochs: 10.
- Document representation: Mean pooling of word embeddings.
- Classifier: Logistic Regression.

Word2Vec learns dense vector representations of words. Each comment is represented by averaging the embeddings of its known words, and the resulting document vectors are used for classification.

### Model 3: DistilBERT

A pretrained transformer model was fine-tuned for binary toxicity classification.

- Pretrained model: `distilbert-base-uncased`.
- Framework: PyTorch and Hugging Face Transformers.
- Task: Binary sequence classification.
- Maximum sequence length: 128 tokens.

Unlike traditional methods, DistilBERT uses contextual representations to capture relationships between words and their surrounding text.

### Experimental Consistency

All three approaches use the same training and testing split.

The models are evaluated on the same test set to provide a consistent comparison of their classification performance.

The default classification threshold of 0.5 is used for the main model comparison.

## 5. Model Evaluation

All three models were evaluated on the same test set of 31,915 comments using a default classification threshold of 0.5.

The following metrics were used:

- **Accuracy:** Overall proportion of correctly classified comments.
- **Precision:** Proportion of comments predicted as toxic that are actually toxic.
- **Recall:** Proportion of actual toxic comments correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between toxic and non-toxic comments across different classification thresholds.

Because the dataset is imbalanced, precision, recall, and F1-score are particularly important for evaluating model performance.

### 5.1 Performance Comparison

| Metric | TF-IDF + Logistic Regression | Word2Vec + Logistic Regression | DistilBERT |
|:--|--:|--:|--:|
| Accuracy | 0.9423 | 0.9434 | **0.9698** |
| Precision | 0.6506 | 0.8076 | **0.8440** |
| Recall | **0.8594** | 0.5378 | 0.8401 |
| F1-score | 0.7406 | 0.6456 | **0.8421** |
| ROC-AUC | 0.9730 | 0.9477 | **0.9874** |

DistilBERT achieved the best overall performance, with the highest accuracy, F1-score, and ROC-AUC.

TF-IDF achieved the highest recall, indicating that it identified a slightly larger proportion of toxic comments than the other models at the default threshold.

Word2Vec achieved relatively high precision but had substantially lower recall, meaning that it was more conservative when predicting toxic comments.

### 5.2 Visual Results

The following figures provide additional insights into model performance:

- `results/figures/model_performance_comparison.png` — Comparison of the three models across evaluation metrics.
- `results/figures/baseline_roc_curve.png` — ROC curve for TF-IDF + Logistic Regression.
- `results/figures/word2vec_roc_curve.png` — ROC curve for Word2Vec + Logistic Regression.
- `results/figures/distilbert_roc_curve.png` — ROC curve for DistilBERT.
- `results/figures/baseline_pr_curve.png` — Precision-Recall curve for TF-IDF.
- `results/figures/word2vec_pr_curve.png` — Precision-Recall curve for Word2Vec.
- `results/figures/distilbert_pr_curve.png` — Precision-Recall curve for DistilBERT.
- `results/figures/word2vec_confusion_matrix.png` — Confusion matrix for Word2Vec.
- `results/figures/distilbert_confusion_matrix.png` — Confusion matrix for DistilBERT.

### 5.3 Discussion

The experimental results demonstrate that different text representation methods can lead to substantially different classification performance.

**TF-IDF + Logistic Regression** provides a strong traditional baseline. Its relatively high recall makes it useful when identifying as many toxic comments as possible is a priority.

**Word2Vec + Logistic Regression** produces dense word representations, but mean pooling may lose some word order and contextual information. Its lower recall suggests that this approach was less effective at identifying toxic comments in this experiment.

**DistilBERT** achieves the strongest overall results. Its contextual representations allow it to capture relationships between words more effectively, contributing to better overall classification performance.

For automated content moderation, the choice of classification threshold should depend on the balance between false positives and false negatives. A system that prioritizes detecting harmful content may require a different threshold from one that prioritizes avoiding incorrect moderation decisions.

### 5.4 Limitations

- The dataset has an imbalanced class distribution, which can affect model evaluation.
- Mean pooling in Word2Vec does not fully preserve word order or contextual relationships.
- The Gensim Word2Vec training process produced a low-level numerical warning during training. Although selected similarity calculations were checked, the potential effect on training results cannot be ruled out.
- - The main comparison uses a default classification threshold of 0.5. Threshold optimization should be performed using a separate validation set rather than the test set.

## 6. Repository Structure

The project is organized into the following directories:

```text
text-classification-content-moderation/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 00_python_environment_test.ipynb
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_tfidf_logistic_regression.ipynb
│   ├── 03_word2vec_logistic_regression.ipynb
│   └── 04_bert_classification.ipynb
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
├── requirements.txt
└── README.md
```

### Directory Description

- `data/raw/`: Original dataset files.
- `data/processed/`: Processed data files.
- `notebooks/`: Jupyter notebooks containing data exploration, model implementation, training, and evaluation.
- `src/`: Source code modules.
- `results/figures/`: Generated evaluation plots and visualizations.
- `results/metrics/`: Model performance metrics and comparison tables.
- `reports/`: Project reports and documentation.

## 7. Technologies

The project was developed using the following technologies and libraries.

- Python 3.12
- JupyterLab
- NumPy
- Pandas
- Scikit-learn
- Logistic Regression
- TF-IDF
- NLTK
- Gensim
- Word2Vec
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Matplotlib
- Seaborn
- Git
- GitHub

## 8. Environment Setup

### 8.1 Clone the Repository

```bash
git clone https://github.com/ericshin351/text-classification-content-moderation.git
cd text-classification-content-moderation
```

### 8.2 Create a Virtual Environment

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 8.3 Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 8.4 Obtain the Dataset

Download the Jigsaw Toxic Comment Classification dataset from Kaggle.

Place the original training file in:

```text
data/raw/
```

The dataset is not included in this repository.

### 8.5 Launch JupyterLab

```bash
jupyter lab
```

Open the notebooks in the `notebooks/` directory and execute them in numerical order.

**Note:** Fine-tuning DistilBERT requires additional computational resources. GPU acceleration is recommended where available. On Apple Silicon, PyTorch can use the Metal Performance Shaders (MPS) backend when supported.
## 9. Project Status

**Current stage: Core implementation and experimental evaluation completed.**

### Completed Tasks

- [x] Python environment setup
- [x] Project directory structure created
- [x] Git repository initialized
- [x] Dataset acquisition and preparation
- [x] Exploratory Data Analysis (EDA)
- [x] TF-IDF + Logistic Regression implementation
- [x] Word2Vec-based classification
- [x] DistilBERT fine-tuning
- [x] Model evaluation and comparison
- [x] Performance metrics and visualization
- [x] README documentation

### Remaining Improvements

- [ ] Independent validation-set-based threshold optimization
- [ ] Further investigation of the Gensim Word2Vec training warning
- [ ] Final report preparation
- [ ] Final repository review and GitHub synchronization

The project has successfully implemented and evaluated three approaches to toxic comment classification. Further improvements may focus on experimental robustness, reproducibility, and documentation.
## 10. Author

**Eric Shin**

GitHub: [@ericshin351](https://github.com/ericshin351)