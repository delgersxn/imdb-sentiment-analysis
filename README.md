# IMDb Sentiment Analysis with Naive Bayes

---

## Project Structure

```
imdb_sentiment/
├── data/
│   └── aclImdb/                # Place extracted IMDb dataset here
│       ├── train/pos/
│       ├── train/neg/
│       ├── test/pos/
│       └── test/neg/
├── model/                      # Trained model and vectorizer
│   ├── classifier.joblib
│   └── vectorizer.joblib
├── src/                        # Source code files
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── run_pipeline.py             # End-to-end pipeline runner
├── generate_confusion_matrix.py# Generates and saves confusion matrix
├── confusion_matrix.png        # Output plot for report
└── README.md
```

---

## Setup with Conda

### 1. Create and Activate the Environment

```bash
conda create -n imdb-sentiment python=3.10
conda activate imdb-sentiment
```

### 2. Install Required Packages

```bash
pip install pandas nltk scikit-learn joblib matplotlib seaborn
```

### 3. Download NLTK Resources

Open Python shell:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## How to Run

### 1. Run the Full Pipeline (Load → Preprocess → Train → Evaluate)

```bash
python run_pipeline.py
```

### 2. Generate and Save Confusion Matrix (PNG)

```bash
python generate_confusion_matrix.py
```

This creates a high-quality plot saved as:

```
confusion_matrix.png
```

---

## Requirements Summary

- Python 3.10+
- Conda
- Packages:
  - `pandas`
  - `nltk`
  - `scikit-learn`
  - `joblib`
  - `matplotlib`
  - `seaborn`

---
