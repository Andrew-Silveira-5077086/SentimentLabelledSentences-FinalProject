# Group Participants
- Andrew Silveira - 5077086
- Edwin Lopez Castañeda - 9055061
- Vishnu Sivaraj - 9025320

# Sentiment Classification with Dimensionality Reduction

This project builds sentiment classifiers on a collection of short reviews from **Amazon**, **IMDB**, and **Yelp**, and compares classic machine-learning baselines with dimensionality-reduction techniques.

The work is implemented in a single Jupyter notebook:

- `GroupProject-Final.ipynb`

and uses the three standard text files from the **Sentiment Labelled Sentences** dataset:

- `data/amazon_cells_labelled.txt`
- `data/imdb_labelled.txt`
- `data/yelp_labelled.txt`

Each sentence is labeled as **positive (1)** or **negative (0)**.

---

## Project Overview

The notebook walks through a complete traditional NLP pipeline:

1. **Data loading & exploration**  
   - Load the three labeled review files into a single `pandas` DataFrame.  
   - Track the original source (`amazon`, `imdb`, `yelp`) in a dedicated column.  
   - Inspect dataset shape and label/source distributions.  
   - Create a **single stratified train/test split** (75% train / 25% test) on the sentiment label.

2. **Text preprocessing** (token level)  
   - Tokenization with NLTK.  
   - Lowercasing and removal of non-alphabetic tokens.  
   - Stop-word removal using NLTK’s English stop-word list.  
   - Porter stemming.  
   - Store preprocessed tokens and a joined `processed` string representation in the DataFrame.

3. **TF–IDF feature extraction**  
   - `TfidfVectorizer` with:
     - **unigrams + bigrams** (`ngram_range=(1, 2)`),
     - `max_features=5000`,
     - `min_df=2`,
     - built-in English stop-words.  
   - Transform both train and test splits into sparse TF–IDF matrices.  
   - Additionally, build a TF–IDF matrix over all preprocessed sentences to inspect the vocabulary, matrix sparsity, and cosine similarities between documents.

4. **Baseline model – Multinomial Naive Bayes (TF–IDF)**  
   - Train a **Multinomial Naive Bayes** classifier on the TF–IDF features.  
   - Evaluate on the test set using:
     - accuracy,
     - confusion matrix (visualized as a Seaborn heatmap),
     - `classification_report` (precision, recall, F1).  
   - Discuss error patterns: which class generates more **false positives (FP)** vs **false negatives (FN)** and which sentiment is slightly harder to predict.

5. **Dimensionality reduction with Truncated SVD (LSA)**  
   - Apply **TruncatedSVD** to the TF–IDF representation (LSA-style).  
   - Reduce to a low-dimensional latent space (e.g., 50 components).  
   - Inspect the **total explained variance ratio** to see how much information is preserved.

6. **Logistic Regression on SVD features**  
   - Train a **Logistic Regression** classifier on the SVD-reduced features.  
   - Evaluate with accuracy, confusion matrix, and classification report.  
   - Compare performance to the Naive Bayes baseline and discuss the trade-off between:
     - slightly lower accuracy,
     - much more compact feature representations.

7. **Dimensionality study for SVD**  
   - Sweep over a range of SVD dimensions (e.g., 50–600 components).  
   - For each `k`:
     - fit SVD,
     - transform train/test,
     - train Logistic Regression,
     - record test accuracy and explained variance.  
   - Plot **accuracy vs. number of components** and **explained variance vs. number of components** to visualize the compression vs. performance trade-off.

8. **Dimensionality reduction with PCA**  
   - Use **PCA** on appropriately scaled features to obtain another low-dimensional representation.  
   - Examine the cumulative explained variance to show how much information PCA captures from sparse TF–IDF data.

9. **Logistic Regression on PCA features**  
   - Train Logistic Regression on PCA-reduced features.  
   - Evaluate with the same metrics as the SVD model.  
   - Compare to:
     - Naive Bayes + TF–IDF,
     - Logistic Regression + SVD,  
     and discuss why PCA tends to underperform SVD on this sparse text representation.

Throughout the notebook, confusion matrices are **annotated and interpreted** so you can see not just “how accurate” a model is, but also *how* it tends to make mistakes.

---

## Repository Structure

```text
.
├── GroupProject-Final.ipynb   # Main notebook with all experiments
├── data/
│   ├── amazon_cells_labelled.txt
│   ├── imdb_labelled.txt
│   └── yelp_labelled.txt
└── README.md                  # (this file)
