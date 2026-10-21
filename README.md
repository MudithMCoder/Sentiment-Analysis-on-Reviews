# IMDb Movie Review Sentiment Analysis

A notebook-based natural language processing project that explores IMDb movie reviews, prepares text data, trains a PyTorch LSTM classifier, and evaluates its predictions for **positive** and **negative** sentiment.

> This repository contains an educational experimentation pipeline, not a packaged application. Run the notebook from top to bottom to prepare data, train a model, and produce evaluation results.

## Contents

- [Project workflow](#project-workflow)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Run the notebook](#run-the-notebook)
- [Model and evaluation](#model-and-evaluation)
- [EDA figures](#eda-figures)
- [Hyperparameter guidance](#hyperparameter-guidance)
- [Limitations and reproducibility](#limitations-and-reproducibility)

## Project workflow

The complete workflow is implemented in [`Pipeline.ipynb`](Pipeline.ipynb):

1. Load the review dataset, inspect its columns, missing values, class counts, and duplicate reviews.
2. Remove duplicate reviews and explore class balance, review lengths, text noise, word frequencies, word clouds, and class-specific bigrams.
3. Clean reviews by stripping HTML and URLs, expanding contractions, lowercasing, removing digits and non-letter characters, and normalizing whitespace. Encode `positive` as `1` and `negative` as `0`.
4. Create stratified training, validation, and test splits (70% / 15% / 15%) using split seed `42`.
5. Tokenize with spaCy, build a vocabulary from training reviews, and convert reviews into padded/truncated integer sequences.
6. Train an LSTM classifier with PyTorch, using the validation split to report per-epoch loss and accuracy.
7. Evaluate the trained model on the held-out test split. The evaluation cell prints a classification report and accuracy, precision, and F1; it also displays a confusion-matrix heatmap and ROC curve with AUC.

## Repository structure

```text
.
├── Data/
│   └── IMDB.csv
├── Docs/
│   └── Model_hyperparameters.md
├── eda_outputs/
│   ├── class_balance.png
│   ├── noise_summary.png
│   ├── top_bigrams.png
│   ├── top_words.png
│   ├── word_count_dist.png
│   ├── word_freq_distribution.png
│   └── wordclouds.png
└── Pipeline.ipynb
```

The EDA images are committed visual outputs. The notebook creates the `eda_outputs/` directory and displays its plots; rerunning it may not rewrite the committed images.

## Dataset

The notebook reads [`Data/IMDB.csv`](Data/IMDB.csv) relative to the repository root. It expects two columns:

| Column | Description |
|---|---|
| `review` | Movie review text |
| `sentiment` | Sentiment label (`positive` or `negative`) |

The saved notebook output shows 50,000 input rows with 25,000 examples for each sentiment and no missing values. Removing duplicate reviews leaves 49,582 rows. The stratified split in the saved output contains 34,707 training, 7,437 validation, and 7,438 test examples.

The repository does not document the dataset's original source or license. Verify that you have the right to use and redistribute the dataset before publishing or reusing it.

## Setup

Use Python 3.12 or another environment compatible with the dependencies below. The repository does not currently include a `requirements.txt`, lockfile, or other dependency manifest, so package versions are not pinned.

The notebook imports these third-party packages:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `beautifulsoup4`
- `contractions`
- `spacy`
- `torch`
- `wordcloud`

Install Jupyter Notebook or JupyterLab as well. One way to prepare an environment is:

```bash
python -m venv .venv
# Activate the environment (commands below are platform-specific)
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scikit-learn beautifulsoup4 contractions spacy torch wordcloud jupyter
python -m spacy download en_core_web_sm
```

Choose a PyTorch installation appropriate for your operating system and hardware; consult the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) if the default pip command does not match your setup. The notebook uses CUDA when PyTorch reports it is available, and otherwise runs on CPU.

## Run the notebook

1. Clone or download the repository and enter its root directory (the directory containing `Pipeline.ipynb` and `Data/`).
2. Activate the Python environment and install the dependencies and spaCy language model described above.
3. Start Jupyter from the repository root:

   ```bash
   jupyter notebook
   ```

   Or use `jupyter lab`.

4. Open `Pipeline.ipynb` and choose **Run All**. The relative path `Data/IMDB.csv` assumes that the notebook kernel's working directory is the repository root.

EDA figures are displayed during notebook execution. The final evaluation cell uses the held-out test data; execute it only after the preprocessing, data-loading, and training cells have completed successfully.

## Model and evaluation

The current notebook configuration is:

| Setting | Value |
|---|---:|
| Vocabulary cap | 15,000 tokens, including padding and unknown-token entries |
| Maximum sequence length | 300 tokens |
| Batch size | 64 |
| Embedding dimension | 128 |
| LSTM hidden dimension | 128 |
| Loss | Binary cross-entropy with logits |
| Optimizer | Adam, learning rate `1e-3` |
| Training epochs | 5 |
| Prediction threshold | 0.5 probability |

The model emits a score for each review. Test-set classification metrics use a 0.5 probability threshold; the ROC curve and AUC use continuous prediction probabilities. No test results are included here because the evaluation cell has no saved output in the notebook. Run it to obtain results for your execution.

The notebook's saved training output records a validation accuracy of **0.8865 at epoch 5**. This is an observed validation result from the saved notebook execution, not a test score or a guaranteed benchmark. The lowest saved validation loss occurred at epoch 4, but the current training code does not save or restore a best-validation checkpoint before evaluating the final-epoch model.

## EDA figures

The repository includes the following generated figures:

- [Class balance](eda_outputs/class_balance.png)
- [Noise summary](eda_outputs/noise_summary.png)
- [Top bigrams](eda_outputs/top_bigrams.png)
- [Top words](eda_outputs/top_words.png)
- [Word-count distribution](eda_outputs/word_count_dist.png)
- [Word-frequency distribution](eda_outputs/word_freq_distribution.png)
- [Word clouds](eda_outputs/wordclouds.png)

## Hyperparameter guidance

See [`Docs/Model_hyperparameters.md`](Docs/Model_hyperparameters.md) for recommendations on validation-based model selection, early stopping, and controlled hyperparameter comparisons. These are suggestions; those tuning experiments are not implemented as an automated search in the notebook.

## Limitations and reproducibility

- Only the data split is seeded (`random_state=42`). Model initialization and data-loader shuffling are not explicitly seeded, so training results can vary.
- There is no pinned environment/dependency manifest; compatible library versions may be needed as dependencies evolve.
- The spaCy `en_core_web_sm` language model must be installed separately.
- The notebook does not save trained weights or provide a standalone inference API. Run the notebook in order to train and evaluate the in-memory model.
- The current evaluation code uses the model at the end of the fixed five-epoch training loop; it does not restore a checkpoint selected by validation loss.
- Dataset source and licensing details are not documented in the repository.
- No automated test suite is provided.
