# Model Selection and Hyperparameter Tuning

## Recommended workflow

### 1. Establish a baseline

Start with the model already added in `Pipeline.ipynb`:

- Embedding dimension: 128
- LSTM hidden dimension: 128
- Learning rate: `1e-3`
- Batch size: 64
- Maximum length: 300
- Epochs: 5

For every epoch, record:

- Training loss
- Validation loss
- Validation accuracy
- Optionally validation precision, recall, and F1 score

Because IMDb is nearly balanced, validation accuracy is meaningful. Still, use validation loss as the early-stopping/model-checkpoint metric, because it reflects prediction confidence and often detects overfitting before accuracy declines.

### 2. Keep the best epoch, not necessarily the final epoch

Training may look like this:

| Epoch | Train loss | Validation loss | Validation accuracy | Interpretation |
|---:|---:|---:|---:|---|
| 1 | 0.59 | 0.47 | 0.79 | Learning |
| 2 | 0.38 | 0.36 | 0.84 | Improving |
| 3 | 0.25 | 0.34 | 0.85 | Best checkpoint |
| 4 | 0.17 | 0.38 | 0.84 | Beginning to overfit |
| 5 | 0.11 | 0.44 | 0.83 | Overfitting |

In this example, select the model from epoch 3, rather than the final epoch.

Use early stopping:

1. Track the lowest validation loss.
2. Save a copy of the model state whenever validation loss improves.
3. Stop after validation loss fails to improve for 2–3 consecutive epochs.
4. Restore the saved best state before comparing that configuration with another one.

Do not choose the model based on training accuracy or training loss.

### 3. Tune a small number of high-impact settings

Do not test every possible combination. Start with a limited, deliberate search.

| Hyperparameter | Good values to try | Why it matters |
|---|---|---|
| Learning rate | `1e-3`, `3e-4`, `1e-4` | Usually the highest-impact training setting |
| Embedding dimension | 100, 128, 200 | Controls token-representation capacity |
| LSTM hidden dimension | 64, 128, 256 | Controls sequence-model capacity |
| Dropout | 0.2, 0.3, 0.5 | Reduces overfitting |
| Weight decay | 0, `1e-5`, `1e-4` | Mild regularization |
| Maximum length | 200, 300, 400 | Balances review context against speed |
| Vocabulary size | 10,000, 15,000, 20,000 | Controls coverage versus model size |

Tune them in this order:

1. Learning rate
2. Model size: embedding and hidden dimensions
3. Regularization: dropout and weight decay
4. Input choices: maximum length and vocabulary size

Keep all other settings fixed while testing one group. For example, find the best learning rate before changing LSTM size. This gives results you can understand and avoids an unnecessarily large experiment grid.

### 4. Compare models fairly

For each candidate configuration:

- Start from newly initialized weights.
- Train using the exact same `X_train` / `y_train`.
- Validate using the exact same `X_val` / `y_val`.
- Use the same maximum training epoch limit and early-stopping rule.
- Record the best validation result, not merely the last epoch.
- Keep a results table.

Example:

| Model | Embedding | Hidden | Learning rate | Dropout | Best val loss | Best val accuracy | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline LSTM | 128 | 128 | 0.001 | 0.0 | 0.341 | 0.851 | 3 |
| Smaller LSTM | 100 | 64 | 0.001 | 0.2 | 0.356 | 0.842 | 4 |
| Regularized LSTM | 128 | 128 | 0.0003 | 0.3 | 0.328 | 0.858 | 6 |

Select the configuration with the best validation loss, using validation accuracy/F1 as a sanity check. If two configurations are essentially tied, choose the smaller or faster model.

### 5. Add dropout correctly

Your current LSTM has one layer. PyTorch’s `nn.LSTM(..., dropout=...)` only applies dropout between LSTM layers, so it has no effect when `num_layers=1`.

For the current single-layer model, add a separate dropout layer before classification:

```python
self.dropout = nn.Dropout(0.3)
```

Then in `forward`:

```python
return self.classifier(self.dropout(hidden[-1])).squeeze(1)
```

That is the smallest useful regularization change for this model.

### 6. Consider one simpler model as a baseline

An LSTM is reasonable, but model selection is stronger if you compare it to one simple alternative:

- Embedding + masked mean pooling + linear classifier

If the simpler model performs similarly to the LSTM on validation data, prefer it: it trains faster and is easier to maintain. Only move to a more complex architecture—such as a bidirectional LSTM—if validation metrics justify it.

Do not introduce several architectures at once. Compare:

1. Masked mean-pooling baseline
2. Current LSTM
3. Optionally a bidirectional LSTM if the regular LSTM clearly benefits from sequential context

### 7. Freeze the decision, then evaluate once on test data

After choosing:

- Architecture
- Hyperparameters
- Best epoch or early-stopping policy
- Decision threshold

Stop touching validation-based choices.

Then:

1. Rebuild or retrain the selected model.
2. Optionally train on the combined training + validation data for the chosen number of epochs.
3. Run it on `test_loader` once.
4. Report final accuracy, precision, recall, F1, and a confusion matrix.

Never use test metrics to choose a learning rate, epoch count, vocabulary size, threshold, or architecture. If you do, the test score is no longer an unbiased final result.

## Practical recommendation for this project

Use this initial tuning sequence:

1. Train the existing LSTM with learning rates `1e-3`, `3e-4`, and `1e-4`.
2. Add `Dropout(0.3)` and compare it to no dropout.
3. Compare hidden dimensions 64, 128, and 256.
4. Keep the best validation configuration.
5. Compare it once against an embedding + masked-mean-pooling baseline.
6. Lock the winner and evaluate the test set once.

Also set a fixed random seed before every experiment, otherwise differences may be caused by random initialization or shuffled batches rather than a better configuration.

I did not run tuning trials or measure model performance; this is a validation protocol tailored to the existing notebook. Risk: changing `MAX_LEN` or vocabulary size affects the data-loader setup, so rebuild those before each corresponding comparison rather than changing them mid-run.
