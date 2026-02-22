Collecting workspace information# Gujarati Grammar Correction — README

---

## 📖 Overview

This project implements a **Gujarati spell and grammar correction system** using a sequence-to-sequence Transformer model. It takes grammatically incorrect or misspelled Gujarati sentences as input and outputs their corrected versions.

---

## 🗂️ Project Structure

```
├── model.py                  # Transformer model definition
├── train.py                  # Training pipeline
├── test.py                   # Inference & evaluation pipeline
├── dataset_generation.py     # Dataset loading, cleaning & synthetic generation
├── gujarati_bpe_6k.json      # Trained BPE tokenizer (vocab size 6144)
├── best_sentence_model.pt    # Best saved model checkpoint
├── log.txt                   # Training logs
└── test_results.txt          # Inference results
```

---

## 🧠 Model Architecture

The model (`SpellTransformer`) is a standard encoder-decoder Transformer with:

| Parameter | Value |
|---|---|
| Embedding Dimension (`d_model`) | 256 |
| Attention Heads (`nhead`) | 4 |
| Encoder Layers | 3 |
| Decoder Layers | 2 |
| Feedforward Dimension | 256 |
| Dropout | 0.1 |
| Vocabulary Size | 6,144 |
| Trainable Parameters | ~4.08M |

- **Tied embeddings**: Input embedding weights are shared with the output projection layer.
- **Positional Encoding**: Sinusoidal `PositionalEncoding` added to embeddings.
- **Tokenizer**: Sentence-Piece BPE tokenizer trained on Gujarati corpus, saved as gujarati_bpe_6k.json.

---

## 📦 Dataset

Handled by dataset_generation.py:

- **Source**: [`autopilot-ai/Gujarati-Grammarly-Datasets`](https://huggingface.co/datasets/autopilot-ai/Gujarati-Grammarly-Datasets) (HuggingFace)
- **Filters applied**:
  - Drop empty rows
  - Max sentence length: **128 characters**
  - Max word length: **25 characters**
- **Synthetic augmentation**: Additional incorrect variants are generated per correct sentence using:
  - Phonetic character confusions (e.g., `શ ↔ સ ↔ ષ`, `ત ↔ ટ`)
  - Vowel matra confusions (e.g., `િ ↔ ી`, `ુ ↔ ૂ`)
- Processed data is cached as `processed_data.csv` to avoid regeneration.

---

## 🏋️ Training

Managed by train.py.

### Run Training

```bash
python train.py --seed 1 --epochs 6 --batch_size 256 --lr 0.0005
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--seed` | `None` | Fixed seed. If omitted, searches seeds 1–50 |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `256` | Batch size |
| `--lr` | `0.0005` | Learning rate |
| `--model_path` | `None` | Path to existing weights to resume training |
| `--log_file` | log.txt | Output log file |

### Seed Search
If no seed is provided, the trainer runs a **partial epoch** (256 batches) for each seed from 1–50 and selects the one with the lowest loss before full training.

### Optimizer & Loss
- **Optimizer**: AdamW (`weight_decay=0.01`)
- **Loss**: Cross-Entropy with `label_smoothing=0.1`, ignoring `<pad>` tokens
- **Gradient clipping**: `max_norm=1.0`
- Best model saved to best_sentence_model.pt whenever validation loss improves.

---

## 🧪 Inference & Testing

Managed by test.py.

### Run Inference

```bash
# Fast greedy decoding (batch)
python test.py --beam_width 1 --samples 4096

# Beam search decoding (more accurate, slower)
python test.py --beam_width 3 --samples 500
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | best_sentence_model.pt | Path to model weights |
| `--batch_size` | `128` | Batch size (greedy only) |
| `--samples` | `4096` | Number of test samples |
| `--beam_width` | `3` | `1` = greedy, `>1` = beam search |
| `--output_file` | test_results.txt | Results output file |

### Inference Modes

| Mode | Function | Description |
|---|---|---|
| **Greedy (Batch)** | `correct_sentence_batch_greedy` | Fast batched decoding, beam width = 1 |
| **Beam Search** | `correct_sentence_beam` | Accurate per-sentence beam search, beam width > 1 |

---

## 📊 Training Progress

The model has been trained for **36 epochs total** (24 + 6 + 6) as noted in log.txt. Recent loss values:

| Epoch | Loss |
|---|---|
| 31 | 1.3447 |
| 32 | 1.3427 |
| 33 | 1.3411 |
| 34 | 1.3396 |
| 35 | 1.3383 |
| 36 | 1.3374 |

---

## ⚙️ Requirements

```bash
pip install torch tokenizers datasets pandas tqdm
```

- Python 3.8+
- PyTorch (CUDA recommended)
- `tokenizers` (HuggingFace)
- `datasets` (HuggingFace)
- `pandas`, `tqdm`

---

## 📝 Notes

- All training output is simultaneously printed to console and saved to log.txt via the `Logger` class.
- The tokenizer and dataset are automatically created and cached on first run.
- CUDA is used automatically if available.