# CODSOFT_TASK5 — Handwritten Text Generation

Implements a character-level Recurrent Neural Network (RNN) from scratch to generate new text based on learned character patterns.

## 📌 Problem Statement

Build a character-level RNN that learns the statistical patterns of a text corpus (letter sequences, spelling, spacing, punctuation) and generates new, original text one character at a time.

## 📂 Dataset

- Character vocabulary sourced from the [Chars74K English dataset](https://www.kaggle.com/datasets/sovitrath/handwritten-alphabets) (`english.csv`) — 62 handwritten glyph labels (0–9, A–Z, a–z)
- Training text: a short bundled sample corpus (`corpus.txt`, ~3,500 characters), since Chars74K contains only isolated single-character images with no sequence information and therefore cannot by itself teach a model letter-to-letter patterns

## 🛠️ Approach

1. **Vocabulary Construction**
   - Combined the Chars74K label alphabet with punctuation/whitespace characters needed for real text (67 characters total)

2. **Model Implementation (from scratch, NumPy only)**
   - Vanilla RNN: `Wxh`, `Whh`, `Why` weight matrices, `tanh` hidden state, softmax output
   - Manually implemented forward pass, cross-entropy loss, and full backpropagation-through-time
   - Gradient clipping and Adagrad optimizer for stable training

3. **Training**
   - Truncated BPTT over 25-character sequence chunks
   - 20,000 training iterations, hidden size 128

4. **Text Generation**
   - Autoregressive sampling: model predicts one character at a time, feeding each prediction back in as the next input
   - Temperature parameter controls randomness (lower = safer/repetitive, higher = more varied)

## 📊 Results

Training loss dropped from **105.1 → 34.3** over 20,000 iterations, showing clear, consistent learning.

Generation quality improved visibly over training:
- **Iteration 1** (untrained): `FvGYgGHPn1YbVvWYXf3s'aX8'3Cqp...` (random noise)
- **Iteration 20,000**: `eet althay mo ping sorruthed parter ingithetercere sall...` (correctly spelled common words like "the", "and", "of"; proper spacing; capitalization after sentence breaks; paragraph structure)

Tested generation at multiple temperatures to demonstrate control over output style:
- **0.5** (conservative): more repetitive, closer to common learned words
- **0.8** (balanced, default): mix of real word fragments and variety
- **1.1** (creative): more randomness, less structure

## 📁 Repository Contents

- `CODSOFT_TASK5_Handwritten_Text_Generation.ipynb` — full notebook (vocabulary, model, training loop, generation)
- `README.md` — this file

## ▶️ How to Run

1. Open the notebook in Google Colab or Jupyter
2. Upload `english.csv` (Chars74K labels)
3. Run all cells in order (the training corpus is embedded directly in the notebook)

## 🔮 Possible Improvements

- Train on a larger, more diverse text corpus for more coherent output
- Upgrade to an LSTM/GRU cell for longer-range memory
- Train on real transcribed handwritten notes if available, for a closer match to the task's literal intent
