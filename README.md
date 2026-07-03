data and models on this link:
https://drive.google.com/drive/folders/10e4VhqSTUjja8whnQ8neg0E6zuctoPaL?usp=drive_link



# Visual Storyteller — Image Captioning with Attention

An image captioning model that generates a natural-language description of a photo. It combines a CNN image encoder with an attention mechanism and an LSTM decoder, trained on the Flickr8k dataset in Google Colab, with beam-search decoding and BLEU-based evaluation at inference time.

## What it does

Given an image, the model generates a caption word-by-word, using attention to focus on different regions of the image as it produces each word — the "Show, Attend and Tell" approach. At inference time, captions are produced with beam search rather than greedy decoding, and the model's caption quality is measured quantitatively with corpus BLEU scores on held-out test images, in addition to qualitative sample outputs.

## Architecture

- **Encoder — ResNet18 (pretrained on ImageNet):** the final classification layers are removed, and the remaining convolutional layers produce a 7×7 grid of 512-dimensional feature vectors per image (via adaptive average pooling). This grid represents different spatial regions of the image.
- **Attention module:** a small feed-forward network that takes the encoder's feature grid and the decoder's current hidden state, and computes a weighted combination of image regions ("where to look") for each decoding step.
- **Decoder — LSTM with attention (LSTMCell):** at each time step, it embeds the previous word, combines it with the attention-weighted image features (passed through a learned gate), and predicts the next word. The LSTM's initial hidden and cell states are derived from the mean-pooled image features.

Trained end-to-end with teacher forcing (the decoder sees the ground-truth previous word during training) and cross-entropy loss, with the `<pad>` token excluded from the loss. The loss also includes a **doubly-stochastic attention regularization** term (weighted by `ALPHA_C`), which penalizes the model if, summed across the whole caption, some image regions receive far more attention than others — this pushes the decoder to attend to the full image over the course of generating a caption rather than fixating on a few pixels.

## Dataset

- **Flickr8k**: ~8,000 images, each with 5 human-written captions.
- Data is expected on Google Drive at:
  ```
  MyDrive/visual_storyteller/
  ├── data/
  │   ├── captions.txt      # columns: image, caption
  │   └── Images/            # .jpg files
  └── models/                 # checkpoints get saved here
  ```
- The notebook auto-detects the image folder even if it's nested differently (e.g. inside an extracted zip), and drops any caption rows whose image file is missing.

## Data pipeline

1. **Load captions** from `captions.txt` into a pandas DataFrame, keep only rows with a matching image file.
2. **Split by image (not by row)**, 80/10/10 train/val/test, so all 5 captions of a given image stay in the same split (prevents leakage). Splits are saved as CSVs for reuse.
3. **Build a vocabulary**: simple regex tokenizer (alphabetic/numeric tokens, lowercased), keeping words that appear at least `MIN_WORD_FREQ` (2) times. Special tokens: `<pad>`, `<start>`, `<end>`, `<unk>`.
4. **Dataset/DataLoader**: images are resized/augmented (random crop + horizontal flip for train, center crop for val) to 224×224 and normalized with ImageNet stats. Captions are encoded to token IDs and padded to a common length within each batch via a `collate_fn` that takes the padding index as an explicit argument (bound once with `functools.partial`), so batching doesn't depend on any global state.

## Training

- Loss: cross-entropy (ignoring `<pad>` positions) plus the doubly-stochastic attention regularization term described above.
- Optimizer: Adam, initial learning rate `1e-3`, only over parameters that require gradients (the encoder is frozen by default — `FINE_TUNE_ENCODER = False`).
- **Learning rate schedule**: `ReduceLROnPlateau` halves the learning rate whenever validation loss hasn't improved for `LR_PATIENCE` (2) epochs.
- **Early stopping**: training stops automatically once validation loss hasn't improved for `EARLY_STOP_PATIENCE` (3) epochs, avoiding wasted compute on a plateaued model.
- Mixed precision (AMP, via the current `torch.amp` API) on GPU, gradient clipping at norm 5.0.
- Up to 8 epochs by default (2 if `FAST_DEV_RUN = True`, which also subsamples to 500 images for quick iteration/debugging), though early stopping may end training sooner.
- After each epoch, both training and validation loss are computed and logged along with the current learning rate; the checkpoint (model weights, optimizer state, vocabulary, config, and training history) is saved to Google Drive (`models/caption_model_best.pth`) whenever validation loss improves, and the final-epoch model is also saved separately (`caption_model_final.pth`).
- Training/validation loss curves are plotted at the end.

## Inference

Two caption-generation strategies are implemented:

- **Greedy decoding**: at each step, picks the single highest-probability next word.
- **Beam search decoding** (used for both the sample outputs and BLEU evaluation, with beam width `BEAM_SIZE = 3`): keeps track of the top-k most probable partial captions at each step rather than committing to one word at a time, and returns the full sequence with the highest overall score once beams reach `<end>` (with a length-safe fallback if none do within `GENERATION_MAX_LEN`). This produces noticeably better captions than greedy decoding.

The notebook loads the best checkpoint, samples a handful of images from the held-out test set, and displays each image alongside its beam-search-generated caption and the human reference captions for a qualitative check.

## Evaluation

Beyond the qualitative examples, the notebook runs a **quantitative BLEU evaluation**: it generates beam-search captions for a sample of test images and computes corpus-level BLEU-1 through BLEU-4 against all 5 human reference captions per image (with smoothing to avoid zero scores on short captions). This gives an actual text-similarity metric for judging caption quality, rather than relying on validation loss alone.

## Key hyperparameters

| Setting | Value |
|---|---|
| Image size | 224×224 |
| Max caption length | 30 tokens |
| Min word frequency | 2 |
| Batch size | 64 |
| Epochs (max / early-stopped) | 8 |
| Learning rate (initial) | 1e-3 |
| LR scheduler patience | 2 epochs |
| Early stopping patience | 3 epochs |
| Attention regularization weight (`ALPHA_C`) | 1.0 |
| Attention / embedding dim | 256 |
| Decoder (LSTM) dim | 512 |
| Dropout | 0.5 |
| Beam width (inference/BLEU) | 3 |
| Encoder | ResNet18, pretrained, frozen |

## Tech stack

- **PyTorch** + **torchvision** (ResNet18, image transforms)
- **NLTK** (`corpus_bleu`) for BLEU scoring
- **pandas / numpy** for data handling
- **PIL** for image loading
- **matplotlib** for plotting loss curves and sample outputs
- **tqdm** for progress bars
- Runs on **Google Colab** with **Google Drive** for data/checkpoint storage, and CUDA (mixed precision) when a GPU is available.

## Reproducibility

All random seeds (Python, NumPy, PyTorch, CUDA) are fixed to `42`, and the train/val/test split, dev-subsampling, and BLEU-evaluation image sampling are all seeded, so results should be repeatable across runs.
