# Emotion Detection Model — Development Report

A full record of the iterative process used to build, debug, and improve a facial emotion classification model — from a from-scratch CNN baseline through to a fine-tuned ResNet18 on RAF-DB, including every bug hit and fix applied along the way.

---

## 1. Project Setup

**Goal:** Classify facial images into emotion categories using PyTorch, starting from a custom CNN baseline and progressing to transfer learning.

**Environment:**
- PyTorch 2.2.2 + CUDA 12.1 build (`cu121`), confirmed compatible with a CUDA 13.1 driver via backward compatibility
- GPU training via `torch.device("cuda")`

**Initial dataset:** FER2013-style data, organized as:
```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
```
All images in JPEG format, 7 classes.

---

## 2. Baseline CNN — Built From Scratch

### 2.1 Preprocessing Pipeline
Settled on the following for the baseline, deliberately keeping it simple:
- **Resize to 48×48** — matches native FER2013 resolution, avoids adding false detail via upscaling, and keeps training fast for rapid iteration
- **Grayscale, 1 channel** — emotion is carried in facial structure, not color
- **Normalize** to a [-1, 1] range
- **No augmentation** in the first pass — baselines should be as simple as possible so later improvements can be measured against a clean reference point

### 2.2 Architecture
A 3-block custom CNN:
```
Block 1: Conv2d(1→32) → BatchNorm → ReLU → MaxPool   (48×48 → 24×24)
Block 2: Conv2d(32→64) → BatchNorm → ReLU → MaxPool   (24×24 → 12×12)
Block 3: Conv2d(64→128) → BatchNorm → ReLU → MaxPool  (12×12 → 6×6)
Classifier: Flatten → Linear(256) → Dropout(0.5) → Linear(64) → Dropout(0.3) → Linear(7)
```
### 2.3 Trial 1 — No validation split, no augmentation
Trained for 25 epochs on training accuracy only.
- **Result:** Train accuracy climbed to 74.47% by epoch 25, loss decreasing steadily.
- **Limitation identified:** No way to know if the model was overfitting — needed a validation split.

### 2.4 Adding Validation
Split training data 80/20 into train/val subsets using `random_split`. Re-ran the same architecture.
- **Result:** Train Acc 84.15%, Val Acc 71.56% — a 12.5% gap, with val loss almost double train loss. **Confirmed overfitting.**

### 2.5 Trial 2 — Augmentation Added
Added `RandomHorizontalFlip` and `RandomRotation(10)` to the training transform only (never to validation, since val must reflect real-world unaugmented data).

**Bug discovered:** Both `train_subset.dataset` and `val_subset.dataset` pointed to the *same* underlying `ImageFolder` object. Setting `val_subset.dataset.transform = val_transform` silently overwrote the train transform too, meaning the "augmented" training run had actually trained on **non-augmented data** the whole time. No error was thrown — just misleading results.

**Fix:** Created two fully separate `ImageFolder` instances (one per transform) pointing at the same directory, then built `Subset` objects from each.

Re-ran with the fix in place:
- **Result:** Best val accuracy ~87.7% at epoch 3, but accuracy steadily declined afterward as training continued to epoch 25 (down to ~79% val) — a textbook overfitting curve once augmentation revealed the true epoch-3 peak.

### 2.6 Early Stopping + Checkpointing
Implemented early stopping (patience-based on validation loss) with `torch.save(model.state_dict(), ...)` to automatically capture the best-performing epoch rather than the final one.

**Bug discovered:** On the first attempt, the "fresh" run reported a near-perfect result at epoch 1 — a clear sign the model object hadn't actually been reinitialized and was continuing from previously trained weights. **Fix:** explicitly reinitialize `model`, `optimizer`, `best_val_loss`, and `patience_counter` before every fresh run.

### 2.7 Trial 3 — Reinitialized Run, Inconsistent Results
After reinitializing, results were unexpectedly poor (~58-64% val) — much lower than the earlier 87% run, despite using the "fixed" pipeline.

**Diagnosis process:**
1. Verified batch shape, pixel range, and label encoding — all correct
2. Checked class distribution — found severe class imbalance: **disgust had only 357 train images vs. ~5,800 for happy** (~16x difference)
3. **Second, deeper bug found:** the dataset-loading code was passing the *output of `random_split`* (which returns `Subset` objects) back into a second `Subset(...)` call — wrapping a Subset inside another Subset and corrupting the index mapping.

**Fix:** Generated plain index lists manually (`list(range(len(dataset)))`), shuffled with a fixed `random.seed(42)` for reproducibility, sliced directly, and passed those plain lists into `Subset()`. Also seeding the split ensured every kernel restart used an identical train/val partition, making runs comparable for the first time.

### 2.8 Class Imbalance Handling
With class imbalance confirmed (disgust: 357 vs. happy: 5,782), two approaches were discussed:
- `WeightedRandomSampler` + weighted `CrossEntropyLoss` (oversampling + loss penalty)
- Simply removing the disgust class, given the dataset was large (~28k images) and losing one class barely affected total volume

**Decision:** Removed the `disgust` folder entirely, reducing to 6 classes, since the imbalance was too extreme relative to dataset size at the time.

### 2.9 Trial 3 (Corrected) — Clean Pipeline, 6 Classes
With the index bug fixed and disgust removed, re-ran for 100 epochs (extended from 50, since the model was still climbing and hadn't triggered early stopping).
- **Result:** Plateaued around 86 epochs (early stopping triggered), best checkpoint at epoch 76: **Train 66.25%, Val 61.11%**
- **Diagnosis:** Train and val tracked closely (~5% gap) — this was **underfitting**, not overfitting. The 3-block CNN had hit its capacity ceiling for the task; it needed more representational power, not more regularization.

**Baseline CNN final result: ~61% validation accuracy.**

---

## 3. Transfer Learning — ResNet18 (Attempt 1: FER2013)

### 3.1 Pipeline Changes for ResNet18
- Resize → 224×224 (ResNet's expected input)
- Grayscale images duplicated across 3 channels (`Grayscale(num_output_channels=3)`) to match RGB input shape while preserving the fact the source images were grayscale
- Normalized using ImageNet mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`)

### 3.2 Corrupted Pretrained Weights Download
`models.resnet18(weights="IMAGENET1K_V1")` repeatedly failed with a hash mismatch error (`invalid hash value`), even after clearing the local torch cache and retrying — download was failing/corrupting near the end each time, likely a network issue.

**Fix:** Manually downloaded the `.pth` weights file directly via browser from the PyTorch model URL, placed it in the local torch hub cache directory (`~/.cache/torch/hub/checkpoints/`) with the exact expected filename, allowing PyTorch to find it locally without re-downloading.

### 3.3 Frozen-Backbone Attempt
Froze all ResNet18 layers except the final fully-connected layer (only 3,078 trainable parameters out of 11.18M total).
- **Result:** Plateaued around 42-43% val accuracy after 10 epochs — **far worse than the baseline CNN.** The frozen ImageNet features weren't adapted enough to the emotion task; too few parameters were being trained to bridge that gap.

### 3.4 Unfreezing Layer4
Unfroze `layer4` (ResNet's last residual block) plus the final layer — 8.4M trainable parameters — and dropped the learning rate to `0.0001`.
- **Result:** Climbed quickly to ~65% train / ~63% val by epoch 5, but by epoch 10 train had reached 84% while val stayed flat near 64% — clear overfitting, confirmed by the widening gap.

### 3.5 Early-Stopping Metric Confusion
Noticed val *accuracy* was improving slightly epoch-to-epoch while early stopping (tracking val *loss*) reported "no improvement," since loss and accuracy occasionally moved in opposite directions.

**Fix:** Switched the early-stopping criterion to track validation accuracy directly instead of validation loss, and increased patience.

### 3.6 CUDA Runtime Errors
Hit `RuntimeError: CUDA error: device-side assert triggered` — root cause was stale GPU state from a previous run with a different `num_classes` value still loaded into memory. **Fix:** Restart kernel and re-run all cells fresh, in order, whenever the class count or core model definition changes.

### 3.7 Discriminative Learning Rates
To curb overfitting from unfreezing more of the pretrained network, applied **per-layer learning rates** — much smaller for the pretrained body, normal for the new head:
```python
optimizer = Adam([
    {"params": resnet.layer3.parameters(), "lr": 0.000001},
    {"params": resnet.layer4.parameters(), "lr": 0.00001},
    {"params": resnet.fc.parameters(),     "lr": 0.0001}
])
```
This produced the healthiest training curve so far — train and val accuracy rising together epoch-over-epoch with only a 1-2% gap in early epochs.

**Result on FER2013 (Trial 4, layer3+4 unfrozen):** Plateaued at **~63% val accuracy** after 20 epochs — only marginally better than the frozen-layer attempts and the original baseline CNN.

### 3.8 Diagnosis: Data Quality, Not Model Capacity
With multiple architectures and configurations all converging around the same 60-63% ceiling, the bottleneck was identified as **the dataset itself** — FER2013 is known to contain mislabeled images, off-angle faces, and obstructed faces (a manual visual spot-check confirmed this).

**Attempted fix:** Automated face-detection-based cleaning using OpenCV's Haar Cascade classifier to remove images with no detectable face.
- **Result:** Catastrophic over-removal — the detector's default parameters were too strict for this dataset's low-resolution, off-angle images, flagging and deleting **100% of the dataset** as having "no face." Backup of the original data was needed before any further attempts (this was set up but ultimately abandoned in favor of switching datasets entirely).

---

## 4. Switching Datasets — RAF-DB

### 4.1 Decision
Rather than continue fighting FER2013's label/image quality issues, switched to **RAF-DB** (Real-world Affective Faces Database), a higher-quality academic dataset with cleaner, aligned, real-world face images. Requested access directly from the dataset authors and downloaded the **basic emotion** variant (7 classes — the compound-emotion, multi-label, and action-unit variants were intentionally not used, as they serve different tasks).

### 4.2 Organizing RAF-DB
RAF-DB ships as a flat folder of aligned face images plus a separate label `.txt` file mapping filenames to numeric emotion labels — not pre-sorted into class folders like FER2013 was.

**Bugs hit while restructuring:**
1. Initial restructuring script produced empty class folders for every class — the label file's filenames (e.g. `train_00001.jpg`) didn't match the actual image filenames on disk (`train_00001_aligned.jpg`), since RAF-DB's "aligned" image set appends `_aligned` to each filename.
   **Fix:** Inserted `_aligned` before the file extension when building the source path during the copy step.
2. After fixing the filename mismatch, the resulting class distribution was identical to an earlier (initially distrusted) Kaggle version of the dataset — this confirmed that Kaggle copy *was* genuine RAF-DB all along, just organized by numeric folder instead of by name.

### 4.3 Native Class Imbalance Confirmed
Unlike FER2013, this imbalance was treated as a legitimate property of the real dataset (RAF-DB), not a quality problem:
```
angry:     4,772        sad:        1,982
happy:     2,524        disgust:      717
surprise:  1,290        neutral:      705
                         fear:         281
```
**Decision:** Kept all 7 classes this time (rather than removing low-count classes as done with FER2013's disgust) since RAF-DB is a smaller, higher-value dataset — losing a class would lose 2%+ of total data and a whole emotion category from the dataset specifically requested for its quality.

### 4.4 Handling Imbalance — Sampler vs. Weighted Loss
**First attempt:** Combined `WeightedRandomSampler` (oversamples minority classes per batch) **and** `CrossEntropyLoss(weight=...)` (penalizes minority-class errors more heavily) simultaneously.
- **Result:** 84.78% train / 63.93% val — large gap, and the per-class classification report revealed *overcorrection*: angry (the majority class) had high precision but poor recall (real angry images were being misclassified as minority classes), while disgust had poor precision (over-predicted). Stacking both correction methods compounded the effect too aggressively.

**Fix:** Used only the `WeightedRandomSampler`, with plain (unweighted) `CrossEntropyLoss`.
- **Result:** Jumped to **70.89% val accuracy**, with per-class F1 scores improving across nearly every category compared to the double-corrected version.

### 4.5 Kernel Crashes (Windows-Specific)
Encountered hard kernel crashes (not Python exceptions) mid-training. Diagnosed and resolved via:
1. Fully closing other open notebook kernels (multiple notebooks from earlier experiments were still holding GPU memory)
2. Adding `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` at the top of the notebook — a known fix for Windows + PyTorch's duplicate OpenMP runtime conflict, a common silent-crash cause in this environment

### 4.6 Regularization — Dropout
To address the still-present overfitting gap (~18-20% between train and val), added a `Dropout(0.4)` layer immediately before the final classification layer:
```python
resnet.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(resnet.fc.in_features, 7)
)
```
Extended training to 40 epochs with patience of 12 to give the model more room before stopping.
- **Result:** Best validation accuracy of **74.20%** at epoch 31, after which val plateaued (73-74%) while train continued climbing past 90% — indicating this configuration's ceiling had been reached.

---

## 5. Final Evaluation

Loaded the best saved checkpoint (`best_rafdb_resnet18.pth`) and evaluated against RAF-DB's held-out **test** set — data the model had never seen during training or validation.

### Final Test Results
| Metric | Value |
|---|---|
| **Test Accuracy** | **76.37%** |
| Macro avg F1 | 0.67 |
| Weighted avg F1 | 0.76 |

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Angry | 0.92 | 0.85 | 0.88 | 1185 |
| Disgust | 0.49 | 0.38 | 0.43 | 160 |
| Fear | 0.65 | 0.46 | 0.54 | 74 |
| Happy | 0.69 | 0.76 | 0.73 | 680 |
| Neutral | 0.69 | 0.68 | 0.69 | 162 |
| Sad | 0.66 | 0.74 | 0.70 | 478 |
| Surprise | 0.73 | 0.79 | 0.76 | 329 |

Test accuracy (76.37%) slightly exceeding validation accuracy (74.20%) is a good sign — it indicates the model generalizes genuinely well rather than being overfit to the validation split specifically.

---

## 6. Full Results Progression

| Stage | Configuration | Val/Test Accuracy |
|---|---|---|
| Baseline CNN (scratch, no aug, no val split) | 25 epochs | 74.47% train only |
| Baseline CNN + val split | revealed overfitting | 71.56% val |
| Baseline CNN + augmentation (post transform-bug fix) | early peak | 87.7% val *(later found to be inflated by a bug)* |
| Baseline CNN, corrected pipeline, 6 classes | 100 epochs, early stop @86 | 61.11% val |
| ResNet18, frozen backbone | 10 epochs | ~42% val |
| ResNet18, layer4 unfrozen | 20 epochs | ~64% val |
| ResNet18, layer3+4 unfrozen, FER2013 | 20 epochs | ~63% val |
| ResNet18, layer3+4 unfrozen, RAF-DB, sampler+weighted loss | 20 epochs | 63.93% val |
| ResNet18, layer3+4 unfrozen, RAF-DB, sampler only | 20 epochs | 70.89% val |
| **ResNet18 + dropout(0.4), RAF-DB, sampler only** | **40 epochs, early stop @31** | **74.20% val** |
| **Final held-out test evaluation** | — | **76.37% test** |

---

## 7. Key Lessons From This Project

1. **Silent bugs are the real danger in ML, not crashes.** Several of the most damaging issues (the shared-transform bug, the double-`Subset` indexing bug) produced no errors at all — just plausible-looking but wrong numbers. Always sanity-check shapes, class distributions, and sample counts after every data-loading change.
2. **Always reinitialize fully before a "fresh" run.** Model, optimizer, and tracking variables (`best_val_loss`/`best_val_acc`, `patience_counter`) all need to be reset — otherwise a "new" run silently continues from old weights.
3. **A small train/val gap can mean underfitting, not success.** The baseline CNN's close train/val tracking wasn't a sign of a good model — it meant the model lacked the capacity to learn the task at all.
4. **Class-imbalance corrections can stack destructively.** Combining a weighted sampler with a weighted loss function over-corrected in the same direction, actively hurting the majority class's recall.
5. **Data quality has a hard ceiling that no amount of architecture tuning can overcome.** Every ResNet18 configuration on FER2013 converged to roughly the same ~60-63% ceiling regardless of which layers were unfrozen — the real fix was switching to a cleaner dataset (RAF-DB), not further model tuning.
6. **Environment-specific issues are real and not your fault.** Corrupted pretrained-weight downloads, Windows-specific OpenMP kernel crashes, and stale GPU memory from other open notebooks all caused real failures unrelated to the modeling logic itself.

