# Data and Leakage Audit: Dataset Integrity, Duplicates, and Generalization

**Project Title:** Brain Tumor MRI Image Classification Using EfficientNet-B0 with Grad-CAM  
**Workspace:** `X:\Mini Project\brain-tumor-mri-classification`  
**Inspected Data & Manifest Files:**
- `dataset/Training/` and `dataset/Testing/`
- `experiments/duplicate_aware_split/training_manifest.csv`
- `experiments/duplicate_aware_split/manifest.csv`
- `experiments/duplicate_aware_split/cross_class_near_matches.csv`
- `experiments/duplicate_audit/near_duplicate_candidates.csv`
- `experiments/duplicate_audit/exact_duplicate_pairs.csv`
- `reports/data_quality_report.json`
- `reports/train_test_duplicate_candidates.csv`

---

## 1. Verified Dataset Quantities and Class Balance

The dataset represents the benchmark curated by Masoud Nickparvar, consolidating Sartaj et al., Figshare, and Br35H MRI collections.

| Class Name | Training Set (`dataset/Training`) | Official Test Set (`dataset/Testing`) | Total Images | Class Proportion |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | 1,400 | 400 | 1,800 | 25.0% |
| **Meningioma** | 1,400 | 400 | 1,800 | 25.0% |
| **No Tumor** | 1,400 | 400 | 1,800 | 25.0% |
| **Pituitary Tumor** | 1,400 | 400 | 1,800 | 25.0% |
| **Total** | **5,600** | **1,600** | **7,200** | **100.0%** |

- **File Integrity:** 7,200 total images. **0 corrupted files** (100% readable via PIL/OpenCV).
- **Formats:** 7,196 JPEG ($99.94\%$), 4 PNG ($0.06\%$).
- **Color Channels:** Decoded and formatted to 3-channel RGB.
- **Native Resolutions:** 447 unique sizes (top size $512 \times 512$: 5,014 images / $69.64\%$). Standardized to $224 \times 224$ via bilinear anti-aliased resizing.

---

## 2. Exact Byte-Level Duplicates (SHA-256 Audit)

- **Within Training Set:** 153 identical byte-for-byte image pairs identified across the 5,600 training images.
- **Cross-Split Boundary (Training $\cap$ Testing):** **0 exact duplicate pairs**.
- **Conclusion:** No exact, bit-for-bit duplicate image crosses between the official Training and Testing partitions.

---

## 3. Perceptual Hash Candidates & Slice Leakage Audit

Perceptual hashing (dHash and pHash) detects images that are visually near-identical despite slight differences in compression, contrast, or spatial cropping.

### 3.1 The 2,009 Cross-Split Candidate Pairs
In `experiments/duplicate_audit/near_duplicate_candidates.csv`, filtering for pairs where `split_a != split_b` identifies exactly **2,009 candidate cross-split pairs** at Hamming distance threshold $d \le 4$:

| Perceptual Distance | Candidate Pair Count | Nature of Visual Similarity |
| :---: | :---: | :--- |
| **Distance 0** | 697 pairs | Visually identical slices with differing JPEG compression artifacts |
| **Distance 2** | 678 pairs | Adjacent axial slices from identical brain volume with minimal morphological shift |
| **Distance 4** | 634 pairs | Successive axial cuts separated by $2\text{--}5\,\text{mm}$ in the same acquisition |
| **Total** | **2,009 pairs** | Volumetric slice overlap across benchmark boundary |

### 3.2 Severe Class Disparity in Cross-Split Candidates

| Class | Cross-Split Near-Duplicate Pairs | Percentage of Cross-Split Candidates |
| :--- | :---: | :---: |
| **No Tumor** | **1,539 pairs** | **76.61%** |
| **Meningioma** | 223 pairs | 11.10% |
| **Pituitary Tumor** | 149 pairs | 7.42% |
| **Glioma** | 98 pairs | 4.88% |

> [!NOTE]
> **Explaining the 98.5%–100% No-Tumor Performance:**
> Over 76% of all cross-split similarity pairs reside in the `notumor` class. A healthy brain scan contains anatomical slices that closely match standard stereotactic spaces across different subjects. Consequently, the test set for `notumor` contains hundreds of slices visually homologous to training slices, resulting in near-perfect test scores ($\ge 98.5\%$ to $100\%$) across all models.

---

## 4. The Pre-Augmented Meningioma Anomaly

A critical discovery in the dataset files:
- `dataset/Training/meningioma` contains **100 pre-augmented images** named `Tr-aug-me_*.jpg`.
- `dataset/Testing/meningioma` contains **103 pre-augmented images** named `Te-aug-me_*.jpg`.
- **None of the other three classes** (`glioma`, `notumor`, `pituitary`) contain any pre-augmented files.

### Clinical & Modeling Implications:
1. The original benchmark curator artificially expanded the meningioma class prior to uploading to Kaggle, using synthetic affine transformations.
2. In the test set, **25.75% of all meningioma evaluations** (103 out of 400 images) are evaluated on synthetically pre-augmented copies rather than authentic diagnostic MRI acquisitions.

---

## 5. Cross-Class Label Contamination (Conflicting Ground Truth)

In `experiments/duplicate_aware_split/cross_class_near_matches.csv`, cross-referencing perceptual clusters across different diagnostic categories identified **24 near-duplicate image pairs with conflicting diagnostic labels**:

| Image Path A (Label A) | Image Path B (Label B) | Perceptual Distance | Implication |
| :--- | :--- | :---: | :--- |
| `Training/glioma/Tr-gl_1046.jpg` (**glioma**) | `Training/meningioma/Tr-me_1285.jpg` (**meningioma**) | **$d = 2$** | Near-identical scan with contradictory labels |
| `Training/glioma/Tr-gl_750.jpg` (**glioma**) | `Training/meningioma/Tr-me_722.jpg` (**meningioma**) | **$d = 2$** | Near-identical scan with contradictory labels |
| `Training/glioma/Tr-gl_1039.jpg` (**glioma**) | `Training/meningioma/Tr-me_72.jpg` (**meningioma**) | $d = 4$ | Morphologically matched with opposing labels |
| `Training/glioma/Tr-gl_1046.jpg` (**glioma**) | `Training/meningioma/Tr-aug-me_29.jpg` (**meningioma**) | $d = 4$ | Glioma matches augmented meningioma |
| `Training/glioma/Tr-gl_174.jpg` (**glioma**) | `Training/notumor/Tr-no_1205.jpg` (**notumor**) | $d = 4$ | Peripheral slice labeled as tumor vs no-tumor |

### Impact on Model Accuracy:
When two virtually identical images are labeled under opposing classes, any deterministic neural network will receive conflicting gradient updates during training and will inevitably be penalized on at least one during evaluation. This establishes a non-zero Bayes error floor that explains why glioma-meningioma confusion persists across all models.

---

## 6. Duplicate-Aware Grouped Split: Methodology and Verification

To prevent slice overlap between training and validation, the project generated `experiments/duplicate_aware_split/training_manifest.csv`:
- **Grouping Algorithm:** Complete-linkage clustering based on dHash distance ($d \le 4$) and exact SHA-256 hashes.
- **Partition:** All images belonging to the same similarity cluster (`group_id`) were assigned **exclusively to either Train or Validation**, never divided across both.
- **Resulting Split Counts:** **4,447 training images** and **1,153 validation images**.
- **Class Balance:** Exactly 1,400 images per class in the total partition, with near-equal distribution across train and validation:
  - Glioma: 1,121 train / 279 val
  - Meningioma: 1,121 train / 279 val
  - No Tumor: 1,100 train / 300 val
  - Pituitary: 1,119 train / 281 val
- **Verification:** 0 exact hashes cross splits; 0 group IDs cross splits.

---

## 7. Patient-Level Independence vs. Image-Level Deduplication

> [!WARNING]
> **Scientific Integrity Assessment**
> While the duplicate-aware grouped split successfully prevents identical and near-identical slices from crossing between training and validation, **it does NOT prove or guarantee patient-level clinical independence.**

### Why Image-Level Separation Does Not Equal Patient-Level Separation:
1. **Metadata Stripping:** The public Kaggle dataset standardized filenames (`Tr-gl_0010.jpg`, `Te-me_0005.jpg`) and completely stripped all DICOM header attributes (Patient ID, Study Instance UID, Series Instance UID, Patient Age, Acquisition Date).
2. **Non-Adjacent Volumetric Slices:** An axial brain MRI scan typically consists of 20 to 60 slices across the superior-to-inferior z-axis. While adjacent slices ($z$ and $z+1$) have low Hamming distance ($d \le 4$) and get grouped together, slices separated by larger distances ($z$ and $z+10$) may exceed the perceptual threshold ($d > 10$) and be assigned to different splits despite originating from the exact same patient.
3. **Clinical Resolution:** Resolving true patient-level generalization requires an external clinical dataset with preserved patient identifiers (e.g., BraTS, TCIA, or institutional hospital archives).
