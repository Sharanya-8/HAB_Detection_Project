
# HAB Detection Project

## 1. Project Introduction

* What are Harmful Algal Blooms (HABs)?
* Causes of HABs
* Environmental impact
* Health/ecological impact
* Why early detection is important
* Why satellite remote sensing is useful
* Why Sentinel-2 is selected
* Why deep learning is used

## 2. Problem Statement

Clearly define the problem:

> Detect and spatially segment potential harmful algal bloom regions in inland water bodies using multispectral satellite imagery and deep-learning-based image segmentation.

## 3. Project Objectives

Break the complete project into objectives.

### Objective 1 — Satellite Data Acquisition & Preprocessing

* Study-area selection
* Sentinel-2 data acquisition
* Temporal filtering
* Spatial/AOI filtering
* Cloud filtering
* Cloud masking
* Atmospheric/surface-reflectance considerations
* Band selection
* Reflectance scaling
* Image compositing
* Raster preparation

### Objective 2 — Spectral Feature Extraction & HAB Candidate Generation

* Water-body extraction
* NDWI
* MNDWI
* NDCI
* FAI
* Other potentially useful spectral indicators
* Threshold-based HAB candidate detection
* HAB pseudo-label generation
* Mask generation
* Visualization

### Objective 3 — Dataset & Training-Patch Preparation

* Multi-temporal image organization
* Image/mask alignment
* Raster dimensions
* Patch extraction
* Patch size
* Patch overlap/stride
* Positive patches
* Negative patches
* Class imbalance
* Data cleaning
* Normalization
* Dataset indexing
* Train/validation/test split

### Objective 4 — Deep Learning HAB Segmentation

#### SegFormer

* Architecture
* Encoder
* Decoder
* Transformer-based feature extraction
* Input representation
* Segmentation head
* Training

#### Swin Transformer

* Window-based attention
* Shifted-window mechanism
* Hierarchical feature extraction
* Segmentation architecture
* Training

Potentially:

* U-Net baseline
* CNN baseline
* Other comparison models

## 4. Input Features

Document **every input channel**.

### Sentinel-2 spectral bands

* B2 — Blue
* B3 — Green
* B4 — Red
* B5 — Red Edge
* B6 — Red Edge
* B7 — Red Edge
* B8 — NIR
* B8A — Narrow NIR
* B11 — SWIR
* B12 — SWIR

### Derived features

* NDWI
* MNDWI
* NDCI
* FAI

The current project design uses these 10 Sentinel-2 bands plus four derived indices, giving a **14-channel feature representation**. 

## 5. Spectral Indices

For **each index**, the README should explain:

* Formula
* Bands used
* Physical meaning
* Why it is useful for HAB detection
* Expected interpretation
* Role in the project

For example:

### NDWI

$$
NDWI = \frac{Green-NIR}{Green+NIR}
$$

Used primarily for water detection.

### MNDWI

$$
MNDWI = \frac{Green-SWIR}{Green+SWIR}
$$

Useful for separating water from land/background features.

### NDCI

Explain:

* Formula
* Red-edge relationship
* Relationship with chlorophyll/algal concentration
* Role in candidate HAB detection.

### FAI

Explain:

* Floating algae signal
* Red/NIR/SWIR relationship
* Why it can highlight floating vegetation/algal material.

The Sentinel-2 preprocessing pipeline explicitly derives NDWI, MNDWI, NDCI and FAI after band selection and reflectance scaling. 

---

# 6. Water Mask Generation

Document the complete procedure:

```text
Sentinel-2
     ↓
NDWI/MNDWI
     ↓
Threshold
     ↓
Binary Water Mask
     ↓
Remove non-water pixels
     ↓
Retain waterbody
```

Explain:

* Why water masking is necessary
* Threshold selection
* Binary mask
* Land/background removal
* How the mask is applied to subsequent features.

---

# 7. HAB Pseudo-Label Generation

This deserves its own major section.

Explain:

```text
NDCI
   +
MNDWI / Water Mask
   ↓
Threshold conditions
   ↓
Candidate HAB pixels
   ↓
Binary HAB mask
```

Then explain:

* What a pseudo-label is
* Why pseudo-labels are being used
* Difference between pseudo-label and field-verified ground truth
* Threshold-based candidate generation
* Positive class = HAB
* Negative class = non-HAB
* Mask format
* Limitations of pseudo-labels

---

# 8. Image Processing

Document:

* Raster reading
* GeoTIFF handling
* CRS
* Georeferencing
* Resolution
* Image dimensions
* Band ordering
* NoData handling
* Mask alignment
* Resampling, if required
* Normalization/scaling
* RGB visualization

---

# 9. Multi-Temporal Analysis

Since HABs change over time, document:

* Acquisition dates
* Temporal range
* Multiple Sentinel-2 observations
* Cloud filtering
* Temporal compositing
* Multi-date image organization
* Seasonal variation
* Temporal consistency
* Potential temporal train/test splitting

The project pipeline is designed around Sentinel-2 imagery over a 2023–2025 period. 

---

# 10. Dataset Construction

Document the complete dataset pipeline:

```text
Satellite images
       ↓
Preprocessing
       ↓
Feature extraction
       ↓
HAB masks
       ↓
Image + mask alignment
       ↓
Patch extraction
       ↓
Dataset
       ↓
Train / Validation / Test
```

Include:

* Image dataset
* Mask dataset
* Patch dataset
* Metadata
* Naming convention
* Directory structure
* Number of channels
* Patch dimensions
* Class distribution
* Positive/negative samples
* Data augmentation

---

# 11. Class Imbalance

Explain:

* Why HAB pixels may be much fewer than non-HAB pixels
* Positive/negative pixel imbalance
* Positive/negative patch imbalance
* Sampling strategy
* Weighted loss
* Dice loss
* Focal loss
* Oversampling/undersampling

---

# 12. Deep Learning Pipeline

The README should explain the complete training process:

```text
14-channel image patch
          ↓
Preprocessing / normalization
          ↓
Deep Learning Model
          ↓
Pixel-wise prediction
          ↓
Sigmoid / probability map
          ↓
Threshold
          ↓
Binary HAB mask
```

---

# 13. SegFormer

A complete section covering:

* What SegFormer is
* Transformer encoder
* Hierarchical representation
* Efficient attention
* Decoder
* Segmentation head
* Input channels
* Output mask
* Loss
* Optimizer
* Learning rate
* Batch size
* Epochs
* Checkpointing
* Training/validation process

---

# 14. Swin Transformer

Similarly:

* What Swin Transformer is
* Vision Transformer concept
* Window attention
* Shifted windows
* Hierarchical architecture
* Feature extraction
* Segmentation head
* Input/output
* Training procedure
* Hyperparameters

---

# 15. Training

Document:

* Training dataset
* Validation dataset
* Test dataset
* Number of epochs
* Batch size
* Optimizer
* Learning rate
* Loss function
* Scheduler
* Early stopping
* Model checkpoint
* GPU/CPU
* Reproducibility/random seed

---

# 16. Evaluation

This should explain **exactly how we decide whether the model is good**.

### Confusion Matrix

* TP
* TN
* FP
* FN

### Precision

$$
Precision = \frac{TP}{TP+FP}
$$

### Recall

$$
Recall = \frac{TP}{TP+FN}
$$

### IoU

$$
IoU = \frac{TP}{TP+FP+FN}
$$

### Dice

$$
Dice = \frac{2TP}{2TP+FP+FN}
$$

### Pixel Accuracy

$$
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
$$

---

# 17. Model Comparison

The README should eventually compare:

| Model             | IoU | Dice | Precision | Recall | Accuracy |
| ----------------- | --: | ---: | --------: | -----: | -------: |
| Spectral baseline |     |      |           |        |          |
| SegFormer         |     |      |           |        |          |
| Swin Transformer  |     |      |           |        |          |

And explain **which model performs best and why**.

---

# 18. Prediction & Visualization

Document the final inference procedure:

```text
New Sentinel-2 image
       ↓
Preprocessing
       ↓
Feature extraction
       ↓
14-channel input
       ↓
Trained model
       ↓
Prediction probability
       ↓
Binary HAB mask
       ↓
Geospatial HAB map
       ↓
Overlay on satellite image
```

Outputs should include:

* Prediction mask
* Probability map
* RGB image
* HAB overlay
* Ground-truth/pseudo-label vs prediction
* Difference/error map

---

# 19. Geospatial Output

Explain:

* Mapping predictions back to geographic coordinates
* GeoTIFF output
* CRS preservation
* Spatial extent
* Pixel resolution
* HAB area calculation
* Percentage of water affected
* Potential GIS visualization

---

# 20. Project Architecture

Then show the complete repository structure:

```text
HAB_Detection_Project/
│
├── config/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── patches/
│   ├── labels/
│   └── ...
│
├── models/
│   ├── segformer/
│   └── swin/
│
├── scripts/
│   ├── data acquisition
│   ├── preprocessing
│   ├── feature extraction
│   ├── label generation
│   ├── patch generation
│   ├── training
│   ├── evaluation
│   └── visualization
│
├── utils/
│
├── results/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 21. Technologies Used

For example:

* Python
* Google Earth Engine
* Sentinel-2
* NumPy
* Pandas
* Rasterio
* GeoPandas
* Matplotlib
* PyTorch
* SegFormer
* Swin Transformer
* Git
* GitHub
* VS Code

---

# 22. Reproducibility

Explain:

* Environment setup
* Dependencies
* Configuration
* Random seeds
* Dataset preparation
* Training commands
* Model checkpoints
* Output locations

---

# 23. Limitations

Not just "future work."

Explain scientific limitations:

* Pseudo-label uncertainty
* Absence/limited availability of field measurements
* Cloud contamination
* Atmospheric effects
* Spectral confusion
* Seasonal variability
* Class imbalance
* Limited spatial coverage
* Generalization to other lakes
* Model computational requirements

---

# 24. Future Extensions

Potentially:

* More water bodies
* More years
* Field-validated labels
* Chlorophyll-a measurements
* Real-time monitoring
* Automated satellite ingestion
* Larger training dataset
* Better temporal modeling
* Ensemble models
* Web/GIS dashboard
* HAB severity classification
* Early-warning system


# 25. References

Include the actual papers/resources used for:

* Sentinel-2
* HAB detection
* NDWI
* MNDWI
* NDCI
* FAI
* SegFormer
* Swin Transformer
* Remote sensing
* Deep-learning segmentation

