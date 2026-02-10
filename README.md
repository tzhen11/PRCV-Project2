# Project 2: Content-Based Image Retrieval

**Names:** Aafi Mansuri & Terry Zhen

## Description
A content-based image retrieval system that finds similar images using various feature extraction methods and distance metrics.

## Features Implemented
- Task 1: Baseline 7x7 pixel matching
- Task 2: Color histogram (RG chromaticity)
- Task 3: Multi-region histogram
- Task 4: Texture + Color features
- Task 5: ResNet deep network embeddings
- Task 6: DNN vs Classic feature comparison
- Task 7: Custom design (ResNet + Color histogram)

## How to Build
```bash
make buildFeatures
make matchImage
```

## How to Run

**Build features:**
```bash
./buildFeatures.exe <image_directory> <feature_method> <output_csv>
./buildFeatures.exe olympus baseline features.csv
```

**Match images:**
```bash
./matchImage.exe <target_image> <feature_method> <feature_csv> <N>
./matchImage.exe olympus/pic.0535.jpg texture texture.csv 5
```

**Feature methods:** baseline, chistogram, mhistogram, texture, resnet, custom

## Time Travel Days
None used.