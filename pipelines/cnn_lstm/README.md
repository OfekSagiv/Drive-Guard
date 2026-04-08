# Pipeline: CNN + LSTM

Spatial feature extraction using a **CNN backbone**, followed by an **LSTM** temporal classifier.

> **Status: Coming soon**

This pipeline will follow the same general structure as the other pipelines in this repo:
1. Use the shared preprocessing outputs (`ds_driveguard_temporal_roi/`)
2. Train a CNN spatial model to extract per-frame feature vectors
3. Extract features for each 16-frame sequence → `.npy` tensors
4. Train an LSTM on the feature sequences for activity classification
5. Evaluate on the test split and compare against the ViT-Transformer baseline
