# Pipeline: Knowledge Distillation

Trains a lightweight **student model** by distilling knowledge from a larger teacher (e.g., the ViT-SO400M trained in the `vit_transformer` pipeline), then uses the student for temporal classification.

> **Status: Coming soon**

This pipeline will follow the same general structure as the other pipelines in this repo:
1. Use the shared preprocessing outputs (`ds_driveguard_temporal_roi/`)
2. Use a pretrained teacher model to supervise training of a smaller student spatial model
3. Extract features from the student model for each 16-frame sequence → `.npy` tensors
4. Train a temporal model on the distilled feature sequences for activity classification
