# Legacy Experiments

This directory keeps older scripts and notebooks from the instance-segmentation phase.

The current main pipeline is tree canopy semantic segmentation:

```text
tools/random_urban_sampling.py
tools/labeling_app/
tools/prepare_semantic_seg_dataset.py
tools/train_segformer_semantic.py
tools/evaluate_segformer_semantic_model.py
tools/evaluate_segformer_thresholds.py
```

Files here are retained only for reference:

- YOLO11n-seg instance baseline
- DeepForest and Detectree2 pretrained comparisons
- ArcGIS tree model runner
- Old model-training notebooks

