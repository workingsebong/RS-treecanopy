# RS-treecanopy

서울 고해상도 항공사진에서 **tree canopy semantic segmentation** 기반 tree canopy map을 생성하고, 도시 미기후(기온) 분석에 활용합니다.

## Pipeline

```text
V-World WMTS aerial imagery (~0.25 m/px, zoom 19)
    ↓  tools/random_urban_sampling.py
urban random patch sampling (park / nature-park exclusion)
    ↓  tools/labeling_app/
manual canopy polygon labeling
    ↓  tools/prepare_semantic_seg_dataset.py
binary semantic mask conversion
    ↓  tools/train_segformer_semantic.py
restor/tcd-segformer-mit-b2 fine-tuning
    ↓
Tree Canopy Map (binary mask)
    ↓  [next]
polygonize → canopy area → S-DoT AT mixed-effects model
```

## Current Result

Main model: `restor/tcd-segformer-mit-b2` fine-tuned on 36-tile Seoul pilot dataset

| model | setting | IoU | Dice | Precision | Recall |
|-------|---------|-----|------|-----------|--------|
| `nvidia/mit-b0` | fine-tuned | 0.507 | 0.673 | 0.679 | 0.666 |
| `restor/tcd-segformer-mit-b2` | zero-shot, thr=0.30 | 0.654 | 0.791 | 0.763 | 0.821 |
| `restor/tcd-segformer-mit-b2` | fine-tuned, thr=0.55 | **0.691** | **0.817** | 0.763 | 0.879 |

Best model output:

```text
outputs/segformer/tree_canopy_semantic_tcd_mit_b2_finetune_e50/
```

## Main Commands

```bash
conda activate svi_segformer
```

**1. Random urban sampling & tile generation**
```bash
python tools/random_urban_sampling.py
```

**2. Labeling app**
```bash
python tools/labeling_app/server.py --host 127.0.0.1 --port 8765
# → http://127.0.0.1:8765
```

**3. Prepare semantic segmentation dataset**
```bash
python tools/prepare_semantic_seg_dataset.py --overwrite
```

**4. Zero-shot evaluation**
```bash
MPLCONFIGDIR=/tmp/matplotlib \
python tools/evaluate_segformer_semantic_model.py \
  --model-id restor/tcd-segformer-mit-b2 \
  --output-dir outputs/segformer/tree_canopy_semantic_tcd_mit_b2_zeroshot \
  --image-size 512 --device cuda --overwrite
```

**5. Fine-tune**
```bash
MPLCONFIGDIR=/tmp/matplotlib \
python tools/train_segformer_semantic.py \
  --pretrained-model restor/tcd-segformer-mit-b2 \
  --epochs 50 --batch-size 4 --image-size 512 --device cuda \
  --lr 5e-5 --max-canopy-weight 5 \
  --output-dir outputs/segformer/tree_canopy_semantic_tcd_mit_b2_finetune_e50 \
  --overwrite --no-random-fallback
```

**6. Threshold sweep**
```bash
MPLCONFIGDIR=/tmp/matplotlib \
python tools/evaluate_segformer_thresholds.py \
  --run-dir outputs/segformer/tree_canopy_semantic_tcd_mit_b2_finetune_e50 \
  --image-size 512 --device cuda
```

**7. Generate presentation**
```bash
python tools/make_presentation.py
# → papers/presentation/tree_canopy_presentation.pptx
```

## Structure

```text
tools/
  labeling_app/                   local web app for canopy polygon labeling
  random_urban_sampling.py
  prepare_semantic_seg_dataset.py
  train_segformer_semantic.py
  evaluate_segformer_semantic_model.py
  evaluate_segformer_thresholds.py
  render_pretrained_model_review_images.py
  make_presentation.py
  make_research_proposal_presentation.py

data/                             ignored
  raw/
  processed/

outputs/                          ignored
  segformer/                      model training & evaluation results
  figures/                        intermediate figures for presentations
  yolo/                           legacy
  pretrained_baselines/           legacy

papers/
  presentation/                   final PPT files
  manuscript/
  references/
  submission/

legacy/
  tools/                          YOLO, ArcGIS, DeepForest, Detectree2 scripts
  notebooks/
    01_model_training/
    02_inference_vectorize/
```

## Notes

- `data/`, `outputs/`, `.env`, and model weights are git-ignored.
- YOLO11n-seg, DeepForest, Detectree2, ArcGIS experiments moved to `legacy/`.
- Next: canopy mask polygonize → buffer join with S-DoT → mixed-effects AT model.
