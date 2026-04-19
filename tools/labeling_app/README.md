# Tree Crown Labeler

Small local web app for drawing one polygon per tree crown on the 1024 px candidate tiles.

## Run

```bash
conda activate svi_segformer
python tools/labeling_app/server.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

## Inputs

```text
data/processed/labeling_candidates_512_random_seoul_urban/images/*.png
data/processed/labeling_candidates_512_random_seoul_urban/selected_tiles.csv
```

## Outputs

```text
data/processed/labels_512_random_seoul_urban/json/*.json
data/processed/labels_512_random_seoul_urban/yolo_seg/*.txt
data/processed/labels_512_random_seoul_urban/geojson/*.geojson
data/processed/labels_512_random_seoul_urban/classes.txt
```

## Shortcuts

- Click: add polygon point
- Enter or double-click: finish polygon
- Escape: cancel current polygon
- Backspace/Delete: remove selected polygon, or remove the last point while drawing
- S or Ctrl+S: save
- N/P: next or previous tile
- Mouse wheel: zoom
- Space-drag, right-drag, or middle-drag: pan

The app writes JSON first, then exports YOLO-seg txt and GeoJSON from the same polygons.
