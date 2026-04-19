# RS-treecanopy

나무를 찾습니다. 한 그루씩.

---

## 뭐 하는 프로젝트냐면

서울 어딘가의 여름 항공사진을 받아서, 나무 하나하나를 AI한테 찾아내게 시키고, 그 결과로 tree map을 만드는 프로젝트입니다.

Landsat으로 하면 30m/px라서 나무가 픽셀 속에 묻혀버립니다. 그래서 0.25m짜리 항공사진을 씁니다. 나무보다 픽셀이 작아야 나무를 찾죠.

---

## 파이프라인 요약

```
항공사진 (0.25m/px, 여름)
    ↓
타일 분할
    ↓
라벨링 (사람이 직접... 네)
    ↓
YOLO-seg / Mask R-CNN 학습
    ↓
개별 crown mask + 위치 + 면적
    ↓
GeoJSON / Shapefile
```

---

## 기술 스택

- Python 3.10, WSL2
- rasterio, geopandas, shapely
- PyTorch, ultralytics (YOLO), transformers
- V-World WMTS API (테스트용 항공사진)

---

## 현재 상태

- [x] V-World 항공사진 수집 파일럿 완료
- [x] 512px 라벨링 후보 타일 생성
- [x] 공원·도시자연공원구역 제외 마스크 적용
- [x] 프로젝트 전용 polygon 라벨링 앱 추가
- [x] 475개 tree crown instance segmentation polygon 라벨 완료
- [x] YOLO11n-seg 베이스라인 학습
- [ ] Tree map 생성

---

## 라벨링 앱

랜덤 도시부 샘플을 다시 만들 때:

```bash
conda activate svi_segformer
python tools/random_urban_sampling.py
```

기본 실행은 `data/raw` 아래의 UPIS 공원/도시자연공원구역 shapefile을 찾아 제외 마스크로 씁니다. patch나 tile bbox의 5% 이상이 제외 마스크와 겹치면 샘플링/후보 선별에서 빠집니다.

라벨링 앱 실행:

```bash
conda activate svi_segformer
python tools/labeling_app/server.py --host 127.0.0.1 --port 8765
```

브라우저에서 엽니다.

```text
http://127.0.0.1:8765
```

입력은 `data/processed/labeling_candidates_512_random_seoul_urban/images`의 후보 타일이고, 저장하면 `data/processed/labels_512_random_seoul_urban` 아래에 JSON, YOLO-seg txt, GeoJSON이 같이 생깁니다.

---

## 모델 학습

YOLO-seg 학습셋 생성:

```bash
conda activate svi_segformer
python tools/prepare_yolo_seg_dataset.py --overwrite
```

YOLO11n-seg 베이스라인 학습:

```bash
YOLO_CONFIG_DIR=/tmp/Ultralytics MPLCONFIGDIR=/tmp/matplotlib \
yolo task=segment mode=train model=yolo11n-seg.pt \
  data=data/processed/yolo_seg_512_random_seoul_urban/dataset.yaml \
  epochs=50 imgsz=512 batch=2 device=cpu workers=0 \
  project=outputs/yolo name=tree_crown_512_yolo11n_seg_e50 exist_ok=True patience=20
```

현재 베이스라인은 36개 타일, 475개 instance polygon으로 학습했습니다. Patch 단위로 train/val/test를 나눴고, test split 기준 mask mAP50은 약 `0.353`, mask mAP50-95는 약 `0.134`입니다.

---

## 폴더 구조

```
├── notebooks/     탐색·실험용
├── src/           재사용 코드
├── data/          데이터 (gitignore)
└── outputs/       결과물 (gitignore)
```

---

> 문의: 나무한테 하세요.
