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
- [x] 1024px 라벨링 후보 타일 생성
- [x] 프로젝트 전용 polygon 라벨링 앱 추가
- [ ] 라벨링 시작 (두렵다)
- [ ] 모델 학습
- [ ] Tree map 생성

---

## 라벨링 앱

```bash
conda activate svi_segformer
python tools/labeling_app/server.py --host 127.0.0.1 --port 8765
```

브라우저에서 엽니다.

```text
http://127.0.0.1:8765
```

입력은 `data/processed/labeling_candidates_1024/images`의 후보 타일이고, 저장하면 `data/processed/labels_1024` 아래에 JSON, YOLO-seg txt, GeoJSON이 같이 생깁니다.

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
