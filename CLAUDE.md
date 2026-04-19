# CLAUDE.md — Project Context

## 연구자
- 이름: sebong (workingsebong)
- GitHub: github.com/workingsebong/RS-treecanopy

---

## 연구 개요

여름철 고해상도 항공 정사영상을 이용해 개별 나무를 탐지하고 **tree canopy instance segmentation** 기반 tree map을 생성하는 프로젝트.

- 개별 tree crown polygon mask 추출
- 나무 위치(centroid), 개수, canopy 면적 산출
- GIS 호환 벡터 파일 생성 (GeoJSON / Shapefile / GeoPackage)

---

## 파이프라인

```
항공 정사영상 (여름, 0.25~0.5m/px, RGB + NIR 권장)
    ↓ 512×512 또는 1024×1024 타일 분할
타일 이미지
    ↓ 개별 tree crown polygon 라벨링 (CVAT / Labelme / QGIS)
학습 데이터 (공간적으로 분리된 train/val/test)
    ↓ YOLO-seg 또는 Mask R-CNN 학습
Instance segmentation 모델
    ↓ 추론
개별 crown mask + centroid
    ↓ NDVI/NIR로 비식생 오탐 후처리
    ↓ 면적 계산: pixel_count × (pixel_size_m)²
Tree map (위치, 개수, canopy 면적)
    ↓ 벡터화
GeoJSON / Shapefile / GeoPackage
```

---

## 데이터 전략

### 테스트 데이터
- **V-World 항공사진** — 국토정보플랫폼, WMTS API로 타일 수집
  - 해상도: ~0.25m (줌 레벨 19 기준)
  - API 키 발급: vworld.kr → 개발자 센터

### 실제 분석 데이터
- **공간안전정보구역 항공사진** — 추후 확보 예정
  - RGB + NIR 구성 권장

### 데이터 수집 노트북
- `notebooks/01_tile_generation/01_vworld_tile_download.ipynb` — V-World WMTS 타일 수집 → GeoTIFF 병합
- `notebooks/01_tile_generation/02_split_geotiff_to_tiles.ipynb` — GeoTIFF → 1024×1024 라벨링 타일 분할

### 파일럿 결과 (2026-04-19 기준)
- 대상 지역: 서울 여의도 일부 (BBOX: 126.920~126.935, 37.520~37.530)
- 줌 레벨: 19 (≈0.25m/px)
- 결과 파일: `data/raw/vworld_aerial_test_area.tif`
  - 크기: 5888 × 5120 px / 98MB
  - 밴드: RGB 3채널
  - CRS: EPSG:4326
  - 픽셀 채움률: 99.9% (타일 누락 거의 없음)
- API 키: `.env` 파일로 관리 (`VWORLD_API_KEY`), `.gitignore`에 포함

---

## 모델링 전략

### 기본 추천
- **YOLO-seg** — 구현 빠름, 베이스라인용
- **Mask R-CNN** — 문헌 비교 용이, 해석 안정적

### 보조
- NDVI/NIR: canopy 마스크 경계 조정, 비식생 오탐 제거용 (primary가 아님)
- Semantic seg + Watershed: crown이 극히 조밀한 구역 후처리 옵션

---

## 라벨링 기준

- 줄기가 아닌 **수관 외곽 경계** 기준
- 겹치는 crown은 가능한 한 개별 객체로 분리
- 관목, 잔디, 그림자 제외
- 공간적 분리 기준으로 train/val/test 분할

---

## 예상 작업량

| 단계 | 라벨 수 | 목표 |
|------|--------|------|
| 빠른 실험 | 100~300 crown | 방법 작동 확인 |
| 초기 운영 | 500~2000 crown | 특정 지역 tree map |
| 안정 운영 | 수천+ | 다지역·다수종 일반화 |

추천 시작: 200개 라벨 → YOLO-seg 학습 → 오탐 유형 확인 → 부족 유형 추가 → 2~3회 반복

---

## 산출물

- 개별 나무 instance mask
- tree centroid (representative point)
- tree count
- canopy area 통계
- GIS 벡터 파일
- 시각화용 canopy map

---

## 폴더 구조

```
project_root/
├── CLAUDE.md                   ← 이 파일
├── TREE_CANOPY_PROJECT.md      ← 프로젝트 초안
├── environment.yml
├── .env                        ← API 키 (gitignore)
├── .gitignore
│
├── data/
│   ├── raw/                    ← 원본 항공영상 타일 (gitignore)
│   └── processed/              ← 중간 처리 파일, 라벨 (gitignore)
│
├── notebooks/                  ← 탐색/실험용
│   ├── 01_tile_generation/
│   ├── 02_model_training/
│   └── 03_inference_vectorize/
│
├── src/                        ← 재사용 확정 코드
│   ├── preprocessing/          ← 타일 분할, 전처리
│   ├── features/               ← NDVI 등 feature 추출
│   ├── modeling/               ← 모델 학습/추론 로직
│   └── visualization/          ← 시각화 함수
│
├── outputs/                    ← 결과물 (gitignore)
│   ├── figures/
│   ├── tables/
│   └── logs/
│
└── papers/                     ← 논문/발표 자료
    ├── references/
    ├── manuscript/
    ├── presentation/
    └── submission/
```

### 노트북 vs src 구분 원칙

- `notebooks/` — 탐색, 실험, 시각화 확인. 일회성 코드 허용
- `src/` — 반복 사용하는 확정 함수/클래스만. 노트북에서 import해서 사용

---

## 기술 스택

### 환경
- **OS**: Windows 11 + WSL2 (Ubuntu)
- **conda**: Miniconda3 (`~/miniconda3`, WSL 내부 설치)
- **conda 환경명**: `svi_segformer` (기존 환경 재사용 — 필요 패키지 모두 포함)
- **Python**: 3.10
- **GPU**: NVIDIA RTX 4070 SUPER (CUDA 사용 가능)
- **IDE**: VS Code (Remote - WSL 익스텐션으로 WSL 접속)

### 주요 패키지
- **원격탐사**: `rasterio`, `pyproj`
- **지리**: `geopandas`, `shapely`, `contextily`
- **딥러닝**: `torch`, `torchvision`, `transformers`
- **영상처리**: `opencv-python`, `pillow`
- **통계/분석**: `statsmodels`, `scipy`, `scikit-learn`
- **시각화**: `matplotlib`, `seaborn`
- **환경변수 관리**: `python-dotenv`

---

## 환경 설정

```bash
conda activate svi_segformer
python -m ipykernel install --user --name svi_segformer --display-name "Python (svi_segformer)"
```

### GPU 확인

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 작업 시 주의사항

1. **conda 환경 활성화 필수**: `conda activate svi_segformer`
2. **파일명 임의로 바꾸지 말 것** — 노트북 간 의존성
3. figure/plot title 및 annotation은 논문·발표 재사용을 위해 영어로 작성

---

## 업데이트 히스토리

- **2026-04-17**: 초기 CLAUDE.md 생성, 폴더 구조 세팅, 항공영상 기반 instance segmentation 방법론 추가
- **2026-04-19**: V-World WMTS 타일 수집 파일럿 완료 — 여의도 GeoTIFF 생성 확인 (5888×5120px, 0.25m/px), .env 키 관리 및 .gitignore 세팅
