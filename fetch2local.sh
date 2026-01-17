#!/bin/bash
# ==========================================
# Pixel-NeRF Data Fetcher (Optimized Path)
# ==========================================

set -uo pipefail

# 0) 인자 파싱
if [[ $# -lt 2 || ("$1" != "-from_gcs" && "$1" != "-from_gdrive") ]]; then
  echo "Usage (GCS):    bash $0 -from_gcs <bucket/prefix> [-to <extract_path>]"
  echo "Usage (GDrive): bash $0 -from_gdrive <gdrive_path> [-to <extract_path>]"
  exit 1
fi

SOURCE_TYPE="$1"
INPUT_PATH="$2"
OUTPUT_PATH="${4:-/content/data}"
LOAD_DATA_TEMP="/content/load_data"

mkdir -p "$OUTPUT_PATH"

echo "=========================================="
echo "Pixel-NeRF Data Fetcher"
echo "------------------------------------------"
echo "SOURCE TYPE : ${SOURCE_TYPE#-from_}"
echo "INPUT PATH  : ${INPUT_PATH}"
echo "EXTRACT TO  : ${OUTPUT_PATH}"
echo "=========================================="

# 1) 소스별 로드 로직 분리
if [[ "$SOURCE_TYPE" == "-from_gcs" ]]; then
    # GCS: 임시 폴더로 먼저 로드 (rsync)
    mkdir -p "$LOAD_DATA_TEMP"
    BUCKET=$(echo "$INPUT_PATH" | cut -d'/' -f1)
    PREFIX=$(echo "$INPUT_PATH" | cut -d'/' -f2-)
    echo "▢ GCS에서 임시 폴더($LOAD_DATA_TEMP)로 로드 중..."
    gsutil -m rsync -r "gs://${BUCKET}/${PREFIX}" "$LOAD_DATA_TEMP"
    ZIP_SOURCE_DIR="$LOAD_DATA_TEMP"

elif [[ "$SOURCE_TYPE" == "-from_gdrive" ]]; then
    # GDrive: 임시 폴더 안 거치고 드라이브 경로 직접 지정
    if [[ ! -d "$INPUT_PATH" ]]; then
        echo "[✕] ERROR: 구글 드라이브 경로를 찾을 수 없습니다: $INPUT_PATH"
        exit 1
    fi
    echo "▢ 구글 드라이브 경로에서 압축 해제를 준비합니다."
    ZIP_SOURCE_DIR="$INPUT_PATH"
fi

# 2) 압축 해제 실행
ZIP_FILES=("$ZIP_SOURCE_DIR"/*.zip)
ZIP_COUNT=${#ZIP_FILES[@]}

if [[ $ZIP_COUNT -eq 0 ]]; then
    echo "[✕] 압축 파일을 찾을 수 없습니다: $ZIP_SOURCE_DIR"
    exit 1
fi

echo " ↪ 총 $ZIP_COUNT 개의 압축 해제 중..."

i=1
for zip_file in "${ZIP_FILES[@]}"; do
    fname=$(basename "$zip_file")
    echo -ne "[$i/$ZIP_COUNT] 압축 해제 중: $fname... \r"
    
    # -qn: 중복 무시, 조용히 실행 (디스크 I/O 최적화)
    unzip -qn "$zip_file" -d "$OUTPUT_PATH"
    
    if [[ $? -eq 0 ]]; then
        echo -e "[$i/$ZIP_COUNT] [✓] 완료: $fname                         "
    else
        echo -e "[$i/$ZIP_COUNT] [✕] 실패: $fname                         "
    fi
    ((i++))
done

# 3) 결과 요약 (폴더별 객체 수 확인)
echo "------------------------------------------"
echo "▢ 데이터 준비 완료: $OUTPUT_PATH"
echo "▢ 카테고리별 객체(Object) 요약:"

find "$OUTPUT_PATH" -maxdepth 1 -mindepth 1 -type d | while read -r dir; do
    category=$(basename "$dir")
    obj_count=$(find "$dir" -maxdepth 1 -mindepth 1 -type d | wc -l)
    echo "   ↳ $category: ${obj_count} 개의 객체"
done

echo "------------------------------------------"
echo "전체 파일 총합: $(find "$OUTPUT_PATH" -type f | wc -l) 개"
echo "=========================================="

# GCS 사용 후 임시 파일 정리 (선택 사항)
if [[ "$SOURCE_TYPE" == "-from_gcs" ]]; then
    rm -rf "$LOAD_DATA_TEMP"/*.zip
fi