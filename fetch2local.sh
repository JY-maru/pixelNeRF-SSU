#!/bin/bash
# ==========================================
# Pixel-NeRF GCS Auto Downloader + Unzipper (Stable v4 - Pipe Fix)
# ------------------------------------------
# [수정 사항]
# 1. unzip | head 파이프라인에서 발생하는 SIGPIPE 에러 무시 처리 (|| true)
# 2. 루프 진입 시 디버깅용 메시지 출력 추가
# 3. set +e로 루프 내부의 사소한 에러로 인한 스크립트 중단 방지
# ==========================================

set -uo pipefail # -e 옵션 잠시 제거 (안전성 확보)

# 0) 인자 파싱
if [[ $# -lt 2 || "$1" != "-from" ]]; then
  echo "Usage: bash $0 -from <bucket/prefix> [-to <extract_path>]"
  exit 1
fi

INPUT_PATH="$2"
OUTPUT_PATH="${4:-/content/pixNeRF_shapeNet_v2_data}"   # 압축 해제 기본 경로
DOWNLOAD_BASE="/content/load_data"                      # ZIP 저장 전용 폴더

mkdir -p "$DOWNLOAD_BASE" "$OUTPUT_PATH"

# bucket / prefix 분리
BUCKET=$(echo "$INPUT_PATH" | cut -d'/' -f1)
PREFIX=$(echo "$INPUT_PATH" | cut -d'/' -f2-)
REMOTE_PATH="gs://${BUCKET}/${PREFIX}"

echo "=========================================="
echo "Pixel-NeRF GCS Fetcher (Stable v4)"
echo "------------------------------------------"
echo "BUCKET   : ${BUCKET}"
echo "PREFIX   : ${PREFIX}"
echo "SOURCE   : ${REMOTE_PATH}"
echo "ZIP SAVE : ${DOWNLOAD_BASE}"
echo "EXTRACT  : ${OUTPUT_PATH}"
echo "=========================================="

# 1) crcmod 최적화
if ! python3 -c "import crcmod" &>/dev/null; then
  echo "Installing crcmod..."
  pip install -q crcmod
else
  python3 - <<'PY' > /tmp/crcmod_type.txt
import crcmod, sys
print("C extension" if hasattr(crcmod, "_crcmod") else "Pure Python")
PY
  if grep -q "Pure Python" /tmp/crcmod_type.txt; then
    echo "Reinstalling crcmod with C-extension..."
    pip uninstall -y crcmod >/dev/null 2>&1 || true
    apt-get install -y python3-crcmod >/dev/null 2>&1 || true
  else
    echo "crcmod (C-extension) already installed."
  fi
fi

# 2) Colab 인증
IS_COLAB=$(python3 - <<'PY'
import sys
print('google.colab' in sys.modules)
PY
)

if [[ "$IS_COLAB" == "True" ]]; then
  echo "Authenticating Google Cloud (Colab detected)..."
  python3 - <<'PYCODE'
from google.colab import auth
auth.authenticate_user()
print("GCS authentication complete.")
PYCODE
else
  echo "Skipping Colab authentication (not detected)."
fi

# 3) ZIP 검색
echo "Scanning for ZIP files in ${REMOTE_PATH}..."
ZIP_LIST=$(gsutil ls "${REMOTE_PATH}/*.zip" 2>/dev/null || true)

if [[ -z "$ZIP_LIST" ]]; then
  echo "✕ No ZIP files found under ${REMOTE_PATH}"
  exit 1
fi

ZIP_COUNT=$(echo "$ZIP_LIST" | wc -l | tr -d ' ')
echo " [ Found ${ZIP_COUNT} ZIP files ]"
echo "$ZIP_LIST"
echo "------------------------------------------"

# 4) 다운로드
echo " [ Download ] ZIP files to ${DOWNLOAD_BASE}..."
echo "$ZIP_LIST" | tr -d '\r' | gsutil -m cp -n -I "${DOWNLOAD_BASE}/"
echo "✓ Download complete."

# 5) 압축 해제
echo "------------------------------------------"
echo "Extracting all ZIP archives..."

if [[ "$OUTPUT_PATH" == *"/drive/"* ]]; then
  echo "Google Drive detected → Sequential unzip (Smart Path Detection)"
  
  # 루프 시작 전 파일 확인
  shopt -s nullglob
  FILES=("${DOWNLOAD_BASE}"/*.zip)
  shopt -u nullglob

  if [ ${#FILES[@]} -eq 0 ]; then
      echo "✕ Error: No zip files found in ${DOWNLOAD_BASE}"
      exit 1
  fi

  for zipfile in "${FILES[@]}"; do
    name=$(basename "$zipfile" .zip)
    echo "▶ Processing: $name"  # 진행 상황 표시

    # [핵심 수정] 파이프 에러 무시 (|| true)
    first_entry=$(unzip -Z1 "$zipfile" 2>/dev/null | head -n 1 || true)
    
    # first_entry가 비어있으면(오류 등) 기본값 처리
    if [[ -z "$first_entry" ]]; then
        echo "[!]  Warning: Cannot read structure of $name. Extracting to subdir."
        dest="${OUTPUT_PATH}/${name}"
        mkdir -p "$dest"
    elif [[ "$first_entry" == "$name/"* ]]; then
        dest="${OUTPUT_PATH}"
        echo "   → Structure detected ($name/...) → Extracting to root."
    else
        dest="${OUTPUT_PATH}/${name}"
        mkdir -p "$dest"
        echo "   ▢ Flat structure detected → Extracting to subfolder: $name"
    fi

    # 압축 해제
    if unzip -q -o "$zipfile" -d "$dest"; then
        echo "   ✓ Done: $name"
    else
        echo "   ✕ Failed to extract: $name"
    fi
  done

else
  # 로컬/Colab VM 환경 (병렬 처리)
  echo "Using parallel unzip (3 processes)..."
  export OUTPUT_PATH
  find "${DOWNLOAD_BASE}" -maxdepth 1 -name "*.zip" | xargs -P 3 -I{} bash -c '
    zipfile="{}"
    name=$(basename "$zipfile" .zip)
    
    first_entry=$(unzip -Z1 "$zipfile" 2>/dev/null | head -n 1 || true)
    
    if [[ "$first_entry" == "$name/"* ]]; then
        dest="${OUTPUT_PATH}"
        echo "→ ($name) Direct extract..."
    else
        dest="${OUTPUT_PATH}/${name}"
        mkdir -p "$dest"
        echo "▢ ($name) Create folder & extract..."
    fi

    unzip -q -o "$zipfile" -d "$dest"
    echo "✓ Done: $name"
  '
fi

# 6) 결과 요약
echo "------------------------------------------"
echo "✓ All ZIPs extracted successfully."
du -sh "${OUTPUT_PATH}"/* 2>/dev/null | sort -h || echo "Check folder size manually."
echo "=========================================="
echo "ZIPs stored in : ${DOWNLOAD_BASE}/"
echo "Data ready at  : ${OUTPUT_PATH}/"
echo "=========================================="