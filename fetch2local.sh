#!/bin/bash
# ==========================================
# Pixel-NeRF GCS Auto Downloader + Unzipper (Stable v2)
# ------------------------------------------
# 기능 요약:
#   - GCS의 ZIP 파일을 /content/load_data 에 임시 다운로드
#   - 압축 해제 결과를 -to 인자로 지정한 경로에 저장
#   - Colab 환경 자동 감지 및 GCS 인증
#   - crcmod (C-extension) 최적화 설치
# ==========================================

set -euo pipefail

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
echo "Pixel-NeRF GCS Fetcher (Stable v2)"
echo "------------------------------------------"
echo "BUCKET   : ${BUCKET}"
echo "PREFIX   : ${PREFIX}"
echo "SOURCE   : ${REMOTE_PATH}"
echo "ZIP SAVE : ${DOWNLOAD_BASE}"
echo "EXTRACT  : ${OUTPUT_PATH}"
echo "=========================================="

# 1) crcmod 최적화 (빠른 다운로드용)
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

# 2) Colab 환경 감지 및 GCS 인증
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

# 3) ZIP 파일 검색
echo "Scanning for ZIP files in ${REMOTE_PATH}..."
ZIP_LIST=$(gsutil ls "${REMOTE_PATH}/*.zip" 2>/dev/null || true)

if [[ -z "$ZIP_LIST" ]]; then
  echo "❌ No ZIP files found under ${REMOTE_PATH}"
  exit 1
fi

ZIP_COUNT=$(echo "$ZIP_LIST" | wc -l | tr -d ' ')
echo " [ Found ${ZIP_COUNT} ZIP files ]"
echo "$ZIP_LIST"
echo "------------------------------------------"

# 4) ZIP 다운로드
echo " [ Download ] ZIP files to ${DOWNLOAD_BASE}..."
echo "$ZIP_LIST" | tr -d '\r' | gsutil -m cp -n -I "${DOWNLOAD_BASE}/"
echo "✅ Download complete."

# 5) 압축 해제
echo "------------------------------------------"
echo "Extracting all ZIP archives..."

if [[ "$OUTPUT_PATH" == *"/drive/"* ]]; then
  echo "Google Drive detected → Sequential unzip (for I/O safety)"
  for zipfile in "${DOWNLOAD_BASE}"/*.zip; do
    [[ -f "$zipfile" ]] || continue
    name=$(basename "$zipfile" .zip)
    dest="${OUTPUT_PATH}/${name}"
    unzip -tq "$zipfile" >/dev/null 2>&1 || { echo "Skipping corrupted: $zipfile"; continue; }
    mkdir -p "$dest"
    echo "Extracting $zipfile → $dest"
    unzip -q -o "$zipfile" -d "$dest"
    echo "Done: $name"
  done
else
  echo "Using parallel unzip (3 processes)..."
  export OUTPUT_PATH
  find "${DOWNLOAD_BASE}" -maxdepth 1 -name "*.zip" | xargs -P 3 -I{} bash -c '
    zipfile="{}"
    name=$(basename "$zipfile" .zip)
    dest="${OUTPUT_PATH}/${name}"
    mkdir -p "$dest"

    first_entry=$(unzip -Z1 "$zipfile" | head -1)
    if [[ "$first_entry" == "$name/"* ]]; then
      tmp_dir="${dest}_tmp"
      mkdir -p "$tmp_dir"
      unzip -q -o "$zipfile" -d "$tmp_dir"
      shopt -s dotglob
      mv "$tmp_dir/$name"/* "$dest"/
      rm -rf "$tmp_dir"
      shopt -u dotglob
    else
      unzip -q -o "$zipfile" -d "$dest"
    fi
    echo "✅ Done: $name"
  '
fi

# 6) 결과 요약
echo "------------------------------------------"
echo "✅ All ZIPs extracted successfully."
du -sh "${OUTPUT_PATH}"/* | sort -h || true
echo "=========================================="
echo "ZIPs stored in : ${DOWNLOAD_BASE}/"
echo "Data ready at  : ${OUTPUT_PATH}/"
echo "=========================================="