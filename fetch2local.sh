#!/bin/bash
set -uo pipefail

show_help() {
    echo "=========================================================="
    echo " [HELP] Pixel-NeRF Data Fetcher 사용법 "
    echo "----------------------------------------------------------"
    echo " 사용법: bash $0 -from <구글드라이브_경로> -to <해제할_경로>"
    echo ""
    echo " 필수 인자:"
    echo "  -from  : 공유폴더가 있는 Google Drive 경로"
    echo "  -to    : 압축을 해제하여 데이터를 저장할 로컬 경로"
    echo ""
    echo " 예시:"
    echo "  bash $0 -from /content/drive/MyDrive/data -to /content/data"
    echo "=========================================================="
    exit 1
}

# 인자 파싱 (옵션 기반)
INPUT_PATH=""
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -from)
            INPUT_PATH="$2"
            shift 2
            ;;
        -to)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            show_help
            ;;
    esac
done

# 인자 유효성 검사 
if [[ -z "$INPUT_PATH" || -z "$OUTPUT_PATH" ]]; then
    echo "[✕] ERROR: 필수 인자가 누락되었습니다."
    show_help
fi

# 구글 드라이브 경로 존재 확인
if [[ ! -d "$INPUT_PATH" ]]; then
    echo "[✕] ERROR: 입력 경로(from)를 찾을 수 없습니다: $INPUT_PATH"
    echo "    (드라이브가 마운트되었는지, 경로에 오타가 없는지 확인하세요.)"
    exit 1
fi

mkdir -p "$OUTPUT_PATH"

echo "=========================================="
echo " SOURCE (GDrive) : $INPUT_PATH"
echo " TARGET (Local)  : $OUTPUT_PATH"
echo "=========================================="

# ZIP 파일 목록 확보
ZIP_FILES=("$INPUT_PATH"/*.zip)

if [ ! -e "${ZIP_FILES[0]}" ]; then
    echo "[✕] ERROR: 해당 경로에 .zip 파일이 존재하지 않습니다."
    exit 1
fi

echo "▢ 압축 해제 시작 (약 10~12분 소요...)"

# 압축 해제 루프
for zip_file in "${ZIP_FILES[@]}"; do
    fname=$(basename "$zip_file")
    # -qn: 중복파일 건너뛰기, 정숙 모드로 I/O 부하 감소
    if unzip -qn "$zip_file" -d "$OUTPUT_PATH"; then
        echo "[✓] 완료: $fname"
    else
        echo "[✕] 실패: $fname"
    fi
done

echo "------------------------------------------"
echo "▢ 작업 완료: $OUTPUT_PATH"
echo "=========================================="