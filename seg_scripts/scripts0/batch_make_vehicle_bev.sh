#!/usr/bin/env bash

set -euo pipefail

# Batch generate vehicle-frame BEV images from extracted raw frames.
#
# Usage:
#   bash batch_make_vehicle_bev.sh
#   bash batch_make_vehicle_bev.sh --debug
#   bash batch_make_vehicle_bev.sh 2026-06-28_01-05-54
#   bash batch_make_vehicle_bev.sh --debug 2026-06-28_01-05-54 2026-06-28_01-08-45
#
# Notes:
#   - Input must be raw extracted images, not already-undistorted images.
#   - apply_vehicle_bev.py performs undistortion internally.
#   - These parameters match the verified vehicle_bev_extrinsic.yaml:
#       align=camera-optical, origin=camera-ground, flip-left
#       forward=[0.0, 1.2] m, left=[-0.6, 0.6] m, ppm=266.6667
#       output size: 320x320

WS="${HOME}/Desktop/relocate_ws"

SCRIPT="${WS}/scripts/apply_vehicle_bev.py"
EXTRINSIC="${WS}/data/calib/ground_extrinsic.yaml"

RAW_ROOT="${WS}/data/extracted/frames_raw"
OUT_ROOT="${WS}/data/extracted/bev_vehicle"

ALIGN="camera-optical"
ORIGIN="camera-ground"

FORWARD_MIN="0.0"
FORWARD_MAX="1.2"
LEFT_MIN="-0.6"
LEFT_MAX="0.6"
PPM="266.6667"
GRID_STEP="0.1"

DEBUG_GRID=0
BAG_NAMES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG_GRID=1
            shift
            ;;
        --help|-h)
            echo "Usage:"
            echo "  bash $0"
            echo "  bash $0 --debug"
            echo "  bash $0 2026-06-28_01-05-54"
            echo "  bash $0 --debug 2026-06-28_01-05-54 2026-06-28_01-08-45"
            exit 0
            ;;
        *)
            BAG_NAMES+=("$1")
            shift
            ;;
    esac
done

if [[ ! -f "$SCRIPT" ]]; then
    echo "[ERROR] Cannot find apply_vehicle_bev.py:"
    echo "        $SCRIPT"
    exit 1
fi

if [[ ! -f "$EXTRINSIC" ]]; then
    echo "[ERROR] Cannot find ground_extrinsic.yaml:"
    echo "        $EXTRINSIC"
    exit 1
fi

if [[ ! -d "$RAW_ROOT" ]]; then
    echo "[ERROR] Cannot find raw frames root:"
    echo "        $RAW_ROOT"
    exit 1
fi

mkdir -p "$OUT_ROOT"

echo "========================================"
echo "Vehicle BEV batch generation"
echo "Workspace:        $WS"
echo "Script:           $SCRIPT"
echo "Extrinsic:        $EXTRINSIC"
echo "Raw root:         $RAW_ROOT"
echo "Output root:      $OUT_ROOT"
echo "Align:            $ALIGN"
echo "Origin:           $ORIGIN"
echo "Forward range:    ${FORWARD_MIN} ~ ${FORWARD_MAX} m"
echo "Left range:       ${LEFT_MIN} ~ ${LEFT_MAX} m"
echo "PPM:              $PPM"
echo "Flip left:        ON"
echo "Debug grid:       $DEBUG_GRID"
echo "========================================"

process_one_bag() {
    local bag_name="$1"
    local input_dir="${RAW_ROOT}/${bag_name}"
    local output_dir="${OUT_ROOT}/${bag_name}"

    if [[ ! -d "$input_dir" ]]; then
        echo "[WARN] Skip missing input dir: $input_dir"
        return 0
    fi

    local count
    count=$(find "$input_dir" -maxdepth 1 \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)

    if [[ "$count" -eq 0 ]]; then
        echo "[WARN] Skip empty image dir: $input_dir"
        return 0
    fi

    echo
    echo "----------------------------------------"
    echo "[INFO] Processing bag: $bag_name"
    echo "[INFO] Input images: $count"
    echo "[INFO] Input:  $input_dir"
    echo "[INFO] Output: $output_dir"

    cmd=(
        python3 "$SCRIPT"
        --extrinsic "$EXTRINSIC"
        --input "$input_dir"
        --output "$output_dir"
        --align "$ALIGN"
        --origin "$ORIGIN"
        --forward-min "$FORWARD_MIN"
        --forward-max "$FORWARD_MAX"
        --left-min "$LEFT_MIN"
        --left-max "$LEFT_MAX"
        --ppm "$PPM"
        --grid-step "$GRID_STEP"
        --flip-left
        --no-undistorted-save
    )

    if [[ "$DEBUG_GRID" -eq 1 ]]; then
        cmd+=(--draw-grid)
    fi

    "${cmd[@]}"

    local bev_dir="${output_dir}/bev"
    if [[ -d "$bev_dir" ]]; then
        local bev_count
        bev_count=$(find "$bev_dir" -maxdepth 1 \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l)
        echo "[INFO] BEV images generated: $bev_count"
    else
        echo "[WARN] No BEV dir generated: $bev_dir"
    fi
}

if [[ "${#BAG_NAMES[@]}" -gt 0 ]]; then
    for bag_name in "${BAG_NAMES[@]}"; do
        process_one_bag "$bag_name"
    done
else
    shopt -s nullglob
    for input_dir in "$RAW_ROOT"/*; do
        if [[ -d "$input_dir" ]]; then
            process_one_bag "$(basename "$input_dir")"
        fi
    done
fi

echo
echo "========================================"
echo "[DONE] All requested BEV processing finished."
echo "Output root:"
echo "  $OUT_ROOT"
echo
echo "Check counts:"
echo "  for d in \"$OUT_ROOT\"/*/bev; do echo \"\$(basename \"\$(dirname \"\$d\")\"): \$(find \"\$d\" -maxdepth 1 -iname '*.png' | wc -l)\"; done"
echo "========================================"
