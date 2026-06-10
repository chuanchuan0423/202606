#!/usr/bin/env bash
set -euo pipefail

# Prepare AdaIR derain data from RainTrainL.zip and Rain100L.zip.
# This script removes existing AdaIR derain train/test folders, then rebuilds them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAIR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_ZIP="${ADAIR_ROOT}/data/RainTrainL.zip"
TEST_ZIP="${ADAIR_ROOT}/data/Rain100L.zip"

TRAIN_TMP="${ADAIR_ROOT}/data/_tmp_RainTrainL"
TEST_TMP="${ADAIR_ROOT}/data/_tmp_Rain100L"

TRAIN_DERAIN="${ADAIR_ROOT}/data/Train/Derain"
TRAIN_RAINY="${TRAIN_DERAIN}/rainy"
TRAIN_GT="${TRAIN_DERAIN}/gt"
TRAIN_LIST="${ADAIR_ROOT}/data_dir/rainy/rainTrain.txt"

TEST_DERAIN="${ADAIR_ROOT}/data/test/derain/Rain100L"
TEST_INPUT="${TEST_DERAIN}/input"
TEST_TARGET="${TEST_DERAIN}/target"

require_file() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "Missing required file: ${file}" >&2
    exit 1
  fi
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

echo "AdaIR root: ${ADAIR_ROOT}"
require_file "${TRAIN_ZIP}"
require_file "${TEST_ZIP}"
require_cmd unzip
require_cmd python3

echo "Cleaning old derain train/test folders..."
rm -rf "${TRAIN_DERAIN}" "${TEST_DERAIN}" "${TRAIN_TMP}" "${TEST_TMP}"

echo "Creating target folders..."
mkdir -p "${TRAIN_RAINY}" "${TRAIN_GT}" "${TEST_INPUT}" "${TEST_TARGET}" "$(dirname "${TRAIN_LIST}")"

echo "Unzipping RainTrainL..."
mkdir -p "${TRAIN_TMP}"
unzip -q "${TRAIN_ZIP}" -d "${TRAIN_TMP}"

echo "Unzipping Rain100L..."
mkdir -p "${TEST_TMP}"
unzip -q "${TEST_ZIP}" -d "${TEST_TMP}"

echo "Organizing train set: RainTrainL -> data/Train/Derain/{rainy,gt}"
python3 - <<'PY' "${ADAIR_ROOT}" "${TRAIN_TMP}" "${TRAIN_RAINY}" "${TRAIN_GT}" "${TRAIN_LIST}"
from pathlib import Path
import re
import shutil
import sys

adair_root = Path(sys.argv[1])
train_tmp = Path(sys.argv[2])
train_rainy = Path(sys.argv[3])
train_gt = Path(sys.argv[4])
train_list = Path(sys.argv[5])

src = train_tmp / "RainTrainL"
if not src.is_dir():
    raise SystemExit(f"Cannot find extracted RainTrainL folder: {src}")

def image_id(path: Path) -> int:
    match = re.search(r"-(\d+)\.png$", path.name)
    if not match:
        raise ValueError(f"Unexpected filename: {path.name}")
    return int(match.group(1))

rain_files = sorted(src.glob("rain-*.png"), key=image_id)
gt_files = sorted(src.glob("norain-*.png"), key=image_id)
rain_ids = {image_id(p): p for p in rain_files}
gt_ids = {image_id(p): p for p in gt_files}

missing_gt = sorted(set(rain_ids) - set(gt_ids))
missing_rain = sorted(set(gt_ids) - set(rain_ids))
if missing_gt or missing_rain:
    raise SystemExit(
        f"RainTrainL pairs mismatch. missing_gt={missing_gt[:10]}, missing_rain={missing_rain[:10]}"
    )

with train_list.open("w", encoding="utf-8") as f:
    for idx in sorted(rain_ids):
        rain_src = rain_ids[idx]
        gt_src = gt_ids[idx]
        rain_dst = train_rainy / f"rain-{idx}.png"
        gt_dst = train_gt / f"norain-{idx}.png"
        shutil.copy2(rain_src, rain_dst)
        shutil.copy2(gt_src, gt_dst)
        f.write(f"rainy/rain-{idx}.png\n")

print(f"Train pairs: {len(rain_ids)}")
print(f"Train list: {train_list.relative_to(adair_root)}")
PY

echo "Organizing test set: Rain100L -> data/test/derain/Rain100L/{input,target}"
python3 - <<'PY' "${ADAIR_ROOT}" "${TEST_TMP}" "${TEST_INPUT}" "${TEST_TARGET}"
from pathlib import Path
import re
import shutil
import sys

adair_root = Path(sys.argv[1])
test_tmp = Path(sys.argv[2])
test_input = Path(sys.argv[3])
test_target = Path(sys.argv[4])

src = test_tmp / "Rain100L"
if not src.is_dir():
    raise SystemExit(f"Cannot find extracted Rain100L folder: {src}")

def image_id(path: Path) -> int:
    match = re.search(r"-(\d+)\.png$", path.name)
    if not match:
        raise ValueError(f"Unexpected filename: {path.name}")
    return int(match.group(1))

# Rain100L stores rainy inputs under Rain100L/rainy/ in some releases.
rain_files = sorted(
    [p for p in src.rglob("rain-*.png") if p.name.startswith("rain-")],
    key=image_id,
)
gt_files = sorted(src.glob("norain-*.png"), key=image_id)
rain_ids = {image_id(p): p for p in rain_files}
gt_ids = {image_id(p): p for p in gt_files}

missing_gt = sorted(set(rain_ids) - set(gt_ids))
missing_rain = sorted(set(gt_ids) - set(rain_ids))
if missing_gt or missing_rain:
    raise SystemExit(
        f"Rain100L pairs mismatch. missing_gt={missing_gt[:10]}, missing_rain={missing_rain[:10]}"
    )

for idx in sorted(rain_ids):
    name = f"{idx}.png"
    shutil.copy2(rain_ids[idx], test_input / name)
    shutil.copy2(gt_ids[idx], test_target / name)

print(f"Test pairs: {len(rain_ids)}")
print(f"Test input: {test_input.relative_to(adair_root)}")
print(f"Test target: {test_target.relative_to(adair_root)}")
PY

echo "Verifying prepared derain data..."
python3 - <<'PY' "${ADAIR_ROOT}" "${TRAIN_RAINY}" "${TRAIN_GT}" "${TRAIN_LIST}" "${TEST_INPUT}" "${TEST_TARGET}"
from pathlib import Path
import sys

adair_root = Path(sys.argv[1])
train_rainy = Path(sys.argv[2])
train_gt = Path(sys.argv[3])
train_list = Path(sys.argv[4])
test_input = Path(sys.argv[5])
test_target = Path(sys.argv[6])

lines = [line.strip() for line in train_list.read_text(encoding="utf-8").splitlines() if line.strip()]
missing = []
for line in lines:
    rain_path = train_rainy.parent / line
    gt_path = Path(str(rain_path).replace("/rainy/rain-", "/gt/norain-"))
    if not rain_path.exists() or not gt_path.exists():
        missing.append((rain_path, gt_path))

train_rain_count = len(list(train_rainy.glob("*.png")))
train_gt_count = len(list(train_gt.glob("*.png")))
test_input_count = len(list(test_input.glob("*.png")))
test_target_count = len(list(test_target.glob("*.png")))

print(f"Train rainy images: {train_rain_count}")
print(f"Train gt images: {train_gt_count}")
print(f"Train list entries: {len(lines)}")
print(f"Missing train pairs: {len(missing)}")
print(f"Test input images: {test_input_count}")
print(f"Test target images: {test_target_count}")

if missing:
    raise SystemExit(f"Missing train pair sample: {missing[:3]}")
if train_rain_count != train_gt_count or train_rain_count != len(lines):
    raise SystemExit("Train image count and rainTrain.txt count do not match.")
if test_input_count != test_target_count:
    raise SystemExit("Test input/target image counts do not match.")

print("Derain data is ready.")
print(f"Train folder: {(train_rainy.parent).relative_to(adair_root)}")
print(f"Test folder: {(test_input.parent).relative_to(adair_root)}")
PY

echo "Cleaning temporary folders..."
rm -rf "${TRAIN_TMP}" "${TEST_TMP}"

echo "Done."
