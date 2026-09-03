#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

# Compare two ways of selecting only the calorimeter for GPU tracking.
# testEm3.gdml puts the calorimeter in `caloregion` and the surrounding volume
# in `DefaultRegionForTheWorld`:
#
#     /adept/addGPURegion caloregion
#
#   and:
#
#     /adept/setTrackInAllRegions true
#     /adept/removeGPURegion DefaultRegionForTheWorld
#
# Both runs use the same seed. Their per-volume energy deposits must match.

set -eu -o pipefail

ADEPT_EXECUTABLE=$1
PROJECT_SOURCE_DIR=$2
CI_TEST_DIR=$3
CI_TMP_DIR=$4

cleanup() {
  rm -rf "${CI_TMP_DIR}"
}
trap cleanup EXIT
cleanup

EXPLICIT_DIR="${CI_TMP_DIR}/explicit"
TRACK_ALL_DIR="${CI_TMP_DIR}/track_all"
mkdir -p "${EXPLICIT_DIR}" "${TRACK_ALL_DIR}"

generate_macro() {
  local output=$1
  local track_in_all_regions=$2
  local added_regions=$3
  local removed_regions=$4

  "${CI_TEST_DIR}/python_scripts/macro_generator.py" \
    --template "${CI_TEST_DIR}/example_template.mac" \
    --output "${output}" \
    --gdml_name "${PROJECT_SOURCE_DIR}/examples/data/testEm3.gdml" \
    --num_threads 1 \
    --num_events 1 \
    --num_trackslots 1 \
    --num_hitslots 1 \
    --gun_number 1 \
    --gun_type setDefault \
    --track_in_all_regions "${track_in_all_regions}" \
    --regions "${added_regions}" \
    --removed_regions "${removed_regions}"
}

EXPLICIT_MACRO="${EXPLICIT_DIR}/explicit_add.mac"
TRACK_ALL_MACRO="${TRACK_ALL_DIR}/track_all_remove.mac"

generate_macro "${EXPLICIT_MACRO}" false caloregion ""
generate_macro "${TRACK_ALL_MACRO}" true "" DefaultRegionForTheWorld

grep -Fxq "/adept/addGPURegion caloregion" "${EXPLICIT_MACRO}"
grep -Fxq "/adept/setTrackInAllRegions true" "${TRACK_ALL_MACRO}"
grep -Fxq "/adept/removeGPURegion DefaultRegionForTheWorld" "${TRACK_ALL_MACRO}"

run_configuration() {
  local macro=$1
  local output_dir=$2
  local output_name=$3
  local log=$4

  if ! "${ADEPT_EXECUTABLE}" --allsensitive --accumulated_events \
      -m "${macro}" --output_dir "${output_dir}" --output_file "${output_name}" \
      >"${log}" 2>&1; then
    cat "${log}"
    exit 1
  fi
}

run_configuration "${EXPLICIT_MACRO}" "${EXPLICIT_DIR}" explicit_add "${EXPLICIT_DIR}/explicit_add.log"
run_configuration "${TRACK_ALL_MACRO}" "${TRACK_ALL_DIR}" track_all_remove "${TRACK_ALL_DIR}/track_all_remove.log"

"${CI_TEST_DIR}/python_scripts/check_reproducibility.py" \
  --file1 "${EXPLICIT_DIR}/explicit_add.csv" \
  --file2 "${TRACK_ALL_DIR}/track_all_remove.csv" \
  --tol 0.0

echo "Explicit addGPURegion and track-all/removeGPURegion produced identical scored results."
