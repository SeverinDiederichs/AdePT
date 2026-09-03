#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

set -eu -o pipefail

# Check that a track starting in a removed region stays on the CPU. SpeedOfLight
# kills tracks sent to AdePT, so this track only deposits energy if it stays on
# the CPU.

ADEPT_EXECUTABLE=$1
PROJECT_SOURCE_DIR=$2
CI_TEST_DIR=$3
CI_TMP_DIR=$4

cleanup() {
  rm -rf "${CI_TMP_DIR}"
}
trap cleanup EXIT
cleanup
mkdir -p "${CI_TMP_DIR}"

MACRO_FILE="${CI_TMP_DIR}/test_remove_gpu_region.mac"
LOG_FILE="${CI_TMP_DIR}/test_remove_gpu_region.log"

"${CI_TEST_DIR}/python_scripts/macro_generator.py" \
  --template "${CI_TEST_DIR}/test_remove_gpu_region_template.mac" \
  --output "${MACRO_FILE}" \
  --gdml_name "${PROJECT_SOURCE_DIR}/examples/data/testEm3_regions.gdml"

if ! "${ADEPT_EXECUTABLE}" --allsensitive -m "${MACRO_FILE}" >"${LOG_FILE}" 2>&1; then
  cat "${LOG_FILE}"
  exit 1
fi

DEPOSITED_ENERGY=$(awk '/Total energy deposited:/ {print $(NF - 1); exit}' "${LOG_FILE}")
if ! awk -v energy="${DEPOSITED_ENERGY:-0}" 'BEGIN { exit !(energy > 0.0009) }'; then
  echo "Expected the 1 keV primary in removed region Layer3 to remain on CPU and deposit its energy."
  echo "Observed deposited energy: ${DEPOSITED_ENERGY:-missing} MeV"
  cat "${LOG_FILE}"
  exit 1
fi

echo "Initial track in removed region Layer3 remained on the CPU."
