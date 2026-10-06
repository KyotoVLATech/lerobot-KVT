#!/usr/bin/env bash
# Clone the Real-Time EXPO-FT OpenPI fork at the tested commit and apply the iLoHa config
# (expo_pi05_iloha_lora_finetune_sft_joint). Run from libs/expo-ft before `uv sync`.
#
# On hosts whose kernel headers predate KEY_LINK_PHONE, `uv sync` fails building evdev
# (a lerobot dependency the learner does not use); work around it with:
#   CFLAGS="-DKEY_LINK_PHONE=0x1bf" uv sync
set -euo pipefail

OPENPI_DIR=expo_ft/agents/vla/openpi
OPENPI_COMMIT=2abe46282bfdf9f1bc0240f3f9960ec175d1b4a8

if [ ! -d "$OPENPI_DIR/.git" ]; then
    git clone -b real-time-expo-ft https://github.com/pd-perry/openpi.git "$OPENPI_DIR"
fi
git -C "$OPENPI_DIR" checkout "$OPENPI_COMMIT"
if git -C "$OPENPI_DIR" apply --check "$PWD/patches/openpi-iloha-config.patch" 2>/dev/null; then
    git -C "$OPENPI_DIR" apply "$PWD/patches/openpi-iloha-config.patch"
    echo "Applied patches/openpi-iloha-config.patch"
else
    echo "patches/openpi-iloha-config.patch already applied (or conflicts); check $OPENPI_DIR"
fi
