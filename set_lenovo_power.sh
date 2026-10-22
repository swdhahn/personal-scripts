#!/bin/bash

# Define the exact base path from your GUI logs
LEGION_DIR="/sys/module/legion_laptop/drivers/platform:legion/legion"

# 1. Enable Custom Power Mode
# Writing 'custom' to the platform profile activates the mode needed for these limits
echo "custom" > "$LEGION_DIR/platform-profile/platform-profile-0/profile"

# Wait a second for the hardware state to transition
sleep 1

# 2. Apply CPU Limits [W]
echo 30 > "$LEGION_DIR/cpu_longterm_powerlimit"
echo 35 > "$LEGION_DIR/cpu_shortterm_powerlimit"
echo 0  > "$LEGION_DIR/cpu_peak_powerlimit"
echo 0  > "$LEGION_DIR/cpu_cross_loading_powerlimit"
echo 0  > "$LEGION_DIR/cpu_apu_sppt_powerlimit"

# 3. Apply GPU Limits [W] and [°C]
echo 10 > "$LEGION_DIR/gpu_ctgp_powerlimit"
echo 15 > "$LEGION_DIR/gpu_ppab_powerlimit"
echo 0  > "$LEGION_DIR/gpu_temperature_limit"

echo "Legion custom power settings applied successfully."
