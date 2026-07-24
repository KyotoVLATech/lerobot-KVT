#!/bin/bash

# Number of GPUs to find (default to 2 if not provided)
num_gpus_needed=${1:-2}

# Define an array of GPU IDs to exclude
exclude_gpus=() # Add the GPU IDs you want to exclude

# Function to check if an array contains a value
containsElement () {
  for e in "${@:2}"; do
    if [[ "$e" == "$1" ]]; then
      return 0
    fi
  done
  return 1
}

# Get GPU usage
gpu_usage=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n)

# Initialize array for available GPUs
available_gpus=()

# Iterate over each GPU (sorted by memory usage)
while IFS=, read -r gpu_id usage
do
    # Check if this GPU is in the exclude list
    if containsElement "$gpu_id" "${exclude_gpus[@]}"; then
        continue
    fi
    
    # Add to available GPUs
    available_gpus+=($gpu_id)
    
    # Break if we have enough GPUs
    if [ ${#available_gpus[@]} -eq $num_gpus_needed ]; then
        break
    fi
done <<< "$gpu_usage"

# Check if we found enough GPUs
if [ ${#available_gpus[@]} -lt $num_gpus_needed ]; then
    echo "Error: Only found ${#available_gpus[@]} available GPUs, but need $num_gpus_needed" >&2
    exit 1
fi

# Output the GPU indices as comma-separated list
IFS=','
echo "${available_gpus[*]}"