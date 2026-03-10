#!/bin/bash

function copy_to_tmp {
    local bids_dir=$1
    local bids_dir_tmp=$2
    local subjectID=$3
    local sessionID=$4
    local preproc_dir_name=$5
    local -a der_dir_to_copy=("${@:6}")  # All other arguments are assumed to be directories to copy

    local derivatives_dir="${bids_dir_tmp}/derivatives"
    local ssD="sub-${subjectID}/ses-${sessionID}"

    echo "bids_dir_tmp=${bids_dir_tmp}"
    mkdir -p "${derivatives_dir}/${preproc_dir_name}/${ssD}"
    echo -e "Copying files to scratch..."

    for der_dir in "${der_dir_to_copy[@]}"; do 
        echo "copying ${der_dir}..."
        mkdir -p "${derivatives_dir}/${der_dir}/${ssD}"
        cp -r \
            "${bids_dir}/derivatives/${der_dir}/${ssD}" \
            "${derivatives_dir}/${der_dir}/sub-${subjectID}"
        cp "${bids_dir}/derivatives/${der_dir}/dataset_description.json" \
            "${bids_dir_tmp}/derivatives/${der_dir}"
    done
    cp "${bids_dir}/dataset_description.json" "${bids_dir_tmp}"
}

function check_dwi_data {
    local bids_dir=$1
    local preproc_dir_name=$2
    local ssD=$3

    local dwi_path_pattern="${bids_dir}/derivatives/${preproc_dir_name}/${ssD}/dwi/*_dwi.nii.gz"
    echo "Checking for DWI data at ${dwi_path_pattern}..."
    
    if ls ${dwi_path_pattern} 1> /dev/null 2>&1; then
        echo "DWI data found, continuing."
    else
        echo "No DWI data found, exiting."
        exit 2
    fi
}
