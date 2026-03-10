#!/bin/bash

base_dir=/data/rauschecker1/jkw/projects/alzheimers/aria/aria_prediction_modeling/data/ucsf_aria
bids_dir=${base_dir}/data/BIDS
dcm_dir=${base_dir}/data/dcm
mkdir -p ${bids_dir}
mkdir -p ${dcm_dir}
download_csv=${base_dir}/2025-07-08_aria-accessions.csv

# download each accession in csv file
while IFS=, read accession_number rename_subjectID rename_sessionID
do
    echo "Downloading ${accession_number}"
    if [ -d "${dcm_dir}/sub-${rename_subjectID}/ses-${rename_sessionID}" ]; then
        echo "Already downloaded ${accession_number}"
        continue
    fi
    python /data/rauschecker1/utils/common_tools/run_air-download.py -pf "490" -pj "171" \
        ${accession_number} -o ${dcm_dir} \
        2>&1 | tee ${dcm_dir}/download_${accession_number}.log

    sbatch --job-name="sequence_classifier_${rename_subjectID}_${rename_sessionID}" \
            /data/rauschecker1/jkw/projects/alzheimers/aria/aria_prediction_modeling/dev/datagen/run_classify.qsh ${dcm_dir} ${accession_number} ${rename_subjectID} ${rename_sessionID} ${bids_dir}

done < ${download_csv}