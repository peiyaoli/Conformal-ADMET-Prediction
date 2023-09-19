#!/bin/sh
# This submit jobs to HPC slurm
for dataset in HLM_Fang2023 hPPB_Lou2022 LD50_Lunghini2019 Lipophilicity_Wang2020 Permeability_Caco2_Wang2020 Permeability_MDCK_Fang2023 RLM_Fang2023 Solubility_Wang2020 VDss_Liu2022
#for dataset in HLM_Fang2023
do
    for ue in BASE DE MCD MVE EDL JMQR JQR
    do
        echo "Submit $dataset_$ue"
        #name="$dataset_$split_$fold_$ue_LDS"
        sbatch -J "BENCHING" --export=ALL,DATASET=$dataset,UE=$ue -o "./slurm_logs/${dataset}_${ue}.out" -e "./slurm_logs/${dataset}_${ue}.err" run_experiment_gpu_LDS_off.slurm
    done
done