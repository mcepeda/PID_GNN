#!/bin/bash

# python submit_jobs_train.py --config config_spread_300424.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/300424/dr_01/40_50/ --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/300424/dr_01/40_50/ --njobs 2000 --nev 100 --queue workday


# python submit_jobs_train.py --config config_pion0.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/single_particles_flat/pion0_0/ --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/single_particles_flat/pion0/ --idx 1 --njobs 1000 --nev 100 --queue workday

python submit_jobs_train.py --config config_pion.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/single_particles_flat/pion_1/ --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/single_particles_flat/pion_1/ --idx 0 --njobs 4000 --nev 100 --queue workday

# python submit_jobs_train.py --config config_electron.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/single_particles_flat/electron_0/ --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/single_particles_flat/electron/ --idx 2 --njobs 1000 --nev 100 --queue workday

# python submit_jobs_train.py --config config_muon.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/single_particles_flat/muon_0/ --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/single_particles_flat/muon/ --idx 3 --njobs 4000 --nev 100 --queue workday

python submit_jobs_train.py --config config_resonance.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/single_particles_flat/rho_1/ --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/single_particles_flat/rho_1/ --idx 4 --njobs 4000 --nev 100 --queue workday


