#!/bin/sh

#SBATCH --job-name=test_mpi_cudanative
#SBATCH --output=out_%j.txt
#SBATCH --partition=allgpu
#SBATCH --gres=gpu:titanv:2

#SBATCH --ntasks=2
#SBATCH --time=10:00
#SBATCH --mem=12GB

source /etc/profile
module load compile/gcc/7.2.0 openmpi/3.0.0 lib/cuda/10.1.243

srun nvprof -o "timeline_rank_%q{SLURM_PROCID}_job_%q{SLURM_JOBID}" \
            --context-name "MPI Rank %q{SLURM_PROCID}" \
            --process-name "MPI Rank %q{SLURM_PROCID}" \
            --annotate-mpi openmpi \
            julia --project=. try.jl
