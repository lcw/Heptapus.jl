#!/bin/sh

#SBATCH --job-name=test_mpi_cudanative
#SBATCH --output=out_%j.txt
#SBATCH --partition=allgpu
#SBATCH --gres=gpu:titanv:2

#SBATCH --ntasks=2
#SBATCH --time=10:00
#SBATCH --mem=12GB

# Batch this job with `sbatch run_slrm.sh`

source /etc/profile
module load compile/gcc/7.2.0 openmpi/3.0.0 lib/cuda/10.1.243

# If launching with `srun` then replace `OMPI_COMM_WORLD_RANK` with
# `SLURM_PROCID`.

mpirun nvprof -o "timeline_job_%q{SLURM_JOBID}_rank_%q{OMPI_COMM_WORLD_RANK}" \
              --context-name "MPI Rank %q{OMPI_COMM_WORLD_RANK}" \
              --process-name "MPI Rank %q{OMPI_COMM_WORLD_RANK}" \
              --annotate-mpi openmpi \
              julia --project=. try.jl
