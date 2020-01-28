#!/bin/sh

mpirun -n 2 nvprof -o "timeline_rank_%q{OMPI_COMM_WORLD_RANK}" \
                   --context-name "MPI Rank %q{OMPI_COMM_WORLD_RANK}" \
                   --process-name "MPI Rank %q{OMPI_COMM_WORLD_RANK}" \
                   --annotate-mpi openmpi \
                   julia --project=. try.jl
