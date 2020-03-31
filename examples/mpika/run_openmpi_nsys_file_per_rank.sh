#!/bin/sh

mpirun -np 2 --output-filename output \
    nsys profile --output="profile-%q{OMPI_COMM_WORLD_RANK}" --trace=cuda,nvtx,mpi \
                 --mpi-impl=openmpi --stats=true -e _="$(command -v nsys)" \
        julia --project=. try.jl
