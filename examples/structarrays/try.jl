using CuArrays, StaticArrays, StructArrays

aofs = [SHermitianCompact(@SVector(rand(3))) for i=1:3]
sofa = StructArray(aofs, unwrap = t -> t <: Union{SVector,Tuple})
