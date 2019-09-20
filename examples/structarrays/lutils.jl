
#    myforeachfield((col, val) -> (@inbounds col[I] = val), s, vals)

using StructArrays
import StructArrays: staticschema, map_params, astuple, _map_params
using Base: tuple_type_cons, tuple_type_head, tuple_type_tail, tail

function sstuple(::Type{<:NTuple{N, Any}}) where {N}
    ntuple(j->Symbol(j), N)
end

function sstuple(::Type{NT}) where {NT<:NamedTuple}
    _map_params(x->sstuple(staticschema(x)), NT)
end

function f(s, es, exprs)
    if typeof(s) <: Symbol
        push!(exprs, es)
        return
    end
    for key in keys(s)
        f(getproperty(s,key), vcat(es, key), exprs)
    end
end

using StaticArrays
c = [(a=SHermitianCompact(@SVector(rand(3))), b=(@SVector(rand(2)))) for i=1:5]
d = d = StructArray(c, unwrap = t -> t <: Union{SHermitianCompact,SVector,Tuple})
a = []
b = []
S = StructArrays.staticschema(typeof(d))
s = sstuple(S)
f(s, a, b)
@show b
