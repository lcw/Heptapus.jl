
#    myforeachfield((col, val) -> (@inbounds col[I] = val), s, vals)

using StructArrays
import StructArrays: staticschema, map_params, astuple, _map_params
using Base: tuple_type_cons, tuple_type_head, tuple_type_tail, tail

function _sstuple(::Type{<:NTuple{N, Any}}) where {N}
    ntuple(j->Symbol(j), N)
end

function _sstuple(::Type{NT}) where {NT<:NamedTuple}
    _map_params(x->_sstuple(staticschema(x)), NT)
end

function _getcolproperties!(exprs, s, es=[])
    if typeof(s) <: Symbol
        push!(exprs, es)
        return
    end
    for key in keys(s)
        _getcolproperties!(exprs, getproperty(s,key), vcat(es, key))
    end
end

using StaticArrays
c = [(a=SHermitianCompact(@SVector(rand(3))), b=(@SVector(rand(2)))) for i=1:5]
d = StructArray(c, unwrap = t -> t <: Union{SHermitianCompact,SVector,Tuple})
b = []
S = StructArrays.staticschema(typeof(d))
s = _sstuple(S)
_getcolproperties!(b, s)
@show b

function f(columnsproperties, xs)
  exprs = Expr[]
  for col in columnsproperties
    args = Expr[]
    for prop in col
      if length(args) == 0
        args = [Expr(:call, :_getproperty, :(getfield(xs, $j)), prop) for j in 1:length(xs)]
      else
        for j in 1:length(xs)
          args[j] = Expr(:call, :_getproperty, args[j], prop)
        end
      end
    end
    push!(exprs, Expr(:call, :f, args...))
  end
  push!(exprs, :(return nothing))
  return Expr(:block, exprs...)
end
