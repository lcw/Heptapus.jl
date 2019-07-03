using StaticArrays, CUDAnative

function Base.sort!(a::MArray{Tuple{3}})
  # Use a (Bose-Nelson Algorithm based) sorting network from
  # <http://pages.ripco.net/~jgamble/nw.html>.
  a[2], a[3] = minmax(a[2], a[3])
  a[1], a[3] = minmax(a[1], a[3])
  a[1], a[2] = minmax(a[1], a[2])
end

function knl!()
  a = @MVector [7, 3, 4]
  sort!(a)
  @cuprintf("(%ld, %ld, %ld)\n", a[1], a[2], a[3])
  nothing
end

@cuda knl!()
