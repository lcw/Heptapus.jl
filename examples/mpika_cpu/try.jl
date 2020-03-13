using KernelAbstractions
using KernelAbstractions: CPUEvent, Event

function __wait(; dependencies=nothing)
  wait(CPU(), MultiEvent(dependencies), yield)
end

function main()
  event = CPUEvent(@async(__nonexistent!()))
  event = CPUEvent(@async(__wait(dependencies=event)))
  wait(event)
end
main()
