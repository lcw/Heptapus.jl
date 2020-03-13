using KernelAbstractions
using KernelAbstractions: CPUEvent

function __wait(; dependencies=nothing)
  wait(CPU(), MultiEvent(dependencies), yield)
end

event = CPUEvent(@async(__nonexistent!()))
event = CPUEvent(@async(__wait(dependencies=event)))
wait(event)
