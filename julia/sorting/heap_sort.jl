include("../utils.jl")

mutable struct Heap{A, H}
    array::A
    size::H
end

function Heap(A)
    return Heap(A, length(A))
end

size(heap) = heap.size

parent(i) = floor(Int, i/2)
left(i) = 2i
right(i) = 2i + 1

function max_heapify!(heap, i)
    l = left(i)
    r = right(i)

    largest = i
    A = heap.array

    if l ≤ size(heap) && A[l] > A[largest]
        largest = l
    end

    if r ≤ size(heap) && A[r] > A[largest]
        largest = r
    end

    if largest != i
        exchange!(A, i, largest)
        max_heapify!(heap, largest)
    end
end

function build_max_heap!(A)
    heap = Heap(A)

    start = floor(Int, size(heap) / 2)
    for i in range(start, 1, step = -1)
        max_heapify!(heap, i)
    end
    
    return heap
end

function heap_sort!(A)
    heap = build_max_heap!(A)
    for i in range(length(A), 2, step = -1)
        exchange!(heap.array, 1, i)
        heap.size -= 1
        max_heapify!(heap, 1)
    end
end