include("../utils.jl")

function quick_sort!(A, start, stop)
    if start < stop
        mid = partition(A, start, stop)
        quick_sort!(A, start, mid - 1)
        quick_sort!(A, mid + 1, stop)
    end
end

function partition(A, start, stop)
    pivot = A[stop]
    i = start - 1
    for j in start:(stop-1)
        if A[j] ≤ pivot
            i += 1
            exchange!(A, i, j)
        end
    end
    
    exchange!(A, i+1, stop)

    return i+1
end

function randomized_partition!(A, start, stop)
    pivot = rand(start:stop)
    exchange!(A, pivot, stop)
    return partition(A, start, stop)
end

function randomized_quick_sort!(A, start, stop)
    if start < stop
        mid = randomized_partition!(A, start, stop)
        randomized_quick_sort!(A, start, mid - 1)
        randomized_quick_sort!(A, mid + 1, stop)
    end
end