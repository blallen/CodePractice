function counting_sort(A, k)
    B = Array(zeros(Int, (length(A),))')
    C = zeros(Int, (k+1,))

    for j in 1:length(A)
        C[A[j]+1] += 1
    end # C[i+1] now contains number of elems equal to i
    
    for i in 1:k
        C[i+1] += C[i]
    end # C[i+1] now contains number of elems ≤ i
    
    for j in range(length(A), 1, step = -1)
        B[C[A[j]+1]] = A[j]
        C[A[j]+1] -= 1
    end # B is sorted version of A

    return B
end

function radix_sort!(A)
    max = maximum(A)

    place = 1
    while max // place > 0
        counting_sort!(A, place)
        place *= 10
    end
end

function counting_sort!(A, place)
    B = Array(zeros(Int, (length(A),))')
    C = zeros(Int, (10,))

    for j in 1:length(A)
        index = Int(floor(A[j] / place))
        digit = (index % 10) + 1
        C[digit] += 1
    end # C[i+1] now contains number of elems equal to i

    for i in 1:9
        C[i+1] += C[i]
    end # C[i+1] now contains number of elems ≤ i

    for j in range(length(A), 1, step = -1)
        index = Int(floor(A[j] / place))
        digit = (index % 10) + 1
        B[C[digit]] = A[j]
        C[digit] -= 1
    end # B is sorted version of A

    # copy
    A .= B
end