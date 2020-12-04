function merge_sort!(A, start, stop)
    if start ≥ stop
        return
    end

    middle = floor(Int, (start + stop) / 2)
    merge_sort!(A, start, middle)
    merge_sort!(A, middle + 1, stop)
    merge!(A, start, middle, stop)
end

function merge!(A, start, middle, stop)
    n₁ = middle - start + 1
    n₂ = stop - middle

    L = zeros(Float64, (n₁+1,))
    R = zeros(Float64, (n₂+1,))

    for i in 1:n₁
        L[i] = A[start + i - 1]
    end

    for j in 1:n₂
        R[j] = A[middle + j]
    end

    L[n₁ + 1] = Inf
    R[n₂ + 1] = Inf

    i = j = 1

    for k in start:stop
        if L[i] ≤ R[j]
            A[k] = L[i]
            i += 1
        else
            A[k] = R[j]
            j += 1
        end
    end
end