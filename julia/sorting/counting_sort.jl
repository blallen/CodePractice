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