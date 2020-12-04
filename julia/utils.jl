function exchange!(A, i, j)
    tmp = A[i]
    A[i] = A[j]
    A[j] = tmp
end