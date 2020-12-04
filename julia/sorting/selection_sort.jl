include("../utils.jl")

function selection_sort!(A)
    for i in 1:(length(A)-1)
        min_index = i

        for j in i+1:length(A)
            if A[j] < A[min_index]
                min_index = j
            end
        end

        exchange!(A, i, min_index)
    end
end