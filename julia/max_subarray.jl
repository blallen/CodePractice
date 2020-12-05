function find_max_subarray(A, start, finish)
    max_sum = -Inf
    sum = 0
    max_index = -1

    start > finish ? step = -1 : step = 1

    for i in range(start, finish, step = step)
        sum += A[i]
        if sum > max_sum
            max_sum = sum
            max_index = i
        end
    end

    return (max_index, max_sum)
end

function find_max_crossing_subarray(A, low, mid, high)
    max_left, left_sum = find_max_subarray(A, mid, low)
    max_right, right_sum = find_max_subarray(A, mid+1, high)

    return (max_left, max_right, left_sum + right_sum)
end

function find_maximum_subarray(A, low, high)
    # base case only one elem
    if high == low
        return (low, high, A[low])
    end

    mid = floor(Int, (low + high) / 2)

    left_low, left_high, left_sum = find_maximum_subarray(A, low, mid)
    right_low, right_high, right_sum = find_maximum_subarray(A, mid+1, high)
    x_low, x_high, x_sum = find_max_crossing_subarray(A, low, mid, high)

    if left_sum ≥ right_sum && left_sum ≥ x_sum
        return (left_low, left_high, left_sum)
    elseif right_sum ≥ left_sum && right_sum ≥ x_sum
        return (right_low, right_high, right_sum)
    else
        return (x_low, x_high, x_sum)
    end
end

using Test

@testset "max sub array" begin
    input = [13 -3 -25 20 -3 -16 -23 18 20 -7 12 -5 -22 15 -4 7]
    output = (8, 11, 43)
    @test find_maximum_subarray(input, 1, length(input)) == output
end