using Test

include("insertion_sort.jl")
include("selection_sort.jl")
include("merge_sort.jl")
include("heap_sort.jl")
include("quick_sort.jl")
include("counting_sort.jl")

@testset "test sorting correctness" begin
    inputs = [
        [5 2 4 7 1 3 2 6],
        [4 1 3 2 16 9 10 14 8 7],
        [2 8 7 1 3 5 6 4],
        [2 5 3 0 2 3 0 3],
    ]
    outputs = [
        [1 2 2 3 4 5 6 7],
        [1 2 3 4 7 8 9 10 14 16],
        [1 2 3 4 5 6 7 8],
        [0 0 2 2 3 3 3 5],
    ]

    @testset "insertion sort" begin
        for (input, output) in zip(inputs, outputs)
            A = copy(input)
            insertion_sort!(A)
            @test A == output
        end
    end

    @testset "selection sort" begin
        for (input, output) in zip(inputs, outputs)
            A = copy(input)
            selection_sort!(A)
            @test A == output
        end
    end

    @testset "merge sort" begin
        for (input, output) in zip(inputs, outputs)
            A = copy(input)
            merge_sort!(A, 1, length(A))
            @test A == output
        end
    end

    @testset "heap sort" begin
        for (input, output) in zip(inputs, outputs)
            A = copy(input)
            heap_sort!(A)
            @test A == output
        end
    end

    @testset "quick sort" begin
        for (input, output) in zip(inputs, outputs)
            A = copy(input)
            quick_sort!(A, 1, length(A))
            @test A == output

            A = copy(input)
            randomized_quick_sort!(A, 1, length(A))
            @test A == output
        end
    end

    @testset "counting sort" begin
        for (input, output) in zip(inputs, outputs)
            A = copy(input)
            k = maximum(A)
            B = counting_sort(A, k)
            @test B == output
        end
    end
end

