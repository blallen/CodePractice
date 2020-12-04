using Test

include("insertion_sort.jl")
include("selection_sort.jl")
include("merge_sort.jl")
include("heap_sort.jl")

@testset "test sorting correctness" begin
    inputs = [
        [5 2 4 7 1 3 2 6],
        [4 1 3 2 16 9 10 14 8 7],
    ]
    outputs = [
        [1 2 2 3 4 5 6 7],
        [1 2 3 4 7 8 9 10 14 16 ],
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
            merge_sort!(A, 1, length(A))
            @test A == output
        end
    end
end

