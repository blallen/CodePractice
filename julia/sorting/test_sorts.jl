using Test

include("insertion_sort.jl")
include("selection_sort.jl")
include("merge_sort.jl")

@testset "test sorting correctness" begin
    input = [5 2 4 7 1 3 2 6]
    output = [1 2 2 3 4 5 6 7]

    @testset "insertion sort" begin
        A = input
        insertion_sort!(A)
        @test A == output
    end

    @testset "selection sort" begin
        A = input
        selection_sort!(A)
        @test A == output
    end

    @testset "merge sort" begin
        A = input
        merge_sort!(A, 1, length(A))
        @test A == output
    end
end