using Test

include("heap_sort.jl")

@testset "test helpers" begin
    @testset "max_heapify!" begin
        input = [16 4 10 14 7 9 3 2 8 1]
        output = [16 14 10 8 7 9 3 2 4 1]
        heap = Heap(input)
        max_heapify!(heap, 2)
        @test heap.array == output
    end

    @testset "build_max_heap" begin
        input = [16 4 10 14 7 9 3 2 8 1]
        output = [16 14 10 8 7 9 3 2 4 1]
        heap = build_max_heap(input)
        @test heap.array == output

        input = [4 1 3 2 16 9 10 14 8 7]
        output = [16 14 10 8 7 9 3 2 4 1]
        heap = build_max_heap!(input)
        @test heap.array == output
    end
end