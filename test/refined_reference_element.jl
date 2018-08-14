using Rewrite: refined_element, reference_element, nodes_on_ref_faces, nodes_on_ref_edges, Tets64
using Test

@testset "Refined tet" begin
    N = 8
    tets = refined_element(N, Tets64)

    @test length(tets.levels[1].nodes) == 4
    @test length(tets.levels[2].nodes) == 10

    nodes_per_face_base = nodes_on_ref_faces(tets.levels[1])

    # Test whether the standard numbering of the faces is correct
    @test nodes_per_face_base[1] == [1,2,3]
    @test nodes_per_face_base[2] == [1,2,4]
    @test nodes_per_face_base[3] == [1,3,4]
    @test nodes_per_face_base[4] == [2,3,4]

    # Test whether we get the correct nodes of refined tets as well.
    for i = 1 : N, (face_idx, nodes) in enumerate(nodes_on_ref_faces(tets.levels[i]))
        @test length(nodes) == sum(1 : 2^(i-1) + 1)
    end

    # Test wether the standard numbering of the edges is correct
    nodes_per_edge_base = nodes_on_ref_edges(tets.levels[1])
    @test nodes_per_edge_base[1] == [1,2]
    @test nodes_per_edge_base[2] == [1,3]
    @test nodes_per_edge_base[3] == [1,4]
    @test nodes_per_edge_base[4] == [2,3]
    @test nodes_per_edge_base[5] == [2,4]
    @test nodes_per_edge_base[6] == [3,4]

    for i = 1 : N, (face_idx, nodes) in enumerate(nodes_on_ref_edges(tets.levels[i]))
        @test length(nodes) == 2^(i-1) + 1
    end
end
