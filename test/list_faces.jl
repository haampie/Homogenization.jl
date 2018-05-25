using Rewrite: list_boundary_faces, Mesh, refine_uniformly
using Base.Test
using StaticArrays

@testset "Listing boundary faces" begin
    nodes = SVector{3,Float64}[(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
    elements = [(1,2,3,4)]

    # A tetrahedron should have 4 faces.
    mesh = Mesh(nodes, elements)
    faces = list_boundary_faces(mesh)
    @test length(faces) == 4

    # Two refinements shoud split each face in 16 faces.
    mesh_finer = refine_uniformly(Mesh(nodes, elements), times = 2)
    faces_finer = list_boundary_faces(mesh_finer)
    @test length(faces_finer) == 4 * 16
end