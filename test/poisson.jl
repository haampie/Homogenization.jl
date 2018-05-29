using Rewrite: assemble_matrix, assemble_vector, refine_uniformly, 
               Mesh, list_interior_nodes, nnodes, nelements, sort_element_nodes!
using StaticArrays
using WriteVTK

function poisson()
    nodes = SVector{3,Float64}[(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]

    # Perturb things a bit.
    map!(x -> x .+ randn(3) / 50, nodes, nodes)
    
    # Split in tetrahedra
    elements = [(1,2,3,5),(2,3,4,8),(3,5,7,8),(2,5,6,8),(2,3,5,8)]

    mesh = refine_uniformly(Mesh(nodes,elements), times = 6)

    @show nnodes(mesh) nelements(mesh)

    sort_element_nodes!(mesh.elements)

    interior = list_interior_nodes(mesh)

    println("Assembling")

    A = assemble_matrix(mesh, dot)
    b = assemble_vector(mesh, identity)
    x = zeros(nnodes(mesh))

    println("Solving")
    x[interior] .= A[interior,interior] \ b[interior]

    println("Saving")
    vtk_grid("poisson", mesh) do vtk
        vtk_point_data(vtk, x, "x")
        # vtk_point_data(vtk, b, "b")
    end
end