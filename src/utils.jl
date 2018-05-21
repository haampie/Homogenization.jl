using Base.OneTo

"""
    generate_random_elements(n::Int, m::Ti, k::Int) -> Vector{NTuple{k,Ti}}

Return a bogus mesh of `n` elements with `k` points with global node numbered from 1 to m.
"""
generate_random_elements(n::Int, m::Ti, k::Int) where {Ti} = 
    [Tuple(rand(OneTo(m)) for j = 1 : k) for i = 1 : n]

cell_to_vtk(m::Mesh{2,3}) = VTKCellTypes.VTK_TRIANGLE
cell_to_vtk(m::Mesh{3,4}) = VTKCellTypes.VTK_TETRA

function WriteVTK.vtk_grid(filename::AbstractString, mesh::Mesh{dim,N,Tv}) where {dim,N,Tv}
    celltype = cell_to_vtk(mesh)
    cells = [MeshCell(celltype, SVector(element)) for element in mesh.elements]
    coords = reinterpret(Tv, mesh.nodes, (dim, length(mesh.nodes)))
    return vtk_grid(filename, coords, cells)
end