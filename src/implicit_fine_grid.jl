import Base: @propagate_inbounds, show
"""
ImplicitFineGrid holds the base mesh and the refinements of the reference element
"""
struct ImplicitFineGrid{dim,N,Tv,Ti}
    levels::Int
    reference::MultilevelReference{dim,N,Tv,Ti}
    base::Mesh{dim,N,Tv,Ti}
end

function ImplicitFineGrid(base::Mesh{dim,N,Tv,Ti}, levels::Int) where {dim,N,Tv,Ti}
    reference = refined_element(levels, typeof(base))
    ImplicitFineGrid(levels, reference, base)
end

nlevels(g::ImplicitFineGrid) = g.levels
base_mesh(g::ImplicitFineGrid) = g.base
@propagate_inbounds refined_mesh(g::ImplicitFineGrid, level::Int) = g.reference.levels[level]
@propagate_inbounds local_numbering(g::ImplicitFineGrid, level::Int) = g.reference.numbering[level]

function show(io::IO, g::ImplicitFineGrid)
    base = base_mesh(g)
    finest = refined_mesh(g, nlevels(g))
    print(io, "Implicit grid of cell type ", cell_type(base),
              ". Base mesh has ", nnodes(base), " nodes and ", nelements(base), " elements.",
              " Finest level (", nlevels(g), ") has ", nnodes(finest), " nodes and ", nelements(finest), " elements.",
              " In total at most ", nnodes(finest) * nelements(base), " unknowns.")
end

"""
    construct_full_grid(g::ImplicitFineGrid, level::Int) -> Mesh

Builds the full mesh at a certain level with nodes on the interface repeated.
Be very scared, cause the number of nodes gets large!
"""
function construct_full_grid(g::ImplicitFineGrid{dim,N,Tv,Ti}, level::Int) where {dim,N,Tv,Ti}
    base = base_mesh(g)
    ref_mesh = refined_mesh(g, level)

    # Since we copy nodes on the interface, we have #coarse * #ref nodes & elements
    total_nodes = nelements(base) * nnodes(ref_mesh)
    total_elements = nelements(base) * nelements(ref_mesh)

    nodes = Vector{SVector{dim,Tv}}(total_nodes)
    elements = Vector{NTuple{N,Ti}}(total_elements)

    # Now for each base element we simply apply the coordinate transform to each
    # node, and we copy over each fine element. We only have to renumber the
    # fine elements by the offset of the base element number.

    node_idx = 0
    element_idx = 0
    offset = 0

    @inbounds for element in base.elements
        # Get the coordinate mapping
        J, b = affine_map(base, element)
        
        # Copy the transformed nodes over
        for node in ref_mesh.nodes
            nodes[node_idx += 1] = J * node + b
        end

        # Copy over the elements
        for element in ref_mesh.elements
            elements[element_idx += 1] = element .+ offset
        end

        offset += nnodes(ref_mesh)
    end

    return Mesh(nodes, elements)
end

struct ZeroDirichletConstraint{Ti}
    list_of_faces::Vector{ElementId{Ti}}
end

"""
    apply_dirichlet_constraint!(u, level, ::ZeroDirichletConstraint, ::ImplicitFineGrid)

Apply zero Dirichlet conditions to the nodes on the boundary of implicitly refined
vector `u`. ZeroDirichletConstraint contains the faces of the base mesh, and via
the ImplicitFineGrid we get the local numbering of the faces to zero them out.
"""
function apply_dirichlet_constraint!(u::Matrix{Tv}, level::Int, z::ZeroDirichletConstraint{Ti}, g::ImplicitFineGrid{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
    
    # Maybe get rid of these assertions one day.
    @assert size(u, 1) == nnodes(refined_mesh(g, level))
    @assert size(u, 2) == nelements(base_mesh(g))

    numbering = local_numbering(g, level)

    for faceInfo in z.list_of_faces
        # Extract the base element id
        element = faceInfo.element

        # And the local face number.
        face = faceInfo.local_id

        # Zero out each node on the boundary.
        @inbounds for node = numbering.faces[face]
            u[node, element] = zero(Tv)
        end
    end

    u
end