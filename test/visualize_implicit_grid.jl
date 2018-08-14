using Rewrite: refined_element, Tris, Mesh, Tris64, Tets64, affine_map,
               refine_uniformly, edge_graph, sort_element_nodes!, nodes_on_ref_faces,
               nodes_on_ref_edges, interfaces, face_to_elements, edge_to_elements,
               nelements, nnodes, nodes_per_face_interior, nodes_per_edge_interior,
               get_reference_normals, Tet, node_to_elements, ImplicitFineGrid,
               construct_full_grid, ZeroDirichletConstraint, list_boundary_nodes_edges_faces,
               refined_mesh, apply_constraint!
using StaticArrays
using WriteVTK

function see_whether_faces_edges_nodes_connect_at_interfaces(refinements = 3, gap = 1.0)
    # Make a base mesh
    @time begin
        base_nodes = SVector{3,Float64}[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2/3,2/3,2/3),(-1,0,0),(0,-1,0),(0,0,-1)]
        base_elements = [(1,2,3,4),(1,2,3,8),(1,2,4,7),(1,3,4,6),(2,3,4,5)]
        base = Mesh(base_nodes, base_elements)

        # Make some perturbation to the nodes to make things interesting
        for i = 1 : length(base.nodes)
            base.nodes[i] += SVector(randn()/50, randn()/50, randn()/50)
        end

        # Sort element stuff.
        sort_element_nodes!(base.elements)

        # Refine a reference tri
        ref = refined_element(refinements, Tets64)

        finest_mesh = ref.levels[end]

        fine_nodes = Vector{SVector{3,Float64}}(undef, nelements(base) * nnodes(finest_mesh))
        fine_elements = Vector{NTuple{4,Int}}(undef, nelements(base) * nelements(finest_mesh))

        new_nodes_idx = 0
        new_elements_idx = 0

        @inbounds let element = base.elements[1]
            # coord transform J * x + b
            J, b = affine_map(base, element)

            # Push all the fine nodes & elements
            for x in finest_mesh.nodes
                fine_nodes[new_nodes_idx += 1] = J * x + b
            end

            for el in finest_mesh.elements
                fine_elements[new_elements_idx += 1] = el
            end
        end

        @inbounds for (i, normal) in enumerate(get_reference_normals(Tet{Float64}))
            element = base.elements[i + 1]
            # coord transform J * x + b
            J, b = affine_map(base, element)

            # Push all the fine nodes & elements
            for x in finest_mesh.nodes
                fine_nodes[new_nodes_idx += 1] = J * x + b + gap * normal
            end
            for el in finest_mesh.elements
                fine_elements[new_elements_idx += 1] = el .+ nnodes(finest_mesh) * i
            end
        end

        # Construct the full mesh now.
        fine = Mesh(fine_nodes, fine_elements)
        println("Built the mesh with *all* fine nodes: ", nnodes(fine), " nodes and ", nelements(fine), " elements.")
    end

    @time begin
        # Get mappings from node, edge, face of base mesh to element and local ids.
        face_to_element = face_to_elements(base)
        edge_to_element = edge_to_elements(base)
        _, node_to_element = node_to_elements(base)

        u_faces = zeros(nnodes(finest_mesh), nelements(base))
        u_edges = zeros(nnodes(finest_mesh), nelements(base))
        u_nodes = zeros(nnodes(finest_mesh), nelements(base))

        numbering = ref.numbering[end]

        # Loop over all the faces
        @inbounds for (i, face) in enumerate(face_to_element.cells)

            # Generate some random numbers
            u1 = randn(nodes_per_face_interior(ref, refinements))

            # Loop over the elements
            for j = face_to_element.offset[i] : face_to_element.offset[i+1]-1
                element_id = face_to_element.values[j]
                nodes = numbering.faces_interior[element_id.local_id]
                u_faces[nodes, element_id.element] .= u1
            end
        end

        # Loop over all the edges
        @inbounds for (i, edge) in enumerate(edge_to_element.cells)

            # Generate some random numbers
            u2 = randn(nodes_per_edge_interior(ref, refinements))

            # Loop over the elements
            for j = edge_to_element.offset[i] : edge_to_element.offset[i+1]-1
                element_id = edge_to_element.values[j]
                nodes = numbering.edges_interior[element_id.local_id]
                u_edges[nodes, element_id.element] .= u2
            end
        end

        # Loop over the nodes
        @inbounds for (i, node) in enumerate(node_to_element.cells)

            # Generate a random number for this node.
            u3 = randn()

            # Loop over the element
            for j = node_to_element.offset[i] : node_to_element.offset[i+1]-1
                element_id = node_to_element.values[j]
                node = numbering.nodes[element_id.local_id]
                u_nodes[node, element_id.element] .= u3
            end
        end

        println("Colored faces, edges and nodes")
    end

    @time begin
        println("Now saving.")
        vtk_grid("interpolation_stuff_tet", fine) do vtk
            let u_faces = u_faces, u_edges = u_edges, u_nodes = u_nodes
                vtk_point_data(vtk, u_faces, "faces")
                vtk_point_data(vtk, u_edges, "edges")
                vtk_point_data(vtk, u_nodes, "nodes")
            end
        end
    end
end

function extract_full_fine_grid(total_levels = 6, store_level = 3)
    # Unit cube
    nodes = SVector{3,Float64}[
        (0,0,0),
        (1,0,0),
        (0,1,0),
        (1,1,0),
        (0,0,1),
        (1,0,1),
        (0,1,1),
        (1,1,1)
    ]

    # Split in tetrahedra
    elements = [
        (1,2,3,5),
        (2,3,4,8),
        (3,5,7,8),
        (2,5,6,8),
        (2,3,5,8)
    ]

    base = Mesh(nodes, elements)

    # We apply a zero Dirichlet b.c. to the boundary
    nodes, edges, faces = list_boundary_nodes_edges_faces(base)

    constraint = ZeroDirichletConstraint(nodes, edges, faces)

    # Implicitly refine things.
    implicit = ImplicitFineGrid(base, total_levels)

    # Shows some info about number of nodes
    @show implicit

    # Find the number of nodes per element at level `store_level`
    nodes_per_element = nnodes(refined_mesh(implicit, store_level))

    # Fill each element with a unique value
    u = Matrix{Float64}(undef, nodes_per_element, nelements(base))

    for el = 1 : nelements(base)
        u[:, el] .= float(el)
    end

    # Apply zero Dirichlet boundary conditions.
    apply_constraint!(u, store_level, constraint, implicit)

    # Construct the full grid (expensive if `store_level` is large)
    fine_mesh = construct_full_grid(implicit, store_level)

    # Save the full grid
    vtk = vtk_grid("single_level", fine_mesh) do vtk
        vtk_point_data(vtk, u, "u")
    end
end
