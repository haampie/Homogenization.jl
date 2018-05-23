using Rewrite: refined_element, Tris, Mesh, Tris64, Tets64, affine_map, 
               refine_uniformly, edge_graph, sort_element_nodes!, nodes_on_ref_faces,
               nodes_on_ref_edges, navigation, face_to_elements, edge_to_elements,
               nelements, nnodes, nodes_per_face, nodes_per_edge, get_reference_normals,
               Tet, node_to_elements
using StaticArrays
using WriteVTK

"""
We refine a reference triangle a bunch of times, construct a coarse base mesh,
generate some random numbers for the nodes on the base mesh, and then interpolate
these to the fine grid. Finally we construct the full fine grid with nodes on the
interfaces repeated, to see if the interfaces match.
"""
function interpolate_ref_elements()
    # Make a base mesh
    base_nodes = SVector{2,Float64}[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (2.0, 1.0)]
    base_elements = [(1,2,4),(2,3,4),(3,4,5)]
    base = Mesh(base_nodes, base_elements)
    for i = 1 : 0
        base = refine_uniformly(base, edge_graph(base))
    end
    
    for i = 1 : length(base.nodes)
        base.nodes[i] += SVector(randn()/100, randn()/100)
    end

    @assert all(issorted, base.elements)

    # Refine a reference tri
    ref = refined_element(5, Tris64)

    total_fine_nodes = length(ref.levels[end].nodes)

    fine_nodes = SVector{2,Float64}[]
    fine_elements = NTuple{3,Int}[]

    # Generate some random values that will be interpolated to the fine grid.
    u_coarse = linspace(0, 1, length(base.nodes))

    # This will hold the interpolated values with local numbering per base element.
    u_fine = zeros(total_fine_nodes, length(base.elements))

    for (i, element) in enumerate(base.elements)
        # coord transform J * x + b
        J, b = affine_map(base, element)

        # Get the u values and interpolate them.
        us = [u_coarse[i] for i in element]
        for P = ref.interops
            us = P * us
        end

        u_fine[:, i] .= us

        # Push all the fine nodes & elements
        new_nodes = [J * x + b for x in ref.levels[end].nodes]
        new_elements = [el .+ total_fine_nodes * (i-1) for el in ref.levels[end].elements]
        
        append!(fine_elements, new_elements)        
        append!(fine_nodes, new_nodes)
    end

    # Construct the full mesh now.
    fine = Mesh(fine_nodes, fine_elements)

    vtk_grid("interpolation_stuff", fine) do vtk
        vtk_point_data(vtk, u_fine[:], "u")
    end
end

"""
We refine a reference tetrahedron a bunch of times, construct a coarse base mesh,
generate some random numbers for the nodes on the base mesh, and then interpolate
these to the fine grid. Finally we construct the full fine grid with nodes on the
interfaces repeated, to see if the interfaces match.
"""
function interpolate_ref_elements_tet()
    # Make a base mesh
    base_nodes = SVector{3,Float64}[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    base_elements = [(1,2,3,4)]
    base = Mesh(base_nodes, base_elements)
    for i = 1 : 2
        base = refine_uniformly(base, edge_graph(base))
    end
    
    for i = 1 : length(base.nodes)
        base.nodes[i] += SVector(randn()/100, randn()/100, randn()/100)
    end

    sort_element_nodes!(base.elements)

    @assert all(issorted, base.elements)

    # Refine a reference tri
    ref = refined_element(5, Tets64)

    total_fine_nodes = length(ref.levels[end].nodes)

    fine_nodes = SVector{3,Float64}[]
    fine_elements = NTuple{4,Int}[]

    # Generate some random values that will be interpolated to the fine grid.
    u_coarse = linspace(0, 1, length(base.nodes))

    # This will hold the interpolated values with local numbering per base element.
    u_fine = zeros(total_fine_nodes, length(base.elements))

    for (i, element) in enumerate(base.elements)
        # coord transform J * x + b
        J, b = affine_map(base, element)

        # Get the u values and interpolate them.
        us = [u_coarse[i] for i in element]
        for P = ref.interops
            us = P * us
        end

        u_fine[:, i] .= us

        # Push all the fine nodes & elements
        new_nodes = [J * x + b for x in ref.levels[end].nodes]
        new_elements = [el .+ total_fine_nodes * (i-1) for el in ref.levels[end].elements]
        
        append!(fine_elements, new_elements)        
        append!(fine_nodes, new_nodes)
    end

    # Construct the full mesh now.
    fine = Mesh(fine_nodes, fine_elements)

    vtk_grid("interpolation_stuff_tet", fine) do vtk
        vtk_point_data(vtk, u_fine[:], "u")
    end
end

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
        
        fine_nodes = Vector{SVector{3,Float64}}(nelements(base) * nnodes(finest_mesh))
        fine_elements = Vector{NTuple{4,Int}}(nelements(base) * nelements(finest_mesh))

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
        node_to_element = node_to_elements(base)

        u_faces = zeros(nnodes(finest_mesh), nelements(base))
        u_edges = zeros(nnodes(finest_mesh), nelements(base))
        u_nodes = zeros(nnodes(finest_mesh), nelements(base))

        numbering = ref.numbering[end]

        # Loop over all the faces
        @inbounds for (i, face) in enumerate(face_to_element.cells)

            # Generate some random numbers
            u1 = randn(nodes_per_face(ref, refinements))

            # Loop over the elements
            for j = face_to_element.offset[i] : face_to_element.offset[i+1]-1
                element_id = face_to_element.values[j]
                nodes = numbering.faces[element_id.local_id]
                u_faces[nodes, element_id.element] .= u1
            end
        end

        # Loop over all the edges
        @inbounds for (i, edge) in enumerate(edge_to_element.cells)

            # Generate some random numbers
            u2 = randn(nodes_per_edge(ref, refinements))

            # Loop over the elements
            for j = edge_to_element.offset[i] : edge_to_element.offset[i+1]-1
                element_id = edge_to_element.values[j]
                nodes = numbering.edges[element_id.local_id]
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