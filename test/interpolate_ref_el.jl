using Rewrite: refined_element, Tris, Mesh, Tris64, Tets64, affine_map, 
               refine_uniformly, edge_graph, sort_element_nodes!, nodes_on_ref_faces,
               nodes_on_ref_edges, navigation
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

function face_coloring()
    # Make a base mesh
    base_nodes = SVector{3,Float64}[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2/3,2/3,2/3),(-1,0,0),(0,-1,0),(0,0,-1)]
    base_elements = [(1,2,3,4),(1,2,3,8),(1,2,4,7),(1,3,4,6),(2,3,4,5)]
    base = Mesh(base_nodes, base_elements)

    # Make some perturbation to the nodes to make things interesting
    for i = 1 : length(base.nodes)
        base.nodes[i] += SVector(randn()/20, randn()/20, randn()/20)
    end

    # Sort element stuff.
    sort_element_nodes!(base.elements)

    # Get mappings from node, edge, face to element and local ids.
    node_to_el, edge_to_el, face_to_el = navigation(base)

    # Refine a reference tri
    ref = refined_element(4, Tets64)

    # Get local numbering of the faces / edges / nodes
    nodes_per_face = nodes_on_ref_faces(ref.levels[end])
    nodes_on_edges = nodes_on_ref_edges(ref.levels[end])

    for face in face_to_el
        println(face) 
    end
    return
    total_nodes_per_face = length(nodes_per_face[1])

    total_fine_nodes = length(ref.levels[end].nodes)

    fine_nodes = SVector{3,Float64}[]
    fine_elements = NTuple{4,Int}[]

    # This will hold the interpolated values with local numbering per base element.
    u_fine = zeros(total_fine_nodes, length(base.elements))

    # List matching faces.
    face_1 = rand(total_nodes_per_face)
    u_fine[nodes_per_face[1], 1] .= face_1
    u_fine[nodes_per_face[1], 2] .= face_1

    face_2 = rand(total_nodes_per_face)
    u_fine[nodes_per_face[2], 1] .= face_2
    u_fine[nodes_per_face[1], 3] .= face_2

    face_3 = rand(total_nodes_per_face)
    u_fine[nodes_per_face[3], 1] .= face_3
    u_fine[nodes_per_face[1], 4] .= face_3

    face_4 = rand(total_nodes_per_face)
    u_fine[nodes_per_face[4], 1] .= face_4
    u_fine[nodes_per_face[1], 5] .= face_4

    u_fine[nodes_on_edges, :] = 0.0

    let element = base.elements[1]
        # coord transform J * x + b
        J, b = affine_map(base, element)

        # Push all the fine nodes & elements
        new_nodes = [J * x + b for x in ref.levels[end].nodes]
        new_elements = [el for el in ref.levels[end].elements]
        
        append!(fine_elements, new_elements)        
        append!(fine_nodes, new_nodes)
    end

    let element = base.elements[2]
        # coord transform J * x + b
        J, b = affine_map(base, element)

        normal = @SVector [0.0, 0.0, -1.0]

        # Push all the fine nodes & elements
        new_nodes = [J * x + b + 0.3 * normal  for x in ref.levels[end].nodes]
        new_elements = [el .+ total_fine_nodes * 1 for el in ref.levels[end].elements]
        
        append!(fine_elements, new_elements)        
        append!(fine_nodes, new_nodes)
    end

    let element = base.elements[3]
        # coord transform J * x + b
        J, b = affine_map(base, element)

        normal = @SVector [0.0, -1.0, 0.0]

        # Push all the fine nodes & elements
        new_nodes = [J * x + b + 0.3 * normal for x in ref.levels[end].nodes]
        new_elements = [el .+ total_fine_nodes * 2 for el in ref.levels[end].elements]
        
        append!(fine_elements, new_elements)        
        append!(fine_nodes, new_nodes)
    end

    let element = base.elements[4]
        # coord transform J * x + b
        J, b = affine_map(base, element)

        normal = @SVector [-1.0, 0.0, 0.0]

        # Push all the fine nodes & elements
        new_nodes = [J * x + b + 0.3 * normal for x in ref.levels[end].nodes]
        new_elements = [el .+ total_fine_nodes * 3 for el in ref.levels[end].elements]
        
        append!(fine_elements, new_elements)        
        append!(fine_nodes, new_nodes)
    end

    let element = base.elements[5]
        # coord transform J * x + b
        J, b = affine_map(base, element)

        normal = @SVector [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]

        # Push all the fine nodes & elements
        new_nodes = [J * x + b + 0.3 * normal for x in ref.levels[end].nodes]
        new_elements = [el .+ total_fine_nodes * 4 for el in ref.levels[end].elements]
        
        append!(fine_elements, new_elements)        
        append!(fine_nodes, new_nodes)
    end

    # Construct the full mesh now.
    fine = Mesh(fine_nodes, fine_elements)

    vtk_grid("interpolation_stuff_tet", fine) do vtk
        vtk_point_data(vtk, u_fine[:], "u")
    end
end