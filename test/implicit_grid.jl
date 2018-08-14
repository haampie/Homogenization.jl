using Test
using Rewrite: sort_element_nodes!, refine_uniformly, Mesh,
               ImplicitFineGrid, cell_type, default_quad, update_J,
               ElementValues, refined_mesh, nodes_per_face_interior, reinit!,
               get_x, valrange
using StaticArrays

@testset "Test whether implicit nodes on the interfaces match" begin
    # Refinements
    refs = 5

    # Cube
    nodes = SVector{3,Float64}[(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]
    elements = [(1,2,3,5),(2,3,4,8),(3,5,7,8),(2,5,6,8),(2,3,5,8)]
    coarse_mesh = refine_uniformly(Mesh(nodes, elements), times = 3)
    sort_element_nodes!(coarse_mesh.elements)

    implicit = ImplicitFineGrid(coarse_mesh, refs)

    # Set up element values for the coarse mesh
    cell = cell_type(coarse_mesh)
    element_values = ElementValues(cell, default_quad(cell), update_J)

    n2e = implicit.interfaces.nodes
    e2e = implicit.interfaces.edges
    f2e = implicit.interfaces.faces

    for level = 1 : refs
        local_numbering = implicit.reference.numbering[level]
        ref_mesh = refined_mesh(implicit, level)

        # Loop over all nodes on some interface
        @inbounds for i = 1 : length(n2e.cells)

            # Loop over all the elements belonging to this face
            xs_per_element = map(valrange(n2e, i)) do j
                value = n2e.values[j]
                node = local_numbering.nodes[value.local_id]

                # Update the affine map
                reinit!(element_values, coarse_mesh, coarse_mesh.elements[value.element])

                # Find the actual x coord of each node.
                get_x(element_values, ref_mesh.nodes[node])
            end

            # Test if the others are equal to the first
            @test all(x -> x ≈ first(xs_per_element), xs_per_element)
        end

        # Loop over all edges on some interface
        @inbounds for i = 1 : length(e2e.cells)

            # Loop over all the elements belonging to this face
            xs_per_element = map(valrange(e2e, i)) do j
                value = e2e.values[j]
                nodes = local_numbering.edges_interior[value.local_id]

                # Update the affine map
                reinit!(element_values, coarse_mesh, coarse_mesh.elements[value.element])

                # Find the actual x coord of each node.
                map(k -> get_x(element_values, ref_mesh.nodes[k]), nodes)
            end

            all(isempty, xs_per_element) && continue

            # Test if the others are equal to the first
            @test all(x -> x ≈ first(xs_per_element), xs_per_element)
        end

        # Loop over all faces on some interface
        @inbounds for i = 1 : length(f2e.cells)

            # Loop over all the elements belonging to this face
            xs_per_element = map(valrange(f2e, i)) do j
                value = f2e.values[j]
                nodes = local_numbering.faces_interior[value.local_id]

                # Update the affine map
                reinit!(element_values, coarse_mesh, coarse_mesh.elements[value.element])

                # Find the actual x coord of each node.
                map(k -> get_x(element_values, ref_mesh.nodes[k]), nodes)
            end

            all(isempty, xs_per_element) && continue

            # Test if the others are equal to the first
            @test all(x -> x ≈ first(xs_per_element), xs_per_element)
        end
    end
end
