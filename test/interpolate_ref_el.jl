using Rewrite: refined_element, Tris, Mesh, Tris64, affine_map
using StaticArrays

"""
We refine a reference triangle a bunch of times, construct a coarse base mesh,
generate some random numbers for the nodes on the base mesh, and then interpolate
these to the fine grid. Finally we construct the full fine grid with nodes on the
interfaces repeated, to see if the interfaces match.
"""
function interpolate_ref_elements()
    # Make a base mesh
    base_nodes = SVector{2,Float64}[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (2.0, 1.0)]
    base_elements = [(1,2,4),(2,3,4),(2,3,5)]
    base = Mesh(base_nodes, base_elements)

    # Refine a reference tri
    levels, interops = refined_element(3, Tris64)

    total_fine_nodes = length(levels[end].nodes)

    fine_nodes = SVector{2,Float64}[]
    fine_elements = NTuple{3,Int}[]

    # Generate some random values that will be interpolated to the fine grid.
    u_coarse = [1.0, 2.0, 3.0, 4.0, 5.0]

    # This will hold the interpolated values with local numbering per base element.
    u_fine = zeros(total_fine_nodes, length(base.elements))

    for (i, element) in enumerate(base.elements)
        # coord transform J * x + b
        J, b = affine_map(base, element)

        # Get the u values and interpolate them.
        us = [u_coarse[i] for i in element]
        for P = interops
            us = P * us
        end

        u_fine[:, i] .= us

        # Push all the fine nodes & elements
        new_nodes = [J * x + b for x in levels[end].nodes]
        new_elements = [el .+ total_fine_nodes * (i-1) for el in levels[end].elements]
        
        append!(fine_elements, new_elements)        
        append!(fine_nodes, new_nodes)
    end

    # Construct the full mesh now.
    fine = Mesh(fine_nodes, fine_elements)

    vtk_grid("interpolation_stuff", fine) do vtk
        vtk_point_data(vtk, u_fine[:], "u")
    end
end