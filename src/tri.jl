function my_tri_grid()
    nodes = SVector{2,Float64}[
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (2.0, 1.0)
    ]

    cells = [(1, 2, 4),(2, 3, 4),(2, 3, 5)]

    Mesh(nodes,cells)
end

function my_tet_grid()
    nodes = SVector{3,Float64}[
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0)
    ]

    cells = [(1,2,3,4)]

    Mesh(nodes,cells)
end