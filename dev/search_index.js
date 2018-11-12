var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "page",
    "text": ""
},

{
    "location": "#Homogenization.jl-1",
    "page": "Tutorial",
    "title": "Homogenization.jl",
    "category": "section",
    "text": "Consider the problem -nabla cdot a(x) nabla u + lambda u = f in 2D and 3D where a is a symmetric matrix with piecewise constant coefficients in the domain U subset mathbbR^d with d = 2 3."
},

{
    "location": "#Homogenization.Mesh",
    "page": "Tutorial",
    "title": "Homogenization.Mesh",
    "category": "type",
    "text": "Mesh(nodes, elements) -> Mesh{dim,N,Tv,Ti}\n\nStores the FEM mesh where the spatial dimension is dim, the number of nodes per element is N, the number type is Tv and the integer type Ti.\n\nExample\n\nusing StaticArrays\nnodes = SVector{3,Float64}[(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]\nelements = [(1,2,3,5),(2,3,4,8),(3,5,7,8),(2,5,6,8),(2,3,5,8)]\ncube = Mesh(nodes, elements)\n\n\n\n\n\n"
},

{
    "location": "#Homogenization.hypercube",
    "page": "Tutorial",
    "title": "Homogenization.hypercube",
    "category": "function",
    "text": "hypercube(Tet{Float64}, n, Ti = Int; scale = 1, origin = (1,1,1), sorted = true) -> Mesh\n\nCreate a mesh of n by n by n cubes each split into 5 tetrahedra.\n\n\n\n\n\nhypercube(Tri{Float64}, n, Ti = Int; scale = 1, origin = (1,1,1), sorted = true) -> Mesh\n\nCreate a mesh of n by n squares each split into two triangles.\n\n\n\n\n\n"
},

{
    "location": "#Base-mesh-1",
    "page": "Tutorial",
    "title": "Base mesh",
    "category": "section",
    "text": "We will assume it is feasible to make a triangulation of U such that a is constant on each triangle or tetrahedron. For this we need a meshMeshThere are some helper functions as well to generate a simple meshhypercubeNow we generate random conductivity parameters for each element of the base mesh.using Homogenization\nusing Homogenization: conductivity_per_element, generate_conductivity\n\nn = 32\nbase = hypercube(Tri{Float64}, n)\na = conductivity_per_element(base, generate_conductivity(base, n))Next we could try to visualize the base mesh by exporting it to a file compatible with Paraview. We store the value of a as vectorial data for each element of the base mesh:using WriteVTK\nusing Homogenization: dimension\nvtk_grid(\"checkerboard\", base) do vtk\n    as_matrix = reshape(reinterpret(Float64, a), dimension(base), :)\n    vtk_cell_data(vtk, as_matrix, \"a\")\nendwhich saves a file checkerboard.vtu in the current working directory. It should look like(Image: Paraview screenshot)"
},

{
    "location": "#Implicit-refinement-1",
    "page": "Tutorial",
    "title": "Implicit refinement",
    "category": "section",
    "text": "In FEM we wish to do h-refinement of this triangulation, but the fully refined grid is assumed to be too large to store explicitly."
},

]}
