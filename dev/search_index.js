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
    "text": "Consider the problem -nabla cdot a(x) nabla u + lambda u = f in 2D and 3D where a is a symmetric, positive definite matrix with piecewise constant coefficients in the domain U subset mathbbR^d with d = 2 3 lambda  0 and u = 0 on partial U."
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
    "location": "#Solving-a-\"coarse\"-FEM-problem-1",
    "page": "Tutorial",
    "title": "Solving a \"coarse\" FEM problem",
    "category": "section",
    "text": "For the moment let\'s stick to a very coarse grid for the checkerboard as we have it with just four nodes on the corners of each checkerboard cell (one node per cell). The weak form reads int_U a nabla u cdot nabla v + lambda uv = int fvfor u v in the appropriate space, which we discretize by taking piece-wise linear elements for u and v Let\'s take f = lambda = 1This is done as follows:using Homogenization: assemble_checkerboard, assemble_vector\nA = assemble_checkerboard(base, a, 1.0)\nb = assemble_vector(base, identity)The above does not take into account that u is supposed to be zero at the boundary. To impose the boundary condition we have to detect the boundary in the mesh first.  This is rather simple:using Homogenization: list_interior_nodes\ninterior = list_interior_nodes(base)The boundary condition is simply imposed by partitioning the unknowns x = beginbmatrixx_i  x_bendbmatrix^T such that the linear system reads:beginbmatrixA_ii  A_ib  A_bi  A_bbendbmatrixbeginbmatrixx_i  x_bendbmatrix = beginbmatrixb_i  b_bendbmatrixWith the boundary nodes x_b = 0 this comes down to solving A_iix_i = b_iusing Homogenization: nnodes\nA_int = A[interior, interior]\nb_int = b[interior]\nx_int = A_int \\ b_int\nx = zeros(nnodes(base))\nx[interior] .= x_intFinally we would like to visualize the FEM solution in Paraview, which is done as follows:vtk_grid(\"checkerboard\", base) do vtk\n    as_matrix = reshape(reinterpret(Float64, a), dimension(base), :)\n    vtk_cell_data(vtk, as_matrix, \"a\")\n    vtk_point_data(vtk, x, \"x\")\nendAnd it will look more or less like(Image: Paraview solution)"
},

{
    "location": "#Implicit-refinement-1",
    "page": "Tutorial",
    "title": "Implicit refinement",
    "category": "section",
    "text": "The coarse FEM problem will have a large error as the coefficients are rough and the number of nodes per checkerboard cell is approximately one. Therefore we wish to do h-refinement. The assumption will be that we will not be able to store the fully refined grid (nodes and elements) and neither the corresponding discrete operator (A_int).It makes sense to try multigrid to get h-refinement"
},

]}
