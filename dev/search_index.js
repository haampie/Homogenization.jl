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
    "text": "The coarse FEM problem will have a large error as the coefficients are rough and the number of nodes per checkerboard cell is approximately one. Therefore we wish to do h-refinement. The assumption will be that we will not be able to store the fully refined grid (nodes and elements) and neither the corresponding discrete operator (A).The usual solution is then to use an iterative method to solve A * x = b and compute the entries of the discrete operator A  on demand (matrix-free). However, in our case the coefficients of a are supposed to be constant in each refined element, so we can do better. We refine a single reference triangle or tetrahedron, build local discrete operators and then map the refined reference element to each base mesh element."
},

{
    "location": "#Reusing-discrete-operators-of-a-single-reference-element-1",
    "page": "Tutorial",
    "title": "Reusing discrete operators of a single reference element",
    "category": "section",
    "text": "(Image: Implicit grid)Suppose tau subset U is a single element of the base mesh. We refine tau such that it has N_f nodes. A local discretized operator takes the formA_ij^tau = int_tau a nabla phi_i cdot nabla phi_j  dxwhere phi_i and phi_j are local basis functions (i j = 1 dots N_f) and a is a constant matrix by assumption. If we have some reference triangle or tetrahederon hattau the pullback to reference coordinates looks likeA_ij^tau = int_hattau a J^-1nabla hatphi_i cdot J^-1nabla hatphi_j J  dhatxwhere hatphi_i and hatphi_j are the corresponding reference basis functions and J is some matrix rotating and scaling the triangle or tetrahedron.If we assume each base mesh element is refined the same way, then the only dependence on tau is in J and the matrix a. We can save some computational work by computing d^2 matriceshatA_ij^(kell) = int_hattaufracpartial phi_ipartial hatx_kfracpartial phi_jpartial hatx_ell  dhatx text for  i j = 1 dots N_ffor k ell = 1 dots d This way the local operator A^tau takes the formA^tau = sum_k ell = 1^d (J^-TaJ^-1J)_kellhatA^(kell)So practically, for any base element tau if we wish to work with the matrix A^tau, we compute the mapping J and the tiny matrix J^-TaJ^-1J and then reuse the matrices hatA^(kell) which we have computed up front once and for all."
},

{
    "location": "#Implicit-grid-and-multigrid-(high-level-idea)-1",
    "page": "Tutorial",
    "title": "Implicit grid & multigrid (high-level idea)",
    "category": "section",
    "text": "Let\'s refine the base grid r times, so that we have r + 1 levels; level 1 has no refinements at all. We need a few things:Suppose the base mesh has N_e elements, and for a given level the refined reference element has N_f nodes. For convenient numbering we store nodes on the boundary of a base mesh element multiple times, so a good way to represent the right-hand side b and uknowns x is a matrix of size N_f times N_e.\nNot only do we need to build the matrices hatA^(k ell), also do we have to build interpolation operators from level k to k + 1. These will just be sparse matrices and we just need one on each level of the refined reference element.\nTo apply a \"global\" matrix-vector product with a discretized operator A, we will apply all local operators, which is sufficient for interior nodes of each base mesh element, but for nodes on the boundary of a base mesh element we have to make the data flow and sum the local values together.\nWe have to apply the Dirichlet zero boundary condition by zeroing out nodes on the boundary after multiplication.\nRestricting a function from level k + 1 to k is a completely local operation, but interpolating from k to k+1 also requires summing values along the boundaries of all base mesh elements.We want to accomplish 3, 4 and 5 without storing a global list of connectivity data (i.e. lists of nodes that coincide globally) cause that would run into excessive storage demands again. Rather it is done via a couple sparse mappings solely on the base mesh:From each node to each element (and its local node number);\nFrom each edge to each element (and its local edge number);\nFrom each face to each element (and its local face number) (only in 3D)together with a local pre-computed listing of nodes on corners, the interior of edges and interior of faces of a simplex."
},

{
    "location": "#Putting-this-into-action-1",
    "page": "Tutorial",
    "title": "Putting this into action",
    "category": "section",
    "text": "Let\'s build the above idea in code.First we create a base mesh, which is the coarsest mesh of multigrid:using SparseArrays, LinearAlgebra, Homogenization\nusing Homogenization: generate_conductivity, conductivity_per_element, \n                      list_interior_nodes, assemble_checkerboard,\n                      nnodes, BaseLevel\n\nelementtype = Tri{Float64} # or Tet{Float64}\nn = 32\nbase = hypercube(elementtype, n)\na = conductivity_per_element(base, generate_conductivity(base, n))\nλ = 1.0\ninterior = list_interior_nodes(base)\nF = cholesky(assemble_checkerboard(base, a, λ)[interior,interior])\nbase_level = BaseLevel(Float64, F, nnodes(base), interior)BaseLevel stores a factorized version of the coarse operator and also allocates a vector b and b_intererior. It has all data to efficiently solve a coarse problem with a direct method (it does not pre-allocate a vector x for technical reasons).Now we start working on the implicit grid. We construct many of the things of the previous section in a single call:using Homogenization: ImplicitFineGrid\nrefinements = 3\nimplicit = ImplicitFineGrid(base, refinements)In the REPL this will print a bit of information about the implicit grid, e.g.:julia> implicit\nImplicit grid of cell type Tri{Float64}. Base mesh has 1089 nodes and 2048 elements. Finest level (3) has 15 nodes and 16 elements. In total at most 30720 unknowns.It reads \"In total at most 30720 unknowns\" since this is the product N_e * N_f = 2048 * 15 where nodes on the boundaries of the base mesh are counted multiple times. (The level numbering in the message might be off by one – it\'s the grid after the third refinement).The ImplicitFineGrid type collects a lot of things:julia> implicit.base # reference to the base mesh\njulia> implicit.interfaces.nodes # sparse map from nodes -> connected element + local id\njulia> implicit.interfaces.edges # sparse map from edges -> connected element + local id\njulia> implicit.interfaces.faces # sparse map from faces -> connected element + local id\njulia> implicit.reference.interops # array of interpolation operators from level to level\njulia> implicit.reference.numbering # local numbering of corners, edge and face nodes\njulia> implicit.reference.levels # mesh for each level of the refined reference elementNext, to be able to impose the boundary condition we have to have sparse mappings from the nodes, edges and faces of base mesh elements touching the boundary:using Homogenization: list_boundary_nodes_edges_faces, ZeroDirichletConstraint\nnodes, edges, faces = list_boundary_nodes_edges_faces(implicit.base)\nconstraint = ZeroDirichletConstraint(nodes, edges, faces)Finally we can build the A^(kell) operators:using Homogenization: build_local_diffusion_operators, build_local_mass_matrices,\n                      L2PlusDivAGrad\ndiff_terms = build_local_diffusion_operators(implicit.reference)\nmass_terms = build_local_mass_matrices(implicit.reference)\nlevel_operators = map(zip(diff_terms, mass_terms)) do op\n    diff, mass = op\n    L2PlusDivAGrad(diff, mass, constraint, λ, a)\nendSo level_operators is now an array of these operators for each level of the grid. It has all the values it needs: the value of lambda, the conductivity a per base mesh element and the information to zero out boundary nodes.Next, we allocate the vectors we need for multigrid (for convenience we shape them as matrices as explained above).using Homogenization: LevelState, refined_mesh, nelements, nnodes, base_mesh\nlevel_states = map(1 : refinements) do i\n    mesh = refined_mesh(implicit, i)\n    LevelState(nelements(base_mesh(implicit)), nnodes(mesh), Float64)\nendA LevelState contains just the arrays we need: an approximation to the unknown x, a right-hand side b, and a residual r. We have to have these vectors on every level. These are the only arrays that have a large memory footprint, so that\'s why we allocate them once and for all!julia> size(level_states[end].x)\n(15, 2048)Finally we have to set up the problem. We still have to set up a right-hand side. The right-hand should be constructed locally. For instance:using Homogenization: local_rhs!\nfinest_level = level_states[end]\nlocal_rhs!(finest_level.b, implicit) # integrate v dx locallyNote that local_rhs! does not accept a functional currently, but this function is very simle and can be written by the user basically. Currently it is just hard-coded to use f = 1.Then, we take an initial guess x, this vector should be constructed globally and should satisfy the boundary condition:using Homogenization: broadcast_interfaces!, apply_constraint!\nusing Random\nrand!(finest_level.x) # local values\nbroadcast_interfaces!(finest_level.x, implicit, refinements) # sum boundaries\napply_constraint!(finest_level.x, refinements, constraint, implicit) # impose b.c.Our initial guess resides in finest_level.x.julia> finest_level.x\n15×2048 Array{Float64,2}:\n...Now we can run multigrid iterations:using Homogenization: vcycle!, zero_out_all_but_one!\n\nsmoothing_steps = 1\n\nfor i = 1 : 100\n    vcycle!(implicit, base_level, level_operators, level_states, refinements, smoothing_steps)\n\n    zero_out_all_but_one!(finest_level.r, implicit, refinements)\n    @info \"After cycle $i\" norm(finest_level.r)\nendRunning this, the output should be something like:...\n┌ Info: After cycle 98\n└   norm(finest_level.r) = 0.0005182895775368055\n┌ Info: After cycle 99\n└   norm(finest_level.r) = 0.00047190444233626385\n┌ Info: After cycle 100\n└   norm(finest_level.r) = 0.00042970384073489823Note that computing the norm is done with a hack: we zero out all values of repeated nodes and then take the Euclidean norm.Finally, we might wish to see what the approximate solution looks like. In our case we can still easily store the full grid as the problem is rather small, but in real applications we might not be able to export the full grid. What we can do quite easily is to extract the approximate solution on a certain level. We have to select the values of x on the finest grid, but only those that appear on a coarser level. Fortunately by convention upon refinement new implicit nodes are appended to the back to the nodes array, which means that we can just extract the first so-many rows of x if we want to have the approximate solution on a coarser grid.For example:using Homogenization: construct_full_grid, nnodes, refined_mesh\nusing WriteVTK \n\n# Construct a full mesh compatible with Paraview (nodes are repeated here as well!)\nsave = 2\nfull_mesh = construct_full_grid(implicit, save)\n\nvtk_grid(\"checkerboard_solution\", full_mesh) do vtk\n    x_on_level = reshape(finest_level.x[1 : nnodes(refined_mesh(implicit, save)), :], :)\n    vtk_point_data(vtk, x_on_level, \"x\")\nendIt saves the file checkerboard_solution.vtu and should roughly look like(Image: Multigrid solution)where we look at level 2 with a single refinement."
},

]}
