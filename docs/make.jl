using Documenter, Homogenization

makedocs(
	modules = [Homogenization],
	format = :html,
	doctest = false,
	clean = true,
	sitename = "Homogenization.jl",
	pages = [
		"Home" => "index.md"
	]
)

deploydocs(repo = "github.com/haampie/Homogenization.jl.git")
