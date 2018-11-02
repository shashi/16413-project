This folder contains a Julia notebook and accompanying Julia code.

## Instructions for setting up Julia and the Jupyter kernel for it

Julia can be installed for your respective OS by downloading it from the [downloads page](https://julialang.org/downloads/) (version v1.0.1).

Once install it, you can open the julia prompt by running the Julia executable.

Preferably, cd to this directory and then run Julia, if you can't do that, then start Julia and run

```julia
[shashi@laptop part2-alphago-zero-tictactoe]$ ~/code/julia/julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.0.2-pre.0 (2018-09-30)
 _/ |\__'_|_|_|\__'_|  |  release-1.0/4aea3d2b7c* (fork: 174 commits, 86 days)
|__/                   |

julia> cd("path/to/this/dir")
```


### Installing Jupyter and the kernel

You can set up Julia with Jupyter by installing the IJulia package.

```julia
julia> using Pkg

julia> Pkg.add("IJulia")
...
```

If you do not have Jupyter this step will install it for you within ~/.julia (it does not pollute any other python environment you may have).

Then you can activate the project's environment by running the following:

```julia
julia> using Pkg
julia> Pkg.activate("env")
"/home/shashi/code/16413-project/alphago-tictactoe/part2-alphago-zero-tictactoe/env/Project.toml"
```

Then `Pkg.resolve()` will install all the dependencies of the notebook.

```julia
julia> Pkg.resolve()
  Updating registry at `~/.julia/registries/General`
  Updating git-repo `https://github.com/JuliaRegistries/General.git`
 Resolving package versions...
  Updating `~/code/16413-project/alphago-tictactoe/part2-alphago-zero-tictactoe/env/Project.toml`
 [no changes]
  Updating `~/code/16413-project/alphago-tictactoe/part2-alphago-zero-tictactoe/env/Manifest.toml`
 [no changes]
```

It may take a while to set up everything. Ignore any errors to do with the packages `CUDAnative` and `CUDAdrv` they are harmless for our purposes.


To run the jupyter notebook server, run:
```julia
julia> using IJulia

julia> notebook(dir=".")
```

This will bring up the notebook server in this directory. Click on "AlphaGo Zero - Mastering the game of Go" notebook.

Note that Julia is a just-in-time compiled language. This means the first time you run code, it needs to compile it (which may be slow). Expect to take about 10 minutes to execute the whole notebook the first time from beginning to end, however, you can go back up and run cells and have them be instantanious.

## Troubleshooting

- [Troubleshooting guide for IJulia](https://github.com/JuliaLang/IJulia.jl#troubleshooting)
- If you have trouble installing stuff or anything breaks even after a smooth setup please send me an email about it or open an issue on this repository.

## Fallback static version

(this doesn't have the UI to play the game), but the content should be there.

- [NBViewer link](https://nbviewer.jupyter.org/github/shashi/16413-project/blob/master/part2-alphago-zero-tictactoe/AlphaGo%20Zero%20--%20Mastering%20the%20game%20of%20Go.ipynb)
