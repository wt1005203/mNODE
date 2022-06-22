# mNODE (Metabolomic profile predictor using Neural Ordinary Differential Equations)
This repository contains scripts needed to run `mNODE` (Metabolomic profile predictor using Neural Ordinary Differential Equations) that is capable of predicting metabolomic profiles based on microbial compositions and other additional information such as dietary information. 
![schematic](schematic.png)

## Versions
The version of Python we used is 3.7.3 and the version of Julia we used is 1.6.2.

## Dependencies
Necessary Python packages can be found in "requirements.txt". Installing those packages can be achieved by pip:
```
pip install -r requirements.txt
```
Julia uses the toml to manage the version of packages. Necessary Julia packages are specified in "Project.toml" and "Manifest.toml". To ensure all dependent packages will be correctly installed, we include
```
using Pkg
Pkg.instantiate()
```
which automatically installs all packages specified in "Project.toml" and "Manifest.toml".

## Example
We showed a demonstration of mNODE on the dataset PRISM + NLIBD. First, we need to process both metagenomic profiles and metabolomic profiles using the CLR (Centred Log-Ratio) transformation. The data is processed by the Python script titled "data_processing.py":
```
<PATH_TO_PYTHON> data_processing.py
```
<PATH_TO_PYTHON> is the path to the executable Python file located under the installed folder. After the data processing, we can run mNODE (including hyperparameter calibration and the following predictions) contained in "mNODE.jl" via the command:
```
<PATH_TO_JULIA> ./mNODE.jl
```
<PATH_TO_JULIA> is the path to the executable Julia file located under the installed folder. Finally, to infer microbe-metabolite interactions via the susceptibility method, we can run "inferring_interactions.jl":
 ```
<PATH_TO_JULIA> ./inferring_interactions.jl
```

For example, on the Macbook air M1 2020 we tested, the command for running "mNODE.jl" is 
```
/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia ./mNODE.jl
```
