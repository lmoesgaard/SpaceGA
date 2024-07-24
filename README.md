# SpaceGA
Tools for accelerated searching of chemical spaces using molecular docking. SpaceGA is a new screening method that utilizes the similarity search tool [SpaceLight](https://www.biosolveit.de/spacelight-a-spotlight-on-the-analog-hunter-for-chemical-spaces/) by BiosolveIT and a [graph-based genetic algorithm (GB-GA)](http://dx.doi.org/10.1039/C8SC05372C) to navigate large combinatorial spaces. In short, SpaceGA explores the desired chemical space using the same crossover module as the GB-GA by Jensen but replaces the mutation module with a similarity search using SpaceGA to identify "available mutations". In this way, SpaceGA is restricted to the suggested molecules in the supplemented space only.

# Requirements
SpaceLight requires a license, so such a license is required to run SpaceGA as well.

To run the tools in DockScreen mode, [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU) is required.

AL is built around the GPU implementation of [PyTorch](https://pytorch.org/).

Filtering molecules using [Lilly Medchem Rules](https://github.com/IanAWatson/Lilly-Medchem-Rules) is supported but not required.

Additional requirements are available in requirements.txt.

# How To Run
The repository supports three different types of virtual screens:

| Mode    | Description                                                                       |
|---------|-----------------------------------------------------------------------------------|
| AL      | Active Learning                                                                   |
| GA      | [ graph-based genetic algorithm (GB-GA) ]( http://dx.doi.org/10.1039/C8SC05372C ) |
| SpaceGA | SpaceGA                                                                           |

Virtual screens can be initiated using the respective notebooks, i.e. `<Mode>.ipynb`, or directly in the command line by supplementing a `.json` file:
`python main.py <Mode> <path-to-json-file>.json`

The notebooks are a great place to generate .json files for the desired virtual screens.

## Specifically for AL
Before running, please ensure that fingerprint.npz and SMILES.parquet files are available. The data/convert.py is supplied to demonstrate the conversion of space-separated `.smi` files. Enamine REAL subsets can be downloaded from their [website](https://enamine.net/compound-collections/real-compounds/real-database-subsets). We recommend having no more than 100.000 molecules in each batch file.

# Supported Arguments
Templates for setting up `.json` configuration files are available in the respective notebooks (`<Mode>.ipynb`).
## Search Tool Arguments
| Argument            | dtype    | Description                                                                                         <tr><td colspan="3">**Recurring arguments**</td></tr>
|---------------------|----------|-----------------------------------------------------------------------------------------------------|
| `O`                 | `bool`   | Overwrite output directory                                                                          |
| `o`                 | `str`    | Path to output directory                                                                            |
| `iterations`        | `int`    | Number of iteration                                                                                 |
| `cpu`               | `int`    | Number of CPUs available                                                                            |
| `gpu`               | `int`    | Number of GPUs available                                                                            |
| `mode`              | `str`    | Scoring function (FPSearch, LogPSearch or DockSearch)                                               |
| `scoring_inputs`    | `dict`   | Dictionary with inputs for the scoring function (see below)                                         <tr><td colspan="3">**AL**</td></tr>
| `i`                 | `str`    | Path to directory with the `smis` and `fps` directories. These can prepared using `data/convert.py` |
| `p_size`            | `int`    | Number of molecules evaluated each iteration                                                        |
| `model_name`        | `str`    | Name of ML model to use. ML models are stored in `ml/models.py`                                     |
| `max_models`        | `int`    | Max number of models to run on each GPU for prediction                                              |
| `bsize`             | `int`    | Batch size when training the ML model                                                               |
| `init_split`        | `list`   | How to split data into train/val/test set after first iteration                                     <tr><td colspan="3">**GA**</td></tr>
| `i`                 | `str`    | Path to `.smi` file with molecules to sample the start population                                   |
| `p_size`            | `int`    | Population size                                                                                     |
| `children`          | `ìnt`    | Number of molecules to evaluate per iteration divided by `p_size`                                   |
| `crossover_rate`    | `float`  | Crossover rate                                                                                      |
| `mutation_rate`     | `float`  | Mutation rate                                                                                       |
| `sim_cutoff`        | `float`  | Similarity cutoff applied after each iteration (`1.00`: no filtering)                               |
| `filtering_inputs`  | `dict`   | Dictionary with inputs for filtering (see below)                                                    <tr><td colspan="3">**SpaceGA**</td></tr>
| `i`                 | `str`    | Path to `.smi` file with molecules to sample the start population                                   |
| `p_size`            | `ìnt`    | Population size                                                                                     |
| `children`          | `ìnt`    | Number of molecules to evaluate per iteration divided by `p_size`                                   |
| `crossover_rate`    | `float`  | Crossover rate                                                                                      |
| `sim_cutoff`        | `float`  | Similarity cutoff applied after each iteration (`1.00`: no filtering)                               |
| `space`             | `str`    | Path to desired BiosolveIT space                                                                    |
| `spacelight`        | `str`    | Path to spacelight executable                                                                       |
| `f_comp`            | `int`    | Find top `f_comp`*`children` most similar molecules to compensate for filtering                         |
| `filtering_inputs`  | `dict`   | Dictionary with inputs for filtering (see below)                                                    |

## Scoring Function Arguments
Currently, three different scoring functions are supported: FPSearch, LogPSearch, and DockSearch. FPSearch seeks to maximize Tanimoto similarity to a query molecule. LogPSearch seeks to maximize logP. DockSearch seeks to minimize [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU) docking scores. The scoring arguments are given as a dictionary. The following arguments are required.
| Argument   | dtype | Description                                                                     <tr><td colspan="3">**FPSearch**</td></tr>
|:-----------|:------|:--------------------------------------------------------------------------------|
| "smiles"   | `str` | Query SMILES-string to maximize similarity to                                   <tr><td colspan="3">**LogPSearch**</td></tr>
| `None`     |       |                                                                                 <tr><td colspan="3">**DockSearch**</td></tr>
| "fld_file" | `str` | Path to `.fld` file                                                             |
| "autodock" | `str` | Path to [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU) executable |
| "workdir"  | `str` | Path to working directory                                                       |
| "obabel"   | `str` | Path to [Open Babel](https://openbabel.org/index.html#) executable              |

## Filtering Arguments
Filtering is available for GA and SpaceGA. The filtering arguments are given as a dictionary, only including filters that should be applied. The following arguments are available.
| Argument   | dtype   | Description                                                                          |
|------------|---------|--------------------------------------------------------------------------------------|
| "minlogP"  | `float` | Lowest tolerated logP                                                                |
| "maxlogP"  | `float` | Highest tolerated logP                                                               |
| "minMw"    | `float` | Lowest tolerated molecular weight                                                    |
| "maxMw"    | `float` | Highest tolerated molecular weight                                                   |
| "minHBA"   | `int`   | Lowest tolerated number of hydrogen bond acceptors                                   |
| "maxHBA"   | `int`   | Highest tolerated number of hydrogen bond acceptors                                  |
| "minHBD"   | `int`   | Lowest tolerated number of hydrogen bond donors                                      |
| "maxHBD"   | `int`   | Highest tolerated number of hydrogen bond donors                                     |
| "minRings" | `int`   | Lowest tolerated number of rings                                                     |
| "maxRings" | `int`   | Highest tolerated number of rings                                                    |
| "minRotB"  | `int`   | Lowest tolerated number of rotatable bonds                                           |
| "maxRotB"  | `int`   | Highest tolerated number of rotatable bonds                                          |
| "BRENK"    | `bool`  | Apply [BRENK](https://doi.org/10.1002/cmdc.200700139) filter                         |
| "PAINS"    | `bool`  | Apply [PAINS](https://doi.org/10.1021/jm901137j) filter                              |
| "Lilly"    | `str`   | Path to [Lilly_Medchem_Rules.rb ](https://github.com/IanAWatson/Lilly-Medchem-Rules) |
