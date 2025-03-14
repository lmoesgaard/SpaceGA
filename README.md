# SpaceGA

**SpaceGA** is a tool for accelerated searching of chemical spaces using molecular docking. It integrates the similarity search tool [SpaceLight](https://www.biosolveit.de/spacelight-a-spotlight-on-the-analog-hunter-for-chemical-spaces/) by BiosolveIT with a [graph-based genetic algorithm (GB-GA)](http://dx.doi.org/10.1039/C8SC05372C) to efficiently navigate large combinatorial spaces.

## Overview

SpaceGA explores chemical spaces by using the same crossover module as the GB-GA by Jensen but replaces the mutation module with a similarity search using SpaceLight. This allows SpaceGA to identify "available mutations" within a given chemical space, ensuring that all generated molecules exist within the specified library.

For reproducing the results from the [SpaceGA paper](https://doi.org/10.1021/acs.jcim.4c01308), refer to the `./SpaceGApaper` directory. The most recent version of SpaceGA is available in the `./SpaceGA` directory.

## Installation

You can install SpaceGA using pip:

```sh
pip install -e .
```

## Requirements

- **[SpaceLight](https://www.biosolveit.de/products/#SpaceLight)**: A valid license is required to use SpaceGA.
- **[Lilly Medchem Rules](https://github.com/IanAWatson/Lilly-Medchem-Rules)** (optional): Supports molecular filtering.
- Additional dependencies are listed in `requirements.txt`.

## Usage

You can run SpaceGA directly from the command line using a `.json` file as input:

```sh
python3 /path/to/SpaceGA/main.py /path/to/inputs.json
```

Alternatively, you can use the provided Jupyter notebooks:
- `SpaceGAmanual.ipynb`
- `SpaceGAautomated.ipynb`

These notebooks serve as implementation examples in a Python script.

## Supported Arguments

SpaceGA accepts various arguments specified either in the input `.json` file or when initializing SpaceGA.

### Required Arguments

| Argument     | Type  | Description                                  | Default        |
| ------------ | ----- | -------------------------------------------- | -------------- |
| `o`          | `str` | Path to output directory                     | `'output'`     |
| `i`          | `str` | Path to `.smi` file with input molecules     | `'input.smi'`  |
| `space`      | `str` | Path to the desired BiosolveIT space         | `'space'`      |
| `spacelight` | `str` | Path to the SpaceLight installation          | `'spacelight'` |

### Optional Arguments

| Argument         | Type    | Description                                                       | Default         |
| --------------- | ------- | ----------------------------------------------------------------- | --------------- |
| `O`             | `bool`  | Allow overwrite of output                                        | `False`         |
| `iterations`    | `int`   | Number of iterations                                             | `10`            |
| `p_size`        | `int`   | Population size                                                  | `100`           |
| `children`      | `int`   | Number of molecules evaluated per iteration divided by `p_size` | `100`           |
| `crossover_rate`| `float` | Crossover rate                                                   | `0.2`           |
| `cpu`           | `int`   | Number of CPUs available                                         | `1`             |
| `sl_cpu`        | `int`   | Number of CPUs for SpaceLight processing                         | Same as `cpu`   |
| `al`            | `bool`  | Use active learning-based ML filtering of offspring              | `False`         |
| `sim_cutoff`    | `float` | Similarity cutoff after each iteration (`1.00` means no filter)  | `1.00`          |
| `f_comp`        | `int`   | Number of top `f_comp * children` similar molecules retained    | `100`           |
| `fp_type`       | `str`   | Fingerprint type for SpaceLight                                  | `ECFP4`         |
| `model_name`    | `str`   | Machine learning model name (stored in `SpaceGA/ml/models.py`)  | `'NN1'`         |
| `scoring_tool`  | `dict`  | Dictionary specifying the scoring function                      | `{...}`         |
| `scoring_inputs`| `dict`  | Inputs for the scoring function                                 | `{}`            |
| `filtering_inputs` | `dict` | Inputs for filtering molecules                              | `{}`            |

## Scoring Functions

SpaceGA includes two built-in scoring functions that can be specified as follows:

```python
scoring_tool = {"module": "SpaceGA.scoring", "tool": scoringtool}
```

### Available Scoring Functions

- **FPSearch**: Maximizes Tanimoto similarity to a query molecule.
- **LogPSearch**: Maximizes logP.

### FPSearch Arguments

| Argument | Type  | Description                                   |
| -------- | ----- | --------------------------------------------- |
| `smiles` | `str` | Query SMILES string for similarity maximization |

### LogPSearch Arguments

No additional arguments are required for `LogPSearch`.

### Custom Scoring Functions

Users can define and import custom scoring functions following this format:

```python
from abc import ABC, abstractmethod

class Scorer(ABC):
    @abstractmethod
    def __init__(self, arguments):
        pass
    
    def score(self, smi_lst, name_lst):
        pass
```

Here, `arguments` is a dictionary containing SpaceGA settings, ensuring seamless integration.

## Filtering Options

SpaceGA supports filtering based on molecular properties, which can be configured using a dictionary. Below are the available filtering options:

| Argument   | Type   | Description                                                        | Default |
| ---------- | ------ | ------------------------------------------------------------------ | ------- |
| `minlogP`  | `float` | Minimum logP value                                               | None    |
| `maxlogP`  | `float` | Maximum logP value                                               | None    |
| `minMw`    | `float` | Minimum molecular weight                                         | None    |
| `maxMw`    | `float` | Maximum molecular weight                                         | None    |
| `minHBA`   | `int`   | Minimum number of hydrogen bond acceptors                        | None    |
| `maxHBA`   | `int`   | Maximum number of hydrogen bond acceptors                        | None    |
| `minHBD`   | `int`   | Minimum number of hydrogen bond donors                           | None    |
| `maxHBD`   | `int`   | Maximum number of hydrogen bond donors                           | None    |
| `minRings` | `int`   | Minimum number of rings                                          | None    |
| `maxRings` | `int`   | Maximum number of rings                                          | None    |
| `minRotB`  | `int`   | Minimum number of rotatable bonds                               | None    |
| `maxRotB`  | `int`   | Maximum number of rotatable bonds                               | None    |
| `BRENK`    | `bool`  | Apply [BRENK](https://doi.org/10.1002/cmdc.200700139) filter    | False   |
| `PAINS`    | `bool`  | Apply [PAINS](https://doi.org/10.1021/jm901137j) filter         | False   |
| `Lilly`    | `str`   | Path to [Lilly_Medchem_Rules.rb ](https://github.com/IanAWatson/Lilly-Medchem-Rules)                              | None    |
| `Substructure` | `str` | SMART string of required substructure                          | None    |

