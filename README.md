# SpaceGA
Tools for accelerated searching of chemical spaces using molecular docking.

# Requirements

# How To Run
The repository supports three different types of virtual screens:

| Mode    | Description                   |
|---------|-------------------------------|
| al      | Active Learnign               |
| ga      | Graph-based genetic algorithm |
| spacega | SpaceGA                       |

Virtual screens can be initiated using the respective notebooks, i.e. <Mode>.ipynb, or directly in the command line by supplementing a .json file:
python main.py <Mode> <path-to-json-file>.json

The notebooks are a great place to generate .json files for the desired virtual screens.
