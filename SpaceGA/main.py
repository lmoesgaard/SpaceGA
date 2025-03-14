import sys
import json
import importlib
from spacega import SpaceGA

def get_scoring_tool(module, tool):
    module = importlib.import_module(module)
    try:
        return getattr(module, tool)
    except:
        print(f"Could not find scorer: {tool}")
        sys.exit()

def main(json_filename, generation):
    input = json.load(open(json_filename))
    spacega = SpaceGA(generation=generation, **input)
    scorer = get_scoring_tool(**input["scoring_tool"])(spacega.__dict__)
    assert scorer is not None, "Scorer not set"
    while spacega.gen <= spacega.iterations:
        spacega.gen += 1
        offspring = spacega.reproduce()
        offspring["scores"] = scorer.score(offspring.smi.to_list(), offspring.name.to_list())
        spacega.population = spacega.update_pop(offspring)


if __name__ == '__main__':
    json_filename = sys.argv[1]
    if len(sys.argv) > 2:
        assert sys.argv[2].isdigit(), "Second argument must be an integer"
        main(json_filename, int(sys.argv[2]))
    else:
        main(json_filename, 0)