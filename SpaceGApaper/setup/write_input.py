import shutil
import os
import json
import sys


def save_config(settings):
    json_file = settings.config
    with open(json_file, 'w') as f:
        json.dump(settings.__dict__, f, indent=4, sort_keys=True)


def read_config(config_file):
    with open(config_file, 'r') as f:
        input_data = json.load(f)
    return input_data

def make_workdir(settings):
    if os.path.exists(settings["o"]) and os.path.isdir(settings["o"]):
        if settings["O"]:
            print('Overwriting output')
            shutil.rmtree(settings["o"])
            os.mkdir(settings["o"])
        else:
            print('Output path found. Not overwriting.')
            sys.exit()
    else:
        try:
            os.mkdir(settings["o"])
        except:
            print(f'Could not make output directory')
            sys.exit()
    try:
        shutil.copy(settings["config"], os.path.join(settings["o"], "config.json"))
    except:
        pass