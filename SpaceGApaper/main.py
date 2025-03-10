import sys
import importlib

def get_search_tool(mode):
    module = importlib.import_module("modes")
    try:
        return getattr(module, mode)
    except:
        print(f"Could not find mode: {mode}")
        sys.exit()

def main(mode, json_filename):
    search_tool = get_search_tool(mode)
    tool = search_tool(json_filename)
    tool.run()
    return tool

if __name__ == '__main__':
    mode = sys.argv[1]
    json_filename = sys.argv[2]
    main(mode, json_filename)