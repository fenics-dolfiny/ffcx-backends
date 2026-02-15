import importlib.util
import pathlib

spec = importlib.util.find_spec("ffcx")
ffcx_dir = pathlib.Path(spec.submodule_search_locations[0])

print(ffcx_dir)
