import importlib.util
import pathlib

spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
nvrtc_dir = pathlib.Path(spec.submodule_search_locations[0])

print(nvrtc_dir)
