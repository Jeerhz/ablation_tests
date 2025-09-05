import os
from pathlib import Path

PATH_CURRENT_FOLDER: str = os.path.abspath(os.getcwd())
PATH_BENCHMARK_FOLDER: str = str(Path(PATH_CURRENT_FOLDER) / Path("benchmark"))
PATH_IMAGES_FOLDER: str = str(Path(PATH_CURRENT_FOLDER) / Path("images"))
