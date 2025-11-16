# author: adle ben salem

import os
from pathlib import Path

PATH_CURRENT_FOLDER: str = os.path.abspath(os.getcwd())
PATH_BENCHMARK_FOLDER: str = str(Path(PATH_CURRENT_FOLDER) / Path("benchmark"))
PATH_IMAGES_FOLDER: str = str(Path(PATH_CURRENT_FOLDER) / Path("images"))

# We test value of t from 15 to 60 degrees
BEGIN_T_VALUE: int = 15
END_T_VALUE: int = 60

# We have 6 options to test (excluding -t)
OPTIONS = {
    "n",
    "p",
    "f",
    "c",
    "e",
    "w",
}
