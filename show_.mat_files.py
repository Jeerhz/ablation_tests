# This script assumes that downloading the YorkUrban-LineSegment returns a
# folder named 'LineSegmentAnnotation' in the current directory.

import os
from pathlib import Path
from scipy.io import loadmat
from loguru import logger
from typing import Any, Mapping
import numpy as np
import shutil
import glob
import requests
import zipfile

current_folder_path: str = os.path.abspath(os.getcwd())


# see https://stackoverflow.com/questions/43515481/python-how-to-move-list-of-folders-with-subfolders-to-a-new-directory
def move_folder(origine_path: str, destination_path: str) -> None:
    try:
        shutil.move(origine_path, destination_path)
        logger.info(f"Folder moved from {origine_path} to {destination_path}")
    except Exception as e:
        logger.warning(f"Failed to move folder: {e}")


def download_york_urban_dataset() -> None:
    """Download the YorkUrban-LineSegment dataset from Dropbox and extract it to the current directory."""
    url = "https://www.dropbox.com/scl/fi/lzm76drgp97mmy0fpygz7/YorkUrban-LineSegment.zip?rlkey=emo2b65onipofxikrdzd2kufm&dl=1"
    local_zip_path = "YorkUrban-LineSegment.zip"

    if os.path.isdir(str(Path(current_folder_path) / Path("LineSegmentAnnotation"))):
        logger.warning(
            f"Folder 'LineSegmentAnnotation' already exists in {current_folder_path}"
        )
        return None

    logger.info(f"Downloading dataset from Dropbox: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded zip file to {local_zip_path}")

        # Extract the zip file
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(current_folder_path)
        logger.info(f"Extracted dataset to {current_folder_path}")

        # Optionally remove the zip file after extraction
        os.remove(local_zip_path)
    except Exception as e:
        logger.error(f"Failed to download or extract dataset: {e}")


def load_mat_content(
    path_mat_file: str,
) -> Mapping[str, np.ndarray[tuple[int, int], np.dtype[Any]] | Any] | None:
    if not os.path.isfile(path_mat_file):
        logger.error(f"File does not exist: {path_mat_file}")

    if not path_mat_file.endswith(".mat"):
        logger.warning(f"File does not have a .mat extension: {path_mat_file}")

    try:
        mat_data = loadmat(path_mat_file, squeeze_me=True)
    except Exception as e:
        logger.exception(f"Failed to load .mat file: {e}")

    if not mat_data:
        logger.info("Loaded .mat file is empty.")
        return None

    # Remove meta entries automatically added by scipy
    mat_data = {k: v for k, v in mat_data.items() if not k.startswith("__")}
    return mat_data


def show_mat_content(path_mat_file: str) -> None:
    """
    Load and display the content of a .mat file using scipy, logging the output with loguru.

    Args:
        path_mat_file (str): Path to the .mat file.
    """
    mat_data = load_mat_content(path_mat_file=path_mat_file)
    if mat_data is None:
        logger.warning("No data to show")
        return None

    logger.info(f"Contents of '{path_mat_file}':\n{mat_data}")


def export_GT_py(mat_folder: str) -> None:
    """
    Mimics the MATLAB export_GT.m functionality:
    For each 'P*_GND.mat' file in mat_folder, loads 'line_gnd' and writes it to a '_gt.txt' file.
    """
    path_benchmark_folder: str = str(Path(current_folder_path) / Path("benchmark"))

    if os.path.isdir(path_benchmark_folder):
        logger.warning("Benchmark folder already exists.")
        return None
    else:
        os.mkdir(path_benchmark_folder)

    mat_files: list[str] = glob.glob(os.path.join(mat_folder, "P*_GND.mat"))
    logger.info(f"Matched {len(mat_files)} files: \n {mat_files[0:5]}")
    for mat_file in mat_files:
        mat_data = load_mat_content(mat_file)
        if mat_data is None or "line_gnd" not in mat_data:
            logger.warning(f"'line_gnd' not found in {mat_file}")
            continue
        line_gnd = mat_data["line_gnd"]

        txt_filename = os.path.join(
            path_benchmark_folder, f"{os.path.basename(mat_file)[:8]}_gt.txt"
        )
        np.savetxt(txt_filename, line_gnd, fmt="%.6f", delimiter=" ")
        logger.info(f"Exported {txt_filename}")


# Example usage:
download_york_urban_dataset()
export_GT_py(str(Path(current_folder_path) / "LineSegmentAnnotation"))


# if __name__ == "__main__":
# download_york_urban_dataset()
# path_mat_file = (
#     "/home/adle/Bureau/ablation_tests/1/YorkUrbanDB/P1020171/P1020171LinesAndVP.mat"
# )
# show_mat_content(path_mat_file=path_mat_file)
