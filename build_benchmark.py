# author: adle ben salem

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
from tqdm import tqdm

from config import PATH_BENCHMARK_FOLDER, PATH_CURRENT_FOLDER, PATH_IMAGES_FOLDER  # type: ignore


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

    if os.path.isdir(str(Path(PATH_CURRENT_FOLDER) / Path("LineSegmentAnnotation"))):
        logger.warning(
            f"Folder 'LineSegmentAnnotation' already exists in {PATH_CURRENT_FOLDER}"
        )
        return None

    logger.info(f"Downloading dataset from Dropbox: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with (
            open(local_zip_path, "wb") as f,
            tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        logger.info(f"Downloaded zip file to {local_zip_path}")

        # Extract the zip file
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(PATH_CURRENT_FOLDER)
        logger.info(f"Extracted dataset to {PATH_CURRENT_FOLDER}")

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
    """
    mat_data = load_mat_content(path_mat_file=path_mat_file)
    if mat_data is None:
        logger.warning("No data to show")
        return None

    logger.info(f"Contents of '{path_mat_file}':\n{mat_data}")


def export_GT_py(
    mat_folder: str,
    path_destination_folder: str = PATH_BENCHMARK_FOLDER,
    with_makefile: bool = True,
) -> None:
    """
    Mimics the MATLAB export_GT.m functionality:
    For each 'P*_GND.mat' file in mat_folder, loads 'line_gnd' and writes it to a '_gt.txt' file.
    """
    if os.path.isdir(path_destination_folder):
        logger.info("Benchmark folder already exists.")
        # Check if benchmark folder contains .txt files
        txt_files = glob.glob(os.path.join(path_destination_folder, "*.txt"))
        if len(txt_files) == 102:
            logger.info("Benchmark folder contains 102 .txt files. Returning None.")
            return None
    else:
        os.mkdir(path_destination_folder)

    mat_files: list[str] = glob.glob(os.path.join(mat_folder, "P*_GND.mat"))
    logger.info(f"Matched {len(mat_files)} files: \n {mat_files[0:5]}")
    for mat_file in mat_files:
        mat_data = load_mat_content(mat_file)
        if mat_data is None or "line_gnd" not in mat_data:
            logger.warning(f"'line_gnd' not found in {mat_file}")
            continue
        line_gnd = mat_data["line_gnd"]

        txt_filename = os.path.join(
            path_destination_folder, f"{os.path.basename(mat_file)[:8]}_gt.txt"
        )
        np.savetxt(txt_filename, line_gnd, fmt="%.6f", delimiter=" ")
    logger.info(
        f"Exported {len(mat_files)} ground truth files to {path_destination_folder}"
    )

    if not with_makefile:
        logger.info("Skipping Makefile copy.")
        return None

    # Copy original Makefile to path_destination_folder
    logger.info("Copying Makefile to benchmark folder.")
    makefile_src = os.path.join(PATH_CURRENT_FOLDER, "Makefile")
    makefile_dst = os.path.join(path_destination_folder, "Makefile")
    if os.path.isfile(makefile_src):
        shutil.copy(makefile_src, makefile_dst)
        logger.info(f"Copied Makefile to {makefile_dst}")
    else:
        logger.warning(f"Makefile not found in {PATH_CURRENT_FOLDER}")


def extract_images_from_dataset_folders(
    containing_folder_path: str = PATH_CURRENT_FOLDER,
    destination_path: str = PATH_BENCHMARK_FOLDER,
) -> None:
    """
    Extract images from dataset folders and save them to a benchmark folder.
    """
    if os.path.isdir(destination_path):
        logger.info("Images folder already exists.")
        jpg_files = glob.glob(os.path.join(destination_path, "*.jpg"))
        if len(jpg_files) == 102:
            logger.info("Images folder contains 102 .jpg files. Returning None.")
            return None
    else:
        os.mkdir(destination_path)

    # Get all image path
    images_files = glob.glob(
        os.path.join(containing_folder_path, "P*", "*.jpg"), recursive=True
    )
    if not images_files:
        logger.warning("No images found.")
        return None

    # Move all image files in benchmark folder
    for image_file in images_files:
        shutil.copy(image_file, destination_path)
    logger.info(f"Moved all images to benchmark folder: {destination_path}")


def clean_working_directory() -> None:
    """
    Delete downloaded dataset to only keep benchmark files, ie,
    Delete folder LineSegmentAnnotation and all folders called 'P*', plus some specific files/folders.
    """
    # Folders to remove (wildcards and explicit names)
    folders_to_remove = glob.glob(os.path.join(PATH_CURRENT_FOLDER, "P*"))
    folders_to_remove += [
        os.path.join(PATH_CURRENT_FOLDER, "LineSegmentAnnotation"),
        os.path.join(PATH_CURRENT_FOLDER, "pic_only"),
    ]
    for folder_path in folders_to_remove:
        shutil.rmtree(folder_path, ignore_errors=True)
    logger.info("Cleaned working directory from folders to remove")

    # Files to remove
    files_to_remove = [
        "cameraParameters.mat",
        "ECCV_TrainingAndTestImageNumbers.mat",
    ]
    for file_name in files_to_remove:
        file_path = os.path.join(PATH_CURRENT_FOLDER, file_name)
        try:
            os.remove(file_path)
            logger.info(f"Cleaned working directory: {file_name}")
        except FileNotFoundError:
            pass


def symlink_images_to_folder(
    path_images_folder: str, path_destination_folder: str
) -> None:
    """
    Create symlinks for all images in path_images_folder inside path_destination_folder.
    """
    if not os.path.isdir(path_images_folder):
        logger.error(f"Images folder does not exist: {path_images_folder}")
        return None

    if not os.path.isdir(path_destination_folder):
        logger.warning(f"Destination folder does not exist: {path_destination_folder}")
        os.makedirs(path_destination_folder, exist_ok=True)

    image_files = glob.glob(os.path.join(path_images_folder, "*.jpg"))
    if not image_files:
        logger.warning("No images found to symlink.")
        return None

    for img_path in image_files:
        symlink_path = os.path.join(path_destination_folder, os.path.basename(img_path))
        try:
            if not os.path.exists(symlink_path):
                os.symlink(img_path, symlink_path)
        except Exception as e:
            logger.error(f"Failed to create symlink for {img_path}: {e}")
    logger.info(f"Symlinked {len(image_files)} images to {path_destination_folder}")


def setup_benchmark_environment() -> None:
    """
    Set up the benchmark environment by creating necessary directories and files.
    """
    os.makedirs(PATH_BENCHMARK_FOLDER, exist_ok=True)
    logger.info(f"Created benchmark folder: {PATH_BENCHMARK_FOLDER}")
    download_york_urban_dataset()
    export_GT_py(str(Path(PATH_CURRENT_FOLDER) / "LineSegmentAnnotation"))
    extract_images_from_dataset_folders(destination_path=PATH_IMAGES_FOLDER)
    symlink_images_to_folder(
        path_images_folder=PATH_IMAGES_FOLDER,
        path_destination_folder=PATH_BENCHMARK_FOLDER,
    )
    clean_working_directory()


def modify_makefile_for_options(
    benchmark_folder: str, options: set[str] | dict[str, int]
) -> None:
    """
    Change the Makefile to pass the options to the C++ script.
    """
    makefile_path = os.path.join(benchmark_folder, "Makefile")
    if not os.path.isfile(makefile_path):
        logger.error(f"Makefile not found in {benchmark_folder}")
        return

    # Filter empty options
    if type(options) is dict:
        option_value = options["t"]
        options_str = f"-t {option_value}"
    else:
        filtered_options = [opt for opt in options if opt]
        if filtered_options:
            options_str = "-" + "".join(sorted(filtered_options))
        else:
            options_str = ""

    # Load Makefile
    with open(makefile_path, "r") as f:
        lines = f.readlines()

    # Change line LSD ?= Compare ?= to point to the correct path
    muLSD_path = os.path.abspath(os.path.join(benchmark_folder, "../../Build/muLSD"))
    LSD_compare_path = os.path.abspath(
        os.path.join(benchmark_folder, "../../Build/compare_lines")
    )

    # Ensure parent of muLSD exists
    os.makedirs(os.path.dirname(muLSD_path), exist_ok=True)

    for i, line in enumerate(lines):
        if line.startswith("LSD ?="):
            lines[i] = f"LSD ?= {muLSD_path} {options_str}\n"
            logger.debug(f"Updated line {i} in Makefile: {lines[i].strip()}")
        if line.startswith("COMPARE ?="):
            lines[i] = f"COMPARE ?= {LSD_compare_path}\n"
            logger.debug(f"Updated line {i} in Makefile: {lines[i].strip()}")

    # Write the modified Makefile
    with open(makefile_path, "w") as f:
        f.writelines(lines)
    logger.info(f"Makefile modified for options: {options_str}")


def create_option_t_benchmarks(begin_value: int, end_value: int) -> None:
    """
    Create specialized benchmark folders for option 't' enabled and disabled.
    We call option 't' with a numeric value.
    Create benchmark folders for each value in the specified range.
    """
    for value in range(begin_value, end_value + 1):
        create_specialized_benchmark(options={"t": value})


def create_specialized_benchmark(options: set[str] | dict[str, int]) -> None:
    """
    Create a specialized benchmark folder based on the provided options.
    This is where we will run and store MuLSD results on benchmark
    """
    if type(options) is dict:
        options_string = f"t{options['t']}"
    else:
        options_string = "".join(sorted(options))
    specialized_benchmark_folder: str = str(
        Path(PATH_CURRENT_FOLDER) / Path(f"benchmark_{options_string}")
    )
    if os.path.isdir(specialized_benchmark_folder):
        logger.info(
            f"Specialized benchmark folder already exists: {specialized_benchmark_folder}"
        )
        return
    shutil.copytree(
        src=PATH_BENCHMARK_FOLDER,
        dst=specialized_benchmark_folder,
        dirs_exist_ok=True,
        symlinks=True,
    )

    modify_makefile_for_options(specialized_benchmark_folder, options)
    logger.info(f"Created specialized benchmark folder: {specialized_benchmark_folder}")
