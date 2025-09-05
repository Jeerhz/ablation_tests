import glob
import shutil
import os
import subprocess
from loguru import logger
from build_benchmark import create_specialized_benchmark, setup_benchmark_environment  # type: ignore
import argparse

OPTIONS = {
    "n",
    "p",
    "f",
    "c",
    "s",
    "o",
    "e",
    "t",
    "w",
}


def get_list_combination(
    options_set: set[str], li_combinations: list[set[str]] = [{""}]
) -> list[set[str]]:
    """
    Recursive function to return all combinations of option set
    """
    logger.debug(
        f"Entering get_list_combination with options_set={options_set} and li_combinations={li_combinations}"
    )

    if len(options_set) == 0:
        logger.debug(f"Base case reached with li_combinations={li_combinations}")
        return li_combinations

    options_set_copy: set[str] = options_set.copy()
    option = options_set_copy.pop()
    logger.debug(f"Popped option: {option}")

    for combination in li_combinations.copy():
        combination_with_option = combination.copy()
        combination_with_option.add(option)
        logger.debug(f"Adding combination: {combination_with_option}")
        li_combinations.append(combination_with_option)

    logger.debug(
        f"Recursing with options_set_copy={options_set_copy} and li_combinations={li_combinations}"
    )
    return get_list_combination(options_set_copy, li_combinations)


def clean_specialized_benchmark_in_current_folder() -> None:
    """
    Supprime les dossiers de benchmark spécialisés dans le dossier courant.
    """
    benchmark_folders = glob.glob("benchmark_*")
    for folder in benchmark_folders:
        shutil.rmtree(folder, ignore_errors=True)
    logger.info("Cleaned specialized benchmark folders in current directory.")


def run_benchmark_in_folder(benchmark_folder: str) -> None:
    """
    Exécute le benchmark dans le dossier spécifié.
    """
    if not os.path.isdir(benchmark_folder):
        logger.error(f"Benchmark folder does not exist: {benchmark_folder}")
        return

    # Se placer dans le dossier et lancer make
    original_dir = os.getcwd()
    try:
        os.chdir(benchmark_folder)
        logger.info(f"Running benchmark in {benchmark_folder}")
        subprocess.run(["make"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Benchmark failed: {e}")
    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(description="Benchmark ablation tests")
    parser.add_argument(
        "--setup", action="store_true", help="Set up benchmark environment"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean specialized benchmark folders",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Create and run all specialized benchmarks",
    )
    args = parser.parse_args()

    # If no args are provided, run all
    if not any(vars(args).values()):
        args.setup = True
        args.clean = True
        args.run = True

    if args.setup:
        setup_benchmark_environment()
        logger.info("Benchmark environment set up.")

    if args.clean:
        clean_specialized_benchmark_in_current_folder()

    if args.run:
        all_combinations = get_list_combination(OPTIONS)
        logger.info(f"Number of combinations: {len(all_combinations)}")
        for options in all_combinations:
            options_str = "".join(sorted(options))
            benchmark_folder = f"benchmark_{options_str}"
            create_specialized_benchmark(options)
            run_benchmark_in_folder(benchmark_folder)


if __name__ == "__main__":
    main()
