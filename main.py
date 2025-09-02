import glob
import shutil
import os
import subprocess
from loguru import logger
from build_benchmark import create_specialized_benchmark

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
        subprocess.run(["make", "clean"], check=True)
        subprocess.run(["make", "-s", "-j", "2"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Benchmark failed: {e}")
    finally:
        os.chdir(original_dir)


def main():
    # 1. Setup de l'environnement de benchmark (déjà fait)
    # setup_benchmark_environment()

    # 2. Générer toutes les combinaisons d'options
    all_combinations = get_list_combination(OPTIONS)
    logger.info(f"Nombre de combinaisons: {len(all_combinations)}")

    # 3. Pour chaque combinaison
    for options in all_combinations:
        options_str = "".join(sorted(options))
        benchmark_folder = f"benchmark_{options_str}"

        # Créer le dossier spécialisé
        create_specialized_benchmark(options)

        # Exécuter le benchmark
        run_benchmark_in_folder(benchmark_folder)


if __name__ == "__main__":
    # main()
    clean_specialized_benchmark_in_current_folder()
