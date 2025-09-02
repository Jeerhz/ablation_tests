from loguru import logger
from build_benchmark import setup_benchmark_environment, create_specialized_benchmark

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


def main():
    logger.info("Starting benchmark environment setup...")
    setup_benchmark_environment()
    create_specialized_benchmark({"n", "p", "f"})


if __name__ == "__main__":
    all_combinations: list[set[str]] = get_list_combination(options_set=OPTIONS)
    if len(all_combinations) != 2 ** len(OPTIONS):
        logger.error(
            f"Error in get_list_combination: expected {2 ** len(OPTIONS)} combinations, got {len(all_combinations)}"
        )
    logger.success(f"Nombre de combinaisons: {len(all_combinations)}")
