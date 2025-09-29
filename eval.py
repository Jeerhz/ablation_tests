# Open all benchmark_ folders
from collections import defaultdict
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from config import PATH_CURRENT_FOLDER  # type: ignore
from loguru import logger

from models import BenchmarkScore, Scores, ImageScores  # type: ignore
from scipy.stats import wilcoxon


def extract_scores_from_benchmarks(
    path_containing_benchmarks: str = PATH_CURRENT_FOLDER,
) -> list[BenchmarkScore]:
    """Extract scores from benchmark folders."""

    logger.info(f"Extracting scores from benchmarks in {path_containing_benchmarks}")

    benchmark_folders = glob.glob(
        os.path.join(path_containing_benchmarks, "benchmark_*")
    )

    all_benchmark_scores: list[BenchmarkScore] = []

    for folder in benchmark_folders:
        options: str = os.path.basename(folder).replace("benchmark_", "")
        benchmark_score_files = glob.glob(os.path.join(folder, "*_eval.txt"))
        nb_benchmark_images = len(benchmark_score_files)

        logger.info(f"Found {nb_benchmark_images} benchmark score files in {folder}")
        benchmark_score: BenchmarkScore = BenchmarkScore(options=options, scores=[])

        for score_file in benchmark_score_files:
            image_name = os.path.basename(score_file).replace("_eval.txt", "")
            with open(score_file, "r") as f:
                content = f.read().strip()
                # Example line: "Prec,Rec,F1,IOU= 0.866506 0.358031 0.506699 0.506122"
                if "=" in content:
                    metrics_str = content.split("=")[1].strip()
                    metrics = metrics_str.split()
                    # logger.debug(
                    #     f"Extracted metrics for Options:{options} - Image:{image_name}: {metrics}"
                    # )
                    benchmark_score.scores.append(
                        ImageScores(
                            image_name=image_name,
                            precision=float(metrics[0]),
                            recall=float(metrics[1]),
                            f1_score=float(metrics[2]),
                            iou=float(metrics[3]),
                        )
                    )
                else:
                    logger.warning(f"Unexpected format in {score_file}: {content}")
                    benchmark_score.scores.append(
                        ImageScores(
                            image_name=image_name,
                            precision=None,
                            recall=None,
                            f1_score=None,
                            iou=None,
                        )
                    )
        all_benchmark_scores.append(
            BenchmarkScore(options=options, scores=benchmark_score.scores)
        )

    if len(all_benchmark_scores) == 0:
        logger.error(
            "No benchmark scores found. Are you sure you are in the folder containing benchmark_* folders?"
        )
        return []
    logger.success(f"All benchmark extracted, first one: {all_benchmark_scores[0]}")
    return all_benchmark_scores


def get_benchmark_score_from_options(
    options: str, benchmarks: list[BenchmarkScore]
) -> BenchmarkScore | None:
    """Retrieve the average Scores for a given options string from the benchmarks list. Return None if not found or if there are several matches."""
    matching_benchmarks = [bm for bm in benchmarks if bm.options == options]
    if not matching_benchmarks:
        return None
    if len(matching_benchmarks) > 1:
        logger.warning(
            f"Multiple benchmarks found for options '{options}': {matching_benchmarks}"
        )
        return None
    return matching_benchmarks[0]


def compare_benchmarks(
    options1: str, options2: str, benchmarks: list[BenchmarkScore]
) -> None | list[ImageScores]:
    """Compare two benchmarks given their options strings.
    Return the difference in absolute value for each image present in both benchmarks and for each metric."""
    logger.info(f"Comparing benchmarks '{options1}' and '{options2}'")
    bm1 = get_benchmark_score_from_options(options1, benchmarks)
    bm2 = get_benchmark_score_from_options(options2, benchmarks)

    if bm1 is None:
        logger.error(f"No benchmark found for options '{options1}'")
        return None
    if bm2 is None:
        logger.error(f"No benchmark found for options '{options2}'")
        return None

    # Return the scores difference for each images present in both benchmarks
    scores_diff: list[ImageScores] = []
    images_bm1 = {s.image_name: s for s in bm1.scores}
    images_bm2 = {s.image_name: s for s in bm2.scores}
    common_images = set(images_bm1.keys()).intersection(set(images_bm2.keys()))
    for image_name in common_images:
        score1 = images_bm1[image_name]
        score2 = images_bm2[image_name]
        scores_diff.append(
            ImageScores(
                image_name=image_name,
                precision=(
                    abs(score2.precision - score1.precision)
                    if score1.precision is not None and score2.precision is not None
                    else None
                ),
                recall=(
                    abs(score2.recall - score1.recall)
                    if score1.recall is not None and score2.recall is not None
                    else None
                ),
                f1_score=(
                    abs(score2.f1_score - score1.f1_score)
                    if score1.f1_score is not None and score2.f1_score is not None
                    else None
                ),
                iou=(
                    abs(score2.iou - score1.iou)
                    if score1.iou is not None and score2.iou is not None
                    else None
                ),
            )
        )
    logger.info(
        f"Compared benchmarks '{options1}' and '{options2}' on {len(common_images)} common images."
    )
    return scores_diff


def test_wilcoxon(
    option: str, benchmarks: list[BenchmarkScore]
) -> dict[str, float] | None:
    """Perform Wilcoxon signed-rank test between the benchmark with the given option and the benchmark without any options.
    Returns a dict of {metric: p_value} for each metric, or None if not enough data.
    """
    bm_with = get_benchmark_score_from_options(option, benchmarks)
    bm_without = get_benchmark_score_from_options("", benchmarks)

    if bm_with is None or bm_without is None:
        logger.error(f"Could not find benchmarks for option '{option}' or no options.")
        return None

    # Match scores by image name
    scores_with = {s.image_name: s for s in bm_with.scores}
    scores_without = {s.image_name: s for s in bm_without.scores}

    logger.debug(f"Scores with option images: {list(scores_with.keys())}")
    logger.debug(f"Scores without option images: {list(scores_without.keys())}")

    common_images = set(scores_with.keys()).intersection(set(scores_without.keys()))

    if not common_images:
        logger.error(
            f"No common images found between option '{option}' and no options."
        )
        return None

    metrics = ["precision", "recall", "f1_score", "iou"]
    p_values: dict[str, float] = {}
    for metric in metrics:
        data_with = []
        data_without = []
        for image_name in common_images:
            score_with = getattr(scores_with[image_name], metric)
            score_without = getattr(scores_without[image_name], metric)
            if score_with is not None and score_without is not None:
                data_with.append(score_with)
                data_without.append(score_without)

        if len(data_with) < 10:
            logger.warning(
                f"Not enough data points for Wilcoxon test on metric '{metric}' with option '{option}'."
            )
            continue

        stat, p_value = wilcoxon(data_with, data_without)
        logger.info(
            f"Wilcoxon test for metric '{metric}' with option '{option}': statistic={stat}, p-value={p_value}"
        )
        p_values[metric] = float(p_value)

    return p_values if p_values else None


def get_top3_diff_images(
    diff_scores: list[ImageScores],
) -> dict[str, list[tuple[str, float]]]:
    """
    Returns the top 3 images with the biggest absolute difference for each metric.
    Returns a dict: {metric: [(image_name, diff_value), ...]}
    """
    metrics = ["precision", "recall", "f1_score", "iou"]
    top_images = {}
    for metric in metrics:
        filtered = [s for s in diff_scores if getattr(s, metric) is not None]
        sorted_scores = sorted(
            filtered, key=lambda s: getattr(s, metric), reverse=True
        )[:3]
        top_images[metric] = [(s.image_name, getattr(s, metric)) for s in sorted_scores]
    return top_images


def plot_benchmark_ranges(benchmarks: list[BenchmarkScore]) -> None:
    """
    Plot min-max ranges for precision, recall, f1, and IoU across benchmarks.
    Highlights 'muLSD', 'no options', and 'fno' in red, and displays the options for min/max values.
    """
    metrics = ["precision", "recall", "f1_score", "iou"]
    metric_labels = {
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-Score",
        "iou": "IoU",
    }
    colors = {
        "min_max": "#2c3e50",  # Dark blue-gray for min/max line
        "muLSD": "#e74c3c",  # Red for muLSD/no options/fno
        "text": "#34495e",  # Dark gray for text
        "grid": "#ecf0f1",  # Light gray for grid
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), sharey=True)
    fig.patch.set_facecolor("white")  # Set background to white
    plt.subplots_adjust(wspace=0.3, left=0.05, right=0.95, top=0.9, bottom=0.3)

    # Collect average scores
    avg_scores_dict = {bm.options: bm.average_scores() for bm in benchmarks}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = {
            opt: getattr(s, metric)
            for opt, s in avg_scores_dict.items()
            if getattr(s, metric) is not None
        }
        if not values:
            ax.set_title(
                metric_labels[metric],
                fontsize=12,
                fontweight="bold",
                color=colors["text"],
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Find min and max options
        min_opt = min(values, key=lambda k: values[k])
        max_opt = max(values, key=lambda k: values[k])
        min_val, max_val = values[min_opt], values[max_opt]

        # Plot min-max line
        ax.plot(
            [min_val, max_val],
            [0, 0],
            color=colors["min_max"],
            lw=2.5,
            linestyle="-",
            alpha=0.8,
            label="Range",
        )

        # Add min and max markers with option labels
        ax.scatter(
            [min_val],
            [0],
            color=colors["min_max"],
            s=100,
            edgecolor="white",
            linewidth=1,
            zorder=3,
            label="Min",
        )
        ax.scatter(
            [max_val],
            [0],
            color=colors["min_max"],
            s=100,
            edgecolor="white",
            linewidth=1,
            zorder=3,
            label="Max",
        )

        # Annotate min and max values with options
        ax.annotate(
            f"{min_val:.3f}\n({min_opt})",
            xy=(min_val, 0),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=colors["text"],
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                alpha=0.8,
                edgecolor=colors["grid"],
            ),
        )
        ax.annotate(
            f"{max_val:.3f}\n({max_opt})",
            xy=(max_val, 0),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=colors["text"],
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                alpha=0.8,
                edgecolor=colors["grid"],
            ),
        )

        # Highlight muLSD, no options, and cfno with different colors
        highlighted_options = [
            ("muLSD", "#e74c3c"),  # Red
            ("", "#e74c3c"),
            ("cfnow", "#2980b9"),  # Blue for cfno
        ]
        for opt, color in highlighted_options:
            if opt in values:
                val = values[opt]
                ax.scatter(
                    [val],
                    [0],
                    color=color,
                    s=120,
                    edgecolor="white",
                    linewidth=1,
                    zorder=4,
                    label=opt if opt else "MuLSD (no options)",
                )
                ax.annotate(
                    f"{val:.3f}\n({opt})" if (opt and opt != "cfnow") else f"{val:.3f}",
                    xy=(val, 0),
                    xytext=(0, -30),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        alpha=0.8,
                        edgecolor=color,
                    ),
                )

        # Style the axis
        ax.set_title(
            metric_labels[metric],
            fontsize=13,
            fontweight="bold",
            color=colors["text"],
            pad=10,
        )
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(colors["grid"])
        ax.spines["bottom"].set_linewidth(0.8)

        # Add grid and minor ticks for precision
        ax.grid(axis="x", color=colors["grid"], linestyle="--", alpha=0.6, which="both")
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(
            axis="x",
            which="both",
            direction="out",
            top=False,
            labelcolor=colors["text"],
            labelsize=9,
        )

        # Add legend
        if idx == 0:
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.35),
                ncol=3,
                fontsize=9,
                frameon=False,
            )

    # Save the figure
    out_file = "ranges_scores.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.success(f"Saved benchmark ranges figure (with options) to {out_file}")


def analyze_option_effects(benchmarks: list[BenchmarkScore]) -> None:
    """
    For each option, compute the mean and std deviation for the four metrics
    with and without the option. Save figures for each metric.
    """
    # Collect all unique options (letters) from all benchmarks
    unique_options_set = set[str]()
    for bm in benchmarks:
        unique_options_set.update(bm.options)
    unique_options_list = sorted(unique_options_set)
    logger.info(f"Unique options found: {unique_options_list}")

    # For each option, separate benchmarks into "with" and "without"
    metrics = ["precision", "recall", "f1_score", "iou"]
    results: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    mulsd_scores: Scores | None = None
    for bm in benchmarks:
        if bm.options == "muLSD" or bm.options == "":
            mulsd_scores = bm.average_scores()
            break

    for option in unique_options_list:
        for bm in benchmarks:
            avg_scores = bm.average_scores()
            has_option = option in bm.options
            for metric in metrics:
                value = getattr(avg_scores, metric)
                if value is not None:
                    results[option][f"{metric}_with"].append(
                        value
                    ) if has_option else results[option][f"{metric}_without"].append(
                        value
                    )

    # Compute mean and std for each option and metric
    summary: dict[str, dict[str, float | None]] = {}
    for option in unique_options_list:
        summary[option] = {}
        for metric in metrics:
            with_values = results[option].get(f"{metric}_with", [])
            without_values = results[option].get(f"{metric}_without", [])
            summary[option][f"{metric}_with_mean"] = (
                float(np.mean(with_values)) if with_values else None
            )
            summary[option][f"{metric}_with_std"] = (
                float(np.std(with_values)) if with_values else None
            )
            summary[option][f"{metric}_without_mean"] = (
                float(np.mean(without_values)) if without_values else None
            )
            summary[option][f"{metric}_without_std"] = (
                float(np.std(without_values)) if without_values else None
            )

    # Log results
    for option, data in summary.items():
        logger.info(f"\nOption: {option}")
        for metric in metrics:
            logger.info(
                f"  {metric}: "
                f"with={data[f'{metric}_with_mean']:.3f}±{data[f'{metric}_with_std']:.3f} "
                f"without={data[f'{metric}_without_mean']:.3f}±{data[f'{metric}_without_std']:.3f}"
            )

    # Plot results
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12), sharex=True)
    # Find MLSD: benchmark with all options activated
    all_options_str = "".join(unique_options_list)
    mlsd_scores: Scores | None = None
    for bm in benchmarks:
        if bm.options == all_options_str:
            mlsd_scores = bm.average_scores()
            break

    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(unique_options_list))
        width = 0.35
        for j, option in enumerate(unique_options_list):
            with_mean = summary[option][f"{metric}_with_mean"]
            with_std = summary[option][f"{metric}_with_std"]
            without_mean = summary[option][f"{metric}_without_mean"]
            without_std = summary[option][f"{metric}_without_std"]
            if with_mean is not None:
                ax.bar(
                    j - width / 2,
                    with_mean,
                    width,
                    yerr=with_std,
                    capsize=5,
                    label="With option" if j == 0 else None,
                    color="blue",
                )
            if without_mean is not None:
                ax.bar(
                    j + width / 2,
                    without_mean,
                    width,
                    yerr=without_std,
                    capsize=5,
                    label="Without option" if j == 0 else None,
                    color="orange",
                )
        # Plot horizontal line for muLSD score
        if mulsd_scores is not None:
            mulsd_val = getattr(mulsd_scores, metric)
            if mulsd_val is not None:
                ax.axhline(
                    mulsd_val,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="muLSD",
                )
        # Plot horizontal line for MLSD score (all options activated)
        if mlsd_scores is not None:
            mlsd_val = getattr(mlsd_scores, metric)
            if mlsd_val is not None:
                ax.axhline(
                    mlsd_val,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label="MLSD (all options)",
                )

        ax.set_title(metric.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(unique_options_list)
        ax.legend()
    plt.tight_layout()
    out_file = "option_effects.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    logger.success(f"Saved option effects figure to {out_file}")


def plot_wilcoxon_results(
    benchmarks: list[BenchmarkScore], output_file: str = "wilcoxon_pvalues.png"
) -> None:
    """
    For each option, perform Wilcoxon test and plot p-values for each metric.
    """
    unique_options_set = set[str]()
    for bm in benchmarks:
        unique_options_set.update(bm.options)
    unique_options_list = sorted(unique_options_set)

    metrics = ["precision", "recall", "f1_score", "iou"]
    p_values_dict: dict[str, dict[str, float]] = {metric: {} for metric in metrics}

    for option in unique_options_list:
        p_values = test_wilcoxon(option, benchmarks)
        if p_values:
            for metric in metrics:
                if metric in p_values:
                    p_values_dict[metric][option] = p_values[metric]

    # Plotting
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12), sharex=True)
    for i, metric in enumerate(metrics):
        ax = axes[i]
        options = list(p_values_dict[metric].keys())
        p_values_list = [p_values_dict[metric][opt] for opt in options]
        ax.bar(options, p_values_list, color="skyblue")
        ax.axhline(0.05, color="red", linestyle="--", label="Significance level (0.05)")
        ax.set_title(f"Wilcoxon p-values for {metric.capitalize()}")
        ax.set_ylabel("p-value")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis="y")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.success(f"Saved Wilcoxon p-values figure to {output_file}")


if __name__ == "__main__":
    all_benchmark_scores = extract_scores_from_benchmarks()
    logger.info(f"Number of average scores: {len(all_benchmark_scores)}")
    plot_benchmark_ranges(all_benchmark_scores)
    analyze_option_effects(all_benchmark_scores)
    plot_wilcoxon_results(all_benchmark_scores)

    # Compare max and min benchmarks for 4 metrics
    diff_scores = compare_benchmarks(
        options1="", options2="cenop", benchmarks=all_benchmark_scores
    )
    if diff_scores is None:
        logger.error("Could not compare benchmarks.")

    else:
        top3_diffs = get_top3_diff_images(diff_scores)
        for metric, top_images in top3_diffs.items():
            logger.info(f"Top 3 images with biggest {metric} difference:")
            for image_name, diff_value in top_images:
                logger.info(f"  {image_name}: {diff_value:.4f}")
