from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "ablations"
    / "dgr_hmdb"
    / "dgr_hmdb_ablations"
    / "experiment_manifest.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "dgr_ablations"
ROUTES = (
    ("ccsbase_to_ccsbase", "C→C"),
    ("ccsbase_to_metlinccs", "C→M"),
    ("metlinccs_to_ccsbase", "M→C"),
    ("metlinccs_to_metlinccs", "M→M"),
)


@dataclass(frozen=True)
class Configuration:
    name: str
    label: str
    result_paths: dict[str, Path]
    reference: bool = False


def resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = manifest_path.parent / path
    return candidate if candidate.exists() else PROJECT_ROOT / path


def load_configurations(manifest_path: Path) -> tuple[Configuration, ...]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Ablation manifest not found: {manifest_path}. Run the DGR ablation suite first."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("state") != "complete":
        raise ValueError(
            f"Ablation manifest is not complete (state={payload.get('state')!r}): {manifest_path}"
        )
    if payload.get("model_type") != "gated_residual_mlp":
        raise ValueError("The figure accepts only DGR-MLP ablation manifests.")
    if payload.get("fingerprint_source") != "rdkit":
        raise ValueError("The article ablation figure is defined for RDKit fingerprints.")
    if payload.get("folds") != 5:
        raise ValueError("The article ablation figure requires five downstream folds.")

    reference = payload["reference"]
    configurations = [
        Configuration(
            name="all_pretraining_tasks",
            label=reference["label"],
            result_paths={
                route_name: resolve_path(path, manifest_path)
                for route_name, path in reference["results"].items()
            },
            reference=True,
        )
    ]
    for item in payload["ablations"]:
        if item.get("model_type") != "gated_residual_mlp":
            raise ValueError(f"Ablation {item.get('name')} does not use DGR-MLP.")
        configurations.append(
            Configuration(
                name=item["name"],
                label=item["label"],
                result_paths={
                    route_name: resolve_path(path, manifest_path)
                    for route_name, path in item["results"].items()
                },
            )
        )
    if len(configurations) != 10:
        raise ValueError(
            f"Expected the reference and nine ablations; found {len(configurations)} configurations."
        )
    expected_routes = {route_name for route_name, _ in ROUTES}
    for configuration in configurations:
        if set(configuration.result_paths) != expected_routes:
            raise ValueError(
                f"Configuration {configuration.name} does not contain the four expected routes."
            )
    return tuple(configurations)


def fold_mae(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        values = [float(row["MAE"]) for row in rows if row["Fold"].strip().lower() != "total"]
    if len(values) != 5:
        raise RuntimeError(f"Expected five folds in {path}; found {len(values)}")
    return np.asarray(values, dtype=float)


def significance_stars(p_value: float) -> str:
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def comparison_matrix(
    configurations: tuple[Configuration, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference = configurations[0]
    reference_folds = {
        route_name: fold_mae(reference.result_paths[route_name]) for route_name, _ in ROUTES
    }
    means = np.zeros((len(configurations), len(ROUTES)), dtype=float)
    standard_deviations = np.zeros_like(means)
    deltas = np.zeros_like(means)
    p_values = np.full_like(means, np.nan)

    for row_index, configuration in enumerate(configurations):
        for column_index, (route_name, _) in enumerate(ROUTES):
            reference_values = reference_folds[route_name]
            candidate = fold_mae(configuration.result_paths[route_name])
            means[row_index, column_index] = float(candidate.mean())
            standard_deviations[row_index, column_index] = float(candidate.std())
            deltas[row_index, column_index] = float(candidate.mean() - reference_values.mean())
            if not configuration.reference:
                p_values[row_index, column_index] = float(
                    ttest_rel(candidate, reference_values).pvalue
                )
    return means, standard_deviations, deltas, p_values


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def draw_heatmap(
    ax: plt.Axes,
    configurations: tuple[Configuration, ...],
    deltas: np.ndarray,
    p_values: np.ndarray,
    color_limit: float = 0.40,
) -> mpl.image.AxesImage:
    image = ax.imshow(
        np.clip(deltas, -color_limit, color_limit),
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title("Ablation tests", loc="center", pad=24, fontsize=20, fontweight="bold")
    ax.set_yticks(np.arange(len(configurations)), [item.label for item in configurations])
    ax.set_xticks(np.arange(len(ROUTES)), [route_label for _, route_label in ROUTES])
    ax.set_xticks(np.arange(-0.5, len(ROUTES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(configurations), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_index, configuration in enumerate(configurations):
        for column_index in range(len(ROUTES)):
            value = deltas[row_index, column_index]
            stars = "" if configuration.reference else significance_stars(
                p_values[row_index, column_index]
            )
            label = "REF" if configuration.reference else f"{value:+.2f}{stars}"
            clipped_value = float(np.clip(value, -color_limit, color_limit))
            text_color = "white" if abs(clipped_value) >= color_limit * 0.58 else "#202020"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9.2,
                fontweight="bold" if stars or configuration.reference else "normal",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def save_figure(fig: plt.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"ablation_tests.{suffix}", bbox_inches="tight", facecolor="white")


def write_statistics(
    output_dir: Path,
    configurations: tuple[Configuration, ...],
    means: np.ndarray,
    standard_deviations: np.ndarray,
    deltas: np.ndarray,
    p_values: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_statistics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["configuration", "route", "mean_mae", "std_mae", "delta_mae", "p_value", "stars"]
        )
        for row_index, configuration in enumerate(configurations):
            for column_index, (route_name, _) in enumerate(ROUTES):
                p_value = p_values[row_index, column_index]
                writer.writerow(
                    [
                        configuration.label,
                        route_name,
                        f"{means[row_index, column_index]:.6f}",
                        f"{standard_deviations[row_index, column_index]:.6f}",
                        f"{deltas[row_index, column_index]:.6f}",
                        "" if np.isnan(p_value) else f"{p_value:.8g}",
                        "" if np.isnan(p_value) else significance_stars(float(p_value)),
                    ]
                )

    latex_lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Configuration & C$\rightarrow$C & C$\rightarrow$M & M$\rightarrow$C & M$\rightarrow$M \\",
        r"\midrule",
    ]
    for row_index, configuration in enumerate(configurations):
        cells = []
        for column_index in range(len(ROUTES)):
            stars = "" if configuration.reference else significance_stars(
                float(p_values[row_index, column_index])
            )
            cells.append(
                f"{means[row_index, column_index]:.2f} $\\pm$ "
                f"{standard_deviations[row_index, column_index]:.2f}{stars}"
            )
        label = configuration.label.replace("&", r"\&")
        latex_lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (output_dir / "ablation_table.tex").write_text(
        "\n".join(latex_lines), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the RDKit DGR-MLP ablation figure and statistical tables."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configurations = load_configurations(args.manifest)
    means, standard_deviations, deltas, p_values = comparison_matrix(configurations)
    configure_style()
    fig, ax = plt.subplots(figsize=(10.2, 9.5), constrained_layout=True)
    image = draw_heatmap(ax, configurations, deltas, p_values)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.025)
    colorbar.set_label(
        "ΔMAE vs All pretraining tasks (Å²)\nlower is better",
        rotation=270,
        labelpad=34,
    )
    save_figure(fig, args.output_dir)
    plt.close(fig)
    write_statistics(
        args.output_dir,
        configurations,
        means,
        standard_deviations,
        deltas,
        p_values,
    )
    print(f"Wrote DGR-MLP ablation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
