"""
Generate a full Optuna visualization report for the scenario2_mmfit study.
"""

from pathlib import Path

import optuna
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_rank,
    plot_slice,
    plot_timeline,
)


def format_figure(fig, kind: str) -> None:
    """Apply styling tweaks so annotations remain readable."""
    layout_kwargs = {
        "margin": dict(l=70, r=40, t=70, b=70),
        "font": dict(size=12),
    }
    fig.update_layout(**layout_kwargs)

    if getattr(fig.layout, "title", None) and fig.layout.title.text:
        fig.update_layout(title=dict(text=fig.layout.title.text, x=0.5))

    if kind in {"optimization_history", "param_importances"}:
        fig.update_layout(height=500, width=850)
    elif kind in {"slice", "parallel_coordinate"}:
        fig.update_layout(height=650, width=1000)
    elif kind == "contour":
        fig.update_layout(height=900, width=1000)
    elif kind in {"edf", "rank", "timeline"}:
        fig.update_layout(height=600, width=950)


def save_plot(fig, output_path: Path, kind: str) -> None:
    """Persist a Plotly figure to disk, ensuring the parent directory exists."""
    format_figure(fig, kind)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    study_name = base_dir.name

    storage_path = base_dir / f"{study_name}.db"
    storage_url = f"sqlite:///{storage_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    output_dir = base_dir / "vis"

    save_plot(plot_optimization_history(study), output_dir / "optimization_history.html", "optimization_history")
    save_plot(plot_param_importances(study), output_dir / "param_importances.html", "param_importances")
    save_plot(plot_slice(study), output_dir / "slice.html", "slice")
    save_plot(plot_parallel_coordinate(study), output_dir / "parallel_coordinate.html", "parallel_coordinate")
    save_plot(plot_contour(study), output_dir / "contour.html", "contour")
    save_plot(plot_edf(study), output_dir / "edf.html", "edf")
    save_plot(plot_rank(study), output_dir / "rank.html", "rank")
    save_plot(plot_timeline(study), output_dir / "timeline.html", "timeline")

    if any(trial.intermediate_values for trial in study.get_trials(deepcopy=False)):
        from optuna.visualization import plot_intermediate_values

        save_plot(plot_intermediate_values(study), output_dir / "intermediate_values.html", "intermediate_values")


if __name__ == "__main__":
    main()
