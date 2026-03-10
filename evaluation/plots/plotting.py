from __future__ import annotations

import textwrap
from pathlib import Path

from plotnine import (
    element_blank,
    element_line,
    element_rect,
    element_text,
    theme,
    theme_minimal,
)


def publication_theme():
    return (
        theme_minimal()
        + theme(
            figure_size=(10, 6),
            panel_grid_major_x=element_blank(),
            panel_grid_minor=element_blank(),
            panel_grid_major_y=element_line(color="#d9d9d9", size=0.35),
            axis_text=element_text(color="#1f2937", size=10),
            axis_title=element_text(color="#1f2937", size=11, weight="bold"),
            plot_title=element_text(color="#111827", size=15, weight="bold"),
            plot_subtitle=element_text(color="#374151", size=10),
            plot_caption=element_text(color="#6b7280", size=8),
            legend_title=element_text(color="#1f2937", size=10, weight="bold"),
            legend_text=element_text(color="#1f2937", size=9),
            strip_background=element_rect(fill="#f3f4f6", color="#d1d5db"),
            strip_text=element_text(color="#111827", weight="bold"),
        )
    )


def save_plot(plot, output_dir: Path, filename: str, width: float = 10, height: float = 6, dpi: int = 300) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    plot.save(path, width=width, height=height, dpi=dpi, verbose=False)
    return path


def wrap_label(text: str, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(str(text).replace("_", " "), width=width))
