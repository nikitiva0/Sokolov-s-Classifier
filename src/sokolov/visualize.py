"""План: визуализации промежуточных шагов, оценок и сравнения seed.

Показывать все seed, включая неудачные, с общей шкалой оценок. Высокая оценка
синтетического узора не доказывает узнаваемость или качество классификации.
"""

from pathlib import Path

from .contracts import SynthesisArtifacts


def render_synthesis_report(runs: list[SynthesisArtifacts], output_dir: Path) -> Path:
    """Собрать сетки шагов, траектории логита/sigmoid и сравнение финалов."""
    raise NotImplementedError("Визуализация ещё не реализована")
