"""План: activation maximization и эксперимент с пятью начальными шумами.

x размера [1, 3, image_size, image_size] начинается с шума в [0, 1]:
224 для ResNet18, 112 для ArcFace. Для ArcFace оптимизируется готовый вход лица,
детектор и недифференцируемое выравнивание в цикл оптимизации не включаются.
Оптимизируются только пиксели x; после каждого шага они проецируются в [0, 1].
Цель: logit(x) - tv_weight * TV(x) - l2_weight * mean((x - 0.5)**2).
TV(x) = mean(abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
      + mean(abs(x[:, :, :, 1:] - x[:, :, :, :-1])).
Замороженная модель работает в eval(), но градиент по входу включён.
Сохраняются шаг 0, каждые snapshot_every и финальный шаг; trajectory содержит
логит, sigmoid, TV, L2 и итоговую цель. Все seed используют один checkpoint.
Оценка сохранённого кадра вычисляется после проекции, чтобы совпадать с пикселями.
"""

from pathlib import Path

from .config import Config
from .contracts import SynthesisArtifacts


def synthesize(config: Config, checkpoint: Path, seed: int) -> SynthesisArtifacts:
    """Оптимизировать одно изображение и вернуть пути реальных результатов."""
    raise NotImplementedError("Градиентный синтез ещё не реализован")


def compare_seeds(config: Config, checkpoint: Path) -> list[SynthesisArtifacts]:
    """Запустить одинаковую оптимизацию для каждого generation.seeds."""
    raise NotImplementedError("Сравнение начальных шумов ещё не реализовано")
