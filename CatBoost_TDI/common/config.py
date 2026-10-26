"""Explicit band order and the paper/notebook CatBoost settings."""

from dataclasses import asdict, dataclass

LEGACY_BANDS = (
    'LST', 'NDVI', 'NDWI', 'NDBI', 'Albedo', 'Emissivity', 'LC', 'EULUC',
    'DEM', 'TH', 'BD', 'BH', 'SVF', 'AH', 'VNL', 'LCZ', 'TDI',
)
ALL_FEATURES = LEGACY_BANDS[1:]
# Preserve the feature order in notebook cells 25, 40, 44 and 61.
KEY_FEATURES = ('NDVI', 'NDBI', 'Albedo', 'EULUC', 'DEM', 'AH', 'VNL', 'LCZ', 'TDI')
PAPER_RANKING = (
    'TDI', 'VNL', 'DEM', 'AH', 'LCZ', 'EULUC', 'NDVI', 'NDBI', 'Albedo',
    'BH', 'NDWI', 'BD', 'Emissivity', 'SVF', 'TH', 'LC',
)
INVALID_THRESHOLD = -200.0


@dataclass(frozen=True)
class ModelConfig:
    iterations: int = 8000
    learning_rate: float = 0.08
    depth: int = 10
    random_seed: int = 0
    split_seed: int = 42
    validation_fraction: float = 0.2
    early_stopping_rounds: int = 100
    thread_count: int = -1
    # The notebooks never passed cat_features. Empty keeps their numeric encoding.
    categorical_features: tuple[str, ...] = ()

    def __post_init__(self):
        if self.iterations < 1 or not 1 <= self.depth <= 16:
            raise ValueError('iterations must be positive; depth must be in [1, 16].')
        if not 0 < self.learning_rate <= 1 or not 0 < self.validation_fraction < 1:
            raise ValueError('Invalid learning_rate or validation_fraction.')
        if self.early_stopping_rounds < 1 or self.thread_count == 0:
            raise ValueError('Invalid early_stopping_rounds or thread_count.')
        object.__setattr__(self, 'categorical_features', tuple(self.categorical_features))

    def as_dict(self):
        return asdict(self)

    def catboost_params(self, iterations=None):
        return dict(
            iterations=self.iterations if iterations is None else iterations,
            learning_rate=self.learning_rate, depth=self.depth,
            loss_function='RMSE', eval_metric='RMSE', random_seed=self.random_seed,
            verbose=False, allow_writing_files=False, thread_count=self.thread_count,
        )
