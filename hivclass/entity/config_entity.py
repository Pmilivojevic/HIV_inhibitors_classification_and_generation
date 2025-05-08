from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    local_data_file: Path
    STATUS_FILE: Path
    all_schema: list


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_csv: Path
    train_csv: Path
    test_csv: Path
    train_folder: Path
    test_folder: Path
    params: dict
    dataset_val_status: bool


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    models: Path
    stats: Path
    source_root: Path
    processed_root: Path
    source_filename: str
    processed_filename: List[str]
    tuning: bool
    params: dict


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_folder_path: Path
    source_root: Path
    processed_root: Path
    source_filename: Path
    processed_filename: List[str]
    tuning: bool
    params: dict
