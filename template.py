import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "hivclass"

list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/configuration.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/pipelines/__init__.py",
    f"{project_name}/pipelines/stage_01_data_ingestion_training_pipeline.py",
    f"{project_name}/pipelines/stage_02_data_validation_training_pipeline.py",
    f"{project_name}/pipelines/stage_03_data_transformation_training_pipeline.py",
    f"{project_name}/pipelines/stage_04_model_trainer_training_pipeline.py",
    f"{project_name}/pipelines/stage_05_model_evaluation_training_pipeline.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "research/trials.ipynb",
    "research/01_data_ingestion.ipynb",
    "research/02_data_validation.ipynb",
    "research/03_data_transformation.ipynb",
    "research/04_model_trainer.ipynb",
    "research/05_model_evaluation.ipynb",
    "main.py",
    "requirements.txt",
    "setup.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filename)) or (os.path.getsize(filename) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file {filename}")
    else:
        logging.info(f"{filename} is already created")
