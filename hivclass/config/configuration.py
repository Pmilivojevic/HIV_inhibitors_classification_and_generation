from hivclass.constants import *
from hivclass.utils.main_utils import read_yaml, create_directories
from hivclass.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

class ConfigurationManager:
    def __init__(
        self,
        config_file_path = CONFIG_FILE_PATH,
        params_file_path = PARAMS_FILE_PATH,
        schema_file_path = SCHEMA_FILE_PATH
    ):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)
        
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file
        )
        
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            local_data_file=config.local_data_file,
            STATUS_FILE=config.STATUS_FILE,
            all_schema=schema
        )
        
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.data_transformation
        dataset_val_status_file = self.config.data_validation.STATUS_FILE
        
        with open(dataset_val_status_file, 'r') as f:
            status = f.read()
        
        status = bool(str.split(status)[-1])
        
        create_directories([config.root_dir, config.train_folder, config.test_folder])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_csv=config.data_csv,
            train_csv=config.train_csv,
            test_csv=config.test_csv,
            train_folder=config.train_folder,
            test_folder=config.test_folder,
            params=params.split_size,
            dataset_val_status=status
        )
        
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_parameters
        
        create_directories([config.root_dir, config.models, config.stats])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            models=config.models,
            stats=config.stats,
            source_root=config.source_root,
            processed_root=config.processed_root,
            source_filename=config.source_filename,
            processed_filename=config.processed_filename,
            tuning=config.tuning,
            model_params=params
        )
        
        return model_trainer_config
