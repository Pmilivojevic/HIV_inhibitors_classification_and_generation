from hivclass.pipelines.stage_01_data_ingestion_training_pipeline import (
    DataIngestionTrainingPipeline
)
from hivclass.pipelines.stage_02_data_validation_training_pipeline import (
    DataValidationTrainingPipeline
)
from hivclass.pipelines.stage_03_data_transformation_training_pipeline import (
    DataTransformationTrainingPipeline
)
from hivclass import logger
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import os
from typing import List, Tuple, Union

class MoleculeDataset(Dataset):
    def __init__(
        self,
        source_root,
        processed_root,
        source_filename,
        procesed_filename,
        test=False,
    ):
        self.source_root=source_root
        self.processed_root=processed_root
        self.source_filename=source_filename
        self.procesed_filename=procesed_filename
        self.test=test
        self.data_name = 'test' if self.test else 'train'
        
        super(MoleculeDataset, self).__init__()
    
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        
        return self.source_filename
    
    @property
    def raw_dir(self) -> str:
        return self.source_root
    
    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        
        return self.procesed_filename
    
    @property
    def processed_dir(self) -> str:
        
        return self.processed_root
    
    def download(self) -> None:
        STAGE_NAME = "Data Ingestion"

        try:
            logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
            data_ingestion = DataIngestionTrainingPipeline()
            data_ingestion.main()
            logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")
            
        except Exception as e:
            logger.exception(e)
            raise e
        
        STAGE_NAME = "Data Validation"

        try:
            logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
            data_validation = DataValidationTrainingPipeline()
            data_validation.main()
            logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    def process(self) -> None:
        STAGE_NAME = "Data Transformation"

        try:
            logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
            data_transformation = DataTransformationTrainingPipeline()
            data_transformation.main()
            logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    def len(self) -> int:
        # data_list = os.listdir(os.path.join(self.processed_root, self.data_name))
        data_df = pd.read_csv(os.path.join(self.processed_root, self.data_name + '.csv'))
        
        return data_df.shape[0]
    
    def get(self, idx: int) -> Data:
        data_path = os.path.join(self.processed_dir, self.data_name, f'{self.data_name}_{idx}.pt')
        
        while not os.path.exists(data_path):
            idx += 1
            data_path = os.path.join(self.processed_dir, self.data_name, f'{self.data_name}_{idx}.pt')
        
        sample = torch.load(
            data_path,
            weights_only=False
        )
        
        return sample
