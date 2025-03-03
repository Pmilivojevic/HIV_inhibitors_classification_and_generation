import pandas as pd
from hivclass import logger
from hivclass.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
    
    def validate_dataset(self) -> bool:
        try:
            # Read dataset
            data_df = pd.read_csv(self.config.local_data_file)
            
            # Validate column names
            if set(data_df.columns) != set(self.config.all_schema):
                validation_status = False
                logger.info("Columns in the dataset CSV file do not match the schema!")
            
            # Check for missing values in any column
            elif data_df.isnull().any().any():
                validation_status = False
                missing_cols = data_df.columns[data_df.isnull().any()].tolist()
                logger.info(f"The following columns contain missing values: {missing_cols}")

            else:
                validation_status = True  # Passes all checks
            
            # Write final validation status once
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            return validation_status

        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            raise
