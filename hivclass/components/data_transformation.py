import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
from rdkit import Chem
import torch
from torch_geometric.data import Data
import deepchem as dc
from hivclass.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def test_split_train_balanced(self):
        df = pd.read_csv(self.config.data_csv)
        
        # Separate positive and negative cases
        p_val = df[df.HIV_active == 1].to_numpy()
        n_val = df[df.HIV_active == 0].to_numpy()

        # Ensure class balance by selecting the smaller group as the target
        if len(p_val) >= len(n_val):
            big, small = p_val, n_val
        else:
            big, small = n_val, p_val

        # Stratified test split
        small_train, small_test = train_test_split(
            small,
            test_size=self.config.params,
            random_state=42
        )
        
        big_train, big_test = train_test_split(
            big,
            test_size=(self.config.params * len(small) / len(big)),
            random_state=42
        )

        test = np.concatenate([small_test, big_test])
        
        # Ensure the train set remains balanced by oversampling the smaller class
        train = np.concatenate([
            big_train,
            random.choices(small_train, k=len(big_train) - len(small_train))
        ])

        # Convert back to DataFrame
        train_df = pd.DataFrame(train, columns=df.columns).sample(frac=1, random_state=42)
        train_df.reset_index(drop=True, inplace=True)
        train_df.to_csv(self.config.train_csv, index=False)
        
        test_df = pd.DataFrame(test, columns=df.columns).sample(frac=1, random_state=42)
        test_df.reset_index(drop=True, inplace=True)
        test_df.to_csv(self.config.test_csv, index=False)
    
    def data_preparation(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        datas = [self.config.train_csv, self.config.test_csv]
        save_dirs = [self.config.train_folder, self.config.test_folder]
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

        all_deepchem_data = []

        for data, save_dir in zip(datas, save_dirs):
            data_df = pd.read_csv(data)
            name = os.path.splitext(os.path.basename(data))[0]  # Extract filename without extension

            for mol in tqdm(data_df.itertuples(index=True), total=len(data_df), desc=f"Processing {name}"):

                mol_obj = Chem.MolFromSmiles(mol.smiles)
                if mol_obj is None:
                    continue  # Skip invalid SMILES strings
                
                label = torch.tensor(mol.HIV_active, dtype=torch.int64, device=device)
                
                graph_features = featurizer._featurize(mol_obj)

                data_deepchem = Data(
                    x=torch.tensor(graph_features.node_features, dtype=torch.float, device=device),
                    edge_attr=torch.tensor(graph_features.edge_features, dtype=torch.float, device=device),
                    edge_index=torch.tensor(graph_features.edge_index, dtype=torch.long, device=device),
                    y=label, smiles=mol.smiles
                )
                all_deepchem_data.append(data_deepchem)

                # Save preprocessed molecule graph
                torch.save(data_deepchem, os.path.join(save_dir, f"{name}_{mol[0]}.pt"))
    
    def transformation_compose(self):
        self.test_split_train_balanced()
        self.data_preparation()
