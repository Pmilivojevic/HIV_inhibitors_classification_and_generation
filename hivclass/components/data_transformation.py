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
        if self.config.dataset_val_status:
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
            # train_name_column = [f'train_{i+1}' for i in train_df.index]
            # train_df.insert(0, 'name', train_name_column)
            train_df.insert(0, 'name', 'train')
            # train_df.to_csv(self.config.train_csv, index=False)
            
            test_df = pd.DataFrame(test, columns=df.columns).sample(frac=1, random_state=42)
            test_df.reset_index(drop=True, inplace=True)
            # test_name_column = [f'test{i+1}' for i in train_df.index]
            # test_df.insert(0, 'name', test_name_column)
            test_df.insert(0, 'name', 'test')
            # test_df.to_csv(self.config.test_csv, index=False)
            
            return [train_df, test_df]
        else:
            print("Dataset didn't pass validation!!!")
    
    def data_preparation(self, dfs):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        datas = dfs
        save_dirs = [self.config.train_folder, self.config.test_folder]
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

        all_deepchem_data = []

        for data, save_dir in zip(datas, save_dirs):
            # data_df = pd.read_csv(data)
            # name = os.path.splitext(os.path.basename(data))[0]  # Extract filename without extension
            name = data.name[0]
            i = 0
            data_df = pd.DataFrame(columns=['name', 'smiles', 'HIV_active'])
            
            for mol in tqdm(data.itertuples(index=True), total=len(data), desc=f"Processing {name}"):

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
                filename = os.path.join(save_dir, f"{name}_{i}.pt")
                torch.save(data_deepchem, filename)
                
                if os.path.exists(filename):
                    mol_graph = torch.load(filename, weights_only=False)
                    
                    if mol_graph.y.item() != None:
                        data_df.loc[i] = [f"{name}_{i}.pt", mol_graph.smiles, mol_graph.y.item()]
                        i += 1
            
            data_df.to_csv(os.path.join(save_dir, f"{name}.csv"), index=False)
    
    def transformation_compose(self):
        dfs = self.test_split_train_balanced()
        self.data_preparation(dfs)
