from hivclass.utils.molecule_dataset import MoleculeDataset
from hivclass.entity.config_entity import ModelTrainerConfig
from hivclass.utils.main_utils import create_directories, read_yaml
from hivclass.constants import PARAMS_FILE_PATH
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from pathlib import Path
import os
import numpy as np
import pandas as pd
import yaml
from hivclass.utils.molecule_dataset import MoleculeDataset
from hivclass.utils.mol_gnn import MolGNN
from hivclass.utils.main_utils import (
    save_json,
    plot_metric,
    plot_confusion_matrix,
    plot_roc_curve,
    prepare_yaml_and_inline_lists
)
import torch 
from torch_geometric.data import DataLoader
from box import ConfigBox
import sys
from tqdm import tqdm
from mango import Tuner

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train_val_separation(self, train_dataset):
        data_df = pd.read_csv(os.path.join(
            self.config.processed_root,
            self.config.processed_filename[3]
        ))
        
        train_df, val_df = train_test_split(
            data_df,
            test_size=self.config.params.data_transformation.val_size,
            stratify=data_df.HIV_active,
            random_state=42
        )
        
        train_idxs = train_df.index.tolist()
        val_idxs = val_df.index.tolist()
        
        train = train_dataset.index_select(train_idxs)
        val = train_dataset.index_select(val_idxs)
        
        return train, val
    
    def train(self, params, epoch, model, train_loader, optimizer, criterion, device):
        model.train()
        total_loss = 0.0
        train_preds = []
        train_labels = []
        
        for i, batch in tqdm(enumerate(train_loader)):
            batch.to(device)
            optimizer.zero_grad()
            
            preds = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
            train_preds.extend(np.rint(torch.sigmoid(preds).cpu().detach().numpy()))
            train_labels.extend(batch.y.cpu().detach().numpy())
            
            loss = criterion(torch.squeeze(preds), batch.y.float())
            loss.backward()
            optimizer.step()
            
            if None not in train_preds:
                accuracy = accuracy_score(train_labels, train_preds)
            else:
                accuracy = 0.0
            
            total_loss += loss.item()
            
            print()
            sys.stdout.write(
                "Epoch:%2d/%2d - Batch:%2d/%2d - train_loss:%.4f - train_accuracy:%.4f" %(
                    epoch+1,
                    params.num_epochs,
                    i,
                    len(train_loader),
                    loss.item(),
                    accuracy
                )
            )
            sys.stdout.flush()
        
        return total_loss / len(train_loader), accuracy
    
    def validation(self, epoch, model, val_loader, criterion, best_val_loss, stats_path,  device):
        model.eval()
        total_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch.to(device)
                
                preds = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
                val_preds.extend(np.rint(torch.sigmoid(preds).cpu().detach().numpy()))
                val_labels.extend(batch.y.cpu().detach().numpy())
                
                loss = criterion(torch.squeeze(preds), batch.y.float())
                total_loss += loss.item()
                accuracy = accuracy_score(val_labels, val_preds)
            
            epoch_loss = total_loss / len(val_loader)
            
            if epoch_loss < best_val_loss:
                report = classification_report(
                    val_labels,
                    val_preds,
                    zero_division=0,
                    output_dict=True
                )

                save_json(
                    os.path.join(stats_path, f'report_{epoch}.json'),
                    report
                )

                conf_matrix = confusion_matrix(val_labels, val_preds)

                plot_confusion_matrix(
                    conf_matrix,
                    stats_path,
                    epoch
                )

                auc_score = roc_auc_score(val_labels, val_preds)
                auc_score_dict = {'auc_score': auc_score}

                save_json(
                    os.path.join(stats_path, f'auc_score_{epoch}.json'), 
                    auc_score_dict
                )
                
                plot_roc_curve(
                    val_labels,
                    val_preds,
                    stats_path,
                    epoch
                )
                
        return epoch_loss, accuracy
    
    def train_compose(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", device)
        
        dataset = MoleculeDataset(
            self.config.source_root,
            self.config.processed_root,
            self.config.source_filename,
            self.config.processed_filename
        )
        
        train_dataset, val_dataset = self.train_val_separation(dataset)
        
        def train_tuning(params):
            try:
                params = params[0]
                
                if self.config.tuning:
                    folder_name = str(len(os.listdir(self.config.stats)) + 1)
                else:
                    folder_name = "best_retrain"
                
                models_path = os.path.join(self.config.models, folder_name)
                stats_path = os.path.join(self.config.stats, folder_name)
                
                create_directories([models_path, stats_path])

                # params_dict = params.to_dict()
                # params_yaml = prepare_yaml_and_inline_lists(params_dict)

                # with open(os.path.join(stats_path, "params.yaml"), 'w') as file:
                #     file.write(params_yaml)
                
                with open(os.path.join(stats_path, "params.yaml"), 'w') as file:
                    file.write(yaml.dump(params, sort_keys=False))
                
                params = ConfigBox(params)
                
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
                params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]
                
                print("Loading model...")
                model_params = ConfigBox({k: v for k, v in params.items() if k.startswith("model_")})
                model = MolGNN(feature_size=train_dataset[0].x.shape[1], model_params=model_params)
                model = model.to(device)
                
                weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=params['learning_rate'],
                    momentum=params['sgd_momentum'],
                    weight_decay=params['weight_decay']
                )
                
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.scheduler_gamma)
                
                train_losses = []
                val_losses = []
                train_accuracies = []
                val_accuracies = []
                best_val_loss = float('inf')
                early_stopping_counter = 0
                
                for epoch in tqdm(range(params.num_epochs)):
                    if early_stopping_counter <= params.early_stopping:
                        train_epoch_loss, train_epoch_acc = self.train(
                            params,
                            epoch,
                            model,
                            train_loader,
                            optimizer,
                            criterion,
                            device
                        )
                        
                        train_losses.append(train_epoch_loss)
                        train_accuracies.append(train_epoch_acc)
                        
                        val_epoch_loss, val_epoch_acc = self.validation(
                            epoch,
                            model,
                            val_loader,
                            criterion,
                            best_val_loss,
                            stats_path,
                            device
                        )
                        
                        val_losses.append(val_epoch_loss)
                        val_accuracies.append(val_epoch_acc)
                        
                        print(f'Epoch [{epoch+1}/{params.num_epochs}], '
                            f'Loss: {train_epoch_loss:.4f}, '
                            f'Validation Loss: {val_epoch_loss:.4f}, '
                            f'Train Accuracy: {train_epoch_acc:.2f}%, '
                            f'Validation Accuracy: {val_epoch_acc:.2f}%')
                        
                        if float(val_epoch_loss) < best_val_loss:
                            torch.save(model.state_dict(), os.path.join(models_path, f'model_{epoch}.pth'))
                            best_val_loss = float(val_epoch_loss)
                            early_stopping_counter = 0
                        else:
                            early_stopping_counter += 1
                        
                        scheduler.step()
                    else:
                        print("Early stopping due to no improvement.")
                        epochs_range = range(1, len(train_losses) + 1)
                        
                        plot_metric(
                            stats_path,
                            epochs_range,
                            train_losses,
                            val_losses,
                            'Train Loss',
                            'Validation Loss',
                        )
                        
                        plot_metric(
                            stats_path,
                            epochs_range,
                            train_accuracies,
                            val_accuracies,
                            'Train Accuracies',
                            'Validation Accuracies',
                        )
                        
                        return [best_val_loss]
                
                print(f"Finishing training with best test loss: {best_val_loss}")
                epochs_range = range(1, params.num_epochs + 1)

                plot_metric(
                    stats_path,
                    epochs_range,
                    train_losses,
                    val_losses,
                    'Train Loss',
                    'Validation Loss',
                )
                
                plot_metric(
                    stats_path,
                    epochs_range,
                    train_accuracies,
                    val_accuracies,
                    'Train Accuracies',
                    'Validation Accuracies',
                )
                
                return [best_val_loss]
            
            except Exception as e:
                print(f"[ERROR] Training failed for params: {params}")
                print(f"[ERROR] Exception: {e}")

                # return a large loss so Mango avoids this config
                return [float('inf')]
        
        if self.config.tuning:
            print("Running hyperparameter search...")
            params = self.config.params.HYPERPARAMETERS
            config = dict()
            config["optimizer"] = params.tuning_optimizer[0]
            config["num_iteration"] = params.tuning_iterations[0]
            
            tuner = Tuner(params, objective=train_tuning, conf_dict=config)
            
            results = tuner.minimize()
            
            self.config.params['BEST_PARAMETERS'] = results['best_params']
            params_dict = self.config.params.to_dict()
            params_yaml = prepare_yaml_and_inline_lists(params_dict)
            # best_params = yaml.safe_dump(self.config.params.to_dict(), sort_keys=False, default_flow_style=True)
            
            with open(PARAMS_FILE_PATH, 'w') as file:
                file.write(params_yaml)
            
            for folder in os.listdir(self.config.stats):
                params_path = self.config.stats.joinpath(folder, 'params.yaml')
                tuning_params = read_yaml(params_path)
                
                if self.config.params['BEST_PARAMETERS'] == tuning_params:
                    best_stats_folder_path = self.config.stats.joinpath(folder)
                    best_model_folder_path = self.config.models.joinpath(folder)
                    
                    try:
                        best_stats_folder_path.rename(self.config.stats.joinpath('best_params'))
                        print(f"Folder renamed from {best_stats_folder_path} to {self.config.stats.joinpath('best_params')}")
                    except FileNotFoundError:
                        print(f"Error: The folder {best_stats_folder_path} does not exist.")
                    except FileExistsError:
                        print(f"Error: A folder named {self.config.stats.joinpath('best_params')} already exists.")
                    except PermissionError:
                        print(f"Error: Permission denied to rename the folder.")
                    
                    try:
                        best_model_folder_path.rename(self.config.models.joinpath('best_params'))
                        print(f"Folder renamed from {best_stats_folder_path} to {self.config.stats.joinpath('best_params')}")
                    except FileNotFoundError:
                        print(f"Error: The folder {best_model_folder_path} does not exist.")
                    except FileExistsError:
                        print(f"Error: A folder named {self.config.models.joinpath('best_params')} already exists.")
                    except PermissionError:
                        print(f"Error: Permission denied to rename the folder.")
        else:
            params = self.config.params.BEST_PARAMETERS
            params = params.to_dict()
            _ = train_tuning([params])
