from hivclass.utils.molecule_dataset import MoleculeDataset
from hivclass.entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from hivclass.utils.molecule_dataset import MoleculeDataset
from hivclass.utils.mol_gnn import MolGNN
import os
from hivclass.utils.main_utils import save_json, plot_confusion_matrix, plot_roc_curve
import torch 
from torch_geometric.data import DataLoader
from box import ConfigBox
import sys
from tqdm import tqdm

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def dataset_preparation(self):
        test_dataset = MoleculeDataset(
            source_root=self.config.source_root,
            processed_root=self.config.processed_root,
            source_filename=self.config.source_filename,
            processed_filename=self.config.processed_filename,
            test=True
        )
        
        test_loader = DataLoader(test_dataset, batch_size=self.config.params['batch_size'])
        
        return test_dataset, test_loader
    
    def model_preparation(self, dataset, device):
        params = self.config.params
        params['model_edge_dim'] = dataset[0].edge_attr.shape[1]
        model_params = ConfigBox({k: v for k, v in params.items() if k.startswith("model_")})
        
        model = MolGNN(feature_size=dataset[0].x.shape[1], model_params=model_params).to(device)
        
        model_path = os.path.join(
            self.config.model_folder_path,
            os.listdir(self.config.model_folder_path)[0]
        )
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        return model
    
    def testing(self, test_loader, model, device):
        model.eval()
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_loader)):
                batch = batch.to(device)
                
                preds = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
                test_preds.extend(torch.round(torch.squeeze(preds)).cpu().detach().numpy())
                test_labels.extend(batch.y.cpu().detach().numpy())
                
                accuracy = accuracy_score(test_labels, test_preds)
                
                sys.stdout.write(
                    "Batch:%2d/%2d - test_accuracy:%.4f" %(
                        i,
                        len(test_loader),
                        accuracy
                    )
                )
                sys.stdout.flush()
        
        return test_preds, test_labels
    
    def evaluation(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", device)
        
        test_dataset, test_loader = self.dataset_preparation()
        
        model = self.model_preparation(test_dataset, device)
        
        test_preds, test_labels = self.testing(test_loader, model, device)
        
        report = classification_report(
                    test_labels,
                    test_preds,
                    zero_division=0,
                    output_dict=True
                )

        save_json(
            os.path.join(self.config.root_dir, f'report.json'),
            report
        )

        conf_matrix = confusion_matrix(test_labels, test_preds)

        plot_confusion_matrix(
            conf_matrix,
            self.config.root_dir,
            'test'
        )

        auc_score = roc_auc_score(test_labels, test_labels)
        auc_score_dict = {'auc_score': auc_score}

        save_json(
            os.path.join(self.config.root_dir, f'auc_score.json'), 
            auc_score_dict
        )
        
        plot_roc_curve(
            test_labels,
            test_preds,
            self.config.root_dir,
            'test'
        )
