import os
from box.exceptions import BoxValueError
import yaml
from hivclass import logger
import json
import numpy as np
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_curve
    )
import seaborn as sns
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
# from ruamel.yaml.scalarfloat import ScalarFloat
# from decimal import Decimal
import re
from io import StringIO

@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool=True):
    """
    create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created.
        Defaults to False.
    """
    
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.full_load(yaml_file)
            # logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("yaml file is empty")
    
    except Exception as e:
        raise e

@ensure_annotations
def prepare_yaml_and_inline_lists(data):
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)

    # Fix float representation
    def float_representer(representer, data):
        value = "{:.10g}".format(data)
        if '.' not in value and 'e' not in value:
            value += '.0'
        return representer.represent_scalar('tag:yaml.org,2002:float', value)

    yaml.representer.add_representer(float, float_representer)

    # Recursively replace lists with inline CommentedSeq
    def force_inline_lists(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, list):
                    new_list = []
                    for item in v:
                        if isinstance(item, str) and re.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", item):
                            try:
                                item = float(item)
                            except ValueError:
                                pass
                        new_list.append(item)
                    seq = CommentedSeq(new_list)
                    seq.fa.set_flow_style()
                    d[k] = seq  # ✅ Replace, don’t append
                elif isinstance(v, dict):
                    force_inline_lists(v)
                elif isinstance(v, str) and re.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", v):
                    try:
                        d[k] = float(v)
                    except ValueError:
                        pass

    force_inline_lists(data)
    
    stream = StringIO()
    yaml.dump(data, stream)
    yaml_output = stream.getvalue()

    return yaml_output

@ensure_annotations
def save_json(path: str, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")

@ensure_annotations
def get_size(path: Path) -> str:
    """
    get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    
    size_in_kb = round(os.path.getsize(path)/1024)
    
    return f"~ {size_in_kb} KB"

def plot_metric(
        metric_path,
        range,
        train_matric,
        val_matric,
        type
    ):

    
    plt.figure(figsize=(12, 6))
    plt.plot(range, train_matric, label=f"train_{type}")
    plt.plot(range, val_matric, label=f"val_{type}")
    plt.xlabel('Epochs')
    plt.ylabel(f"{type}")
    plt.title(f'Train/Validation {type}')
    plt.legend()
    plt.savefig(
        os.path.join(
            metric_path,
            f'Train_Val_{type}.png'
        )
    )

def plot_confusion_matrix(conf_matrix, cm_path, epoch):
    # Transpose the matrix to match y-axis as Predicted, x-axis as Actual
    TN = conf_matrix[0,0]
    FP = conf_matrix[0,1]
    FN = conf_matrix[1,0]
    TP = conf_matrix[1,1]
    
    conf_matrix = np.array([[TP, FP], [FN, TN]])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=[1, 0], yticklabels=[1, 0], 
                cbar=True, annot_kws={"size": 12})
    
    # Set axis labels
    plt.xlabel("Actual", fontsize=12)
    plt.ylabel("Predicted", fontsize=12)
    
    plt.title("Confusion Matrix for epoch {epoch}")
    
    # Adjust layout
    plt.tight_layout()
    
    plt.savefig(os.path.join(cm_path, f'cm_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_metric_curve(xs, ys, curve_path, epoch, pr=True):
    if pr:
        xlabel = "Precision"
        ylabel = "Recall"
        title = "Precision-Recall Curve"
        lable = "PR_Curve"
    else:
        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
        title = "Receiver Operating Characteristic Curve"
        lable = "ROC_Curve"
    
    plt.figure()
    plt.plot(xs, ys, label=lable)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(
        curve_path,
        f'{lable}_epoch_{epoch}.png'
    ))

def metric_report(preds_float, preds, labels, report_path, epoch):
    mcc = matthews_corrcoef(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    f2 = fbeta_score(labels, preds, beta=2, zero_division=0)
    
    pr, re, thr1 = precision_recall_curve(labels, preds_float)
    auc_pr = auc(re, pr)
    
    fpr, tpr, thr2 = roc_curve(labels, preds_float)
    auc_roc = auc(fpr, tpr)
    
    cm = confusion_matrix(labels, preds)
    
    report = {
        'mcc': mcc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'f2_score': f2,
        'auc_pr': auc_pr,
        'auc_roc': auc_roc
    }
    
    save_json(
        os.path.join(report_path, f'report_{epoch}.json'),
        report
    )
    
    plot_metric_curve(pr, re, report_path, epoch)
    plot_metric_curve(fpr, tpr, report_path, epoch, pr=False)
    plot_confusion_matrix(cm, report_path, epoch)
    
    return mcc, f2, auc_pr
