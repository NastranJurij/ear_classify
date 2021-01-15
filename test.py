# import numpy as np
import torch
from torch.utils.data import DataLoader
from model import *
from datareader import DataReader
from sklearn.metrics import roc_auc_score, roc_curve, det_curve, precision_recall_curve, f1_score, accuracy_score, confusion_matrix
import os

class Testing():
    def __init__(self, main_path_to_data):
        self.main_path_to_data = main_path_to_data
        
    def test(self, test_info, path_to_model, hyperparameters, model, central_only=False):

        Model = ModelResNet18(*hyperparameters['model_params'])

        Model.load_state_dict(torch.load(path_to_model))
        Model.eval()
        Model.to(os.environ['PU'])

        batch_size=1
        
        # 2. Create dataloader
        test_datareader = DataReader(self.main_path_to_data, test_info)
        test_generator = DataLoader(test_datareader, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

        # 3. Calculate metrics
        predictions = []
        trues = []
        
        for i, item_test in enumerate(test_generator):
            print(f'    {i+1}/{len(test_generator)}')
            prediction = Model.predict(item_test, is_prob=True).cpu().numpy()[0]
            predictions.append(prediction)
            trues.append(item_test[1].numpy()[0])
            
        print('Calculating scores ...')
        fpr, tpr, thresholds = roc_curve(trues, predictions, pos_label=1)
        roc = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}

        fpr, fnr, thresholds = det_curve(trues, predictions, pos_label=1)
        det = {'fpr': fpr, 'fnr': fnr, 'thresholds': thresholds}

        precision, recall, thresholds = precision_recall_curve(trues, predictions, pos_label=1)
        prec_rec = {'precision': precision, 'recall': recall, 'thresholds': thresholds}

        test_results = {'roc_curve': roc, 'det_curve': det, 'prec_rec_curve': prec_rec,
                        'trues': trues, 'predictions': predictions, 'full_predictions': prediction}


        if len(test_generator) > 1:
            auc = roc_auc_score(trues, predictions)
            test_results.update({'auc': auc})

        return test_results
