import numpy as np
import torch
from torch.utils.data import DataLoader
from model import *
from datareader import DataReader
from sklearn.metrics import roc_auc_score
import time
import os
import random


class Training():
    def __init__(self, main_path_to_data, unique_name=''):
        self.main_path_to_data = main_path_to_data
        self.unique_name = unique_name

    def train(self, train_info, valid_info, hyperparameters, model):

        randuid = '%07x' % random.randrange(16**7)
        path_to_model = 'trained_models/' + self.unique_name +  '_' + randuid
        os.mkdir(path_to_model)

        learning_rate = hyperparameters['learning_rate']
        weight_decay = hyperparameters['weight_decay']
        total_epoch = hyperparameters['total_epoch']
        multiplicator = hyperparameters['multiplicator']
        train_batch_size = hyperparameters['train_batch_size']

        # class imbalance
        negative, positive = 0, 0
        for _, label in train_info:
            if label == 0:
                negative += 1
            elif label == 1:
                positive += 1
        
        pos_weight = torch.Tensor([(negative/positive)]).to(os.environ['PU'])
        
        # 4. Create train and validation generators
        train_datareader = DataReader(self.main_path_to_data, train_info)
        train_generator = DataLoader(train_datareader, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=2)
        
        valid_datareader = DataReader(self.main_path_to_data, valid_info)
        valid_generator = DataLoader(valid_datareader, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)
        
        # 5. Prepare model
        Model = ModelResNet18(*hyperparameters['model_params'])

        Model.to(os.environ['PU'])
        
        # 6. Define criterion function, optimizer and scheduler
        criterion_clf = torch.nn.BCEWithLogitsLoss(pos_weight) # pos_weight for class imbalance
        optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, multiplicator, last_epoch=-1)
        
        # 7. Creat lists for tracking AUC and Losses
        aucs = []
        losses = []
        times_train = []
        times_validate = []
        best_auc = -np.inf
        nb_batches = len(train_generator)
        
        # 8. Run training
        for epoch in range(total_epoch):
            start = time.time()
            print('Epoch: %d/%d' % (epoch + 1, total_epoch))
            running_loss = 0

            # A) Train model
            Model.train()  # put model in training mode
            for i_t, item_train in enumerate(train_generator):
                print(f'   {i_t + 1} od {nb_batches}')

                # Forward pass          
                optimizer.zero_grad()
                loss = Model.train_update(item_train, criterion_clf)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Track loss change
                running_loss += loss.item()
            
            times_train.append(time.time() - start)

            # B) Validate model
            predictions = []
            trues = []
            
            Model.eval()  # put model in eval mode
            start = time.time()
            print('Evaluation ...')
            for i_v, item_valid in enumerate(valid_generator):
                # print(f'   {i_v + 1} od {len(valid_generator)}')
                prediction = Model.predict(item_valid, is_prob=True)
                predictions.append(prediction.cpu().numpy()[0])
                trues.append(item_valid[1].numpy()[0])
        
            auc = roc_auc_score(trues, predictions)

            times_validate.append(time.time() - start)

            # C) Track changes, update LR, save best model
            print("AUC: ", auc, ", Running loss: ", running_loss /
                  nb_batches, ", Times: ",  times_train[-1], times_validate[-1])
            
            # If over 1/3 of epochs and best AUC, save model as best model.
            if (epoch >= total_epoch//3) and (auc > best_auc):
                torch.save(Model.state_dict(), path_to_model + '/BEST_model.pth')
                best_auc = auc         
                best_auc_epoch = epoch  
            
            aucs.append(auc)
            losses.append(running_loss/nb_batches)
            scheduler.step()
            
        hyperparameters['best_auc'] = best_auc            
        hyperparameters['best_auc_epoch'] = best_auc_epoch
            
        np.save(path_to_model + '/AUCS.npy', np.array(aucs))
        np.save(path_to_model + '/LOSSES.npy', np.array(losses))
        np.save(path_to_model + '/PARAMS.npy', np.array(hyperparameters))
        np.save(path_to_model + '/TIMES-TRAIN.npy', np.array(times_train))
        np.save(path_to_model + '/TIMES-VALIDATE.npy', np.array(times_validate))

        torch.save(Model.state_dict(), path_to_model + '/LAST_model.pth')
        
        return aucs, losses, path_to_model
