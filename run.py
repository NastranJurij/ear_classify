import os
from train import Training
from test import Testing
import pandas as pd
import numpy as np

if __name__ == '__main__':
    if os.name == 'nt':
        pu = 'cpu'
        batch_size = 16
        main_path_to_data = './'
        pu_ = pu
    else:
        print("ERROR: Check os.name variable!")
        exit()

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['PU'] = pu

    file_name = "./awe_isLeft.csv"
    info = np.array(pd.read_csv(file_name)[['AWE-npy-path', 'isLeft']] )
    test_info  = info[0:250, :]
    train_info = info[250:750, :]
    valid_info = info[750:, :]

    def Train():
        print(f'Training {model}...')
        TrainClass = Training(main_path_to_data, unique_name=model)
        TestClass = Testing(main_path_to_data)

        _, _, path_to_model = TrainClass.train(train_info, valid_info, hyperparameters, model)

        best_model = path_to_model + '/BEST_model.pth'

        test_results = TestClass.test(test_info, best_model, hyperparameters, model)
        print("Test set AUC result: ", test_results['auc'])

        del TrainClass
        del TestClass


    def DefaultHP():
        hyperparameters = {}
        hyperparameters['learning_rate'] = 0.2e-3
        hyperparameters['weight_decay'] = 0.0001
        hyperparameters['total_epoch'] = 10
        hyperparameters['multiplicator'] = 0.95
        hyperparameters['train_batch_size'] = batch_size
        hyperparameters['model_params'] = []
        hyperparameters['pu'] = pu_
        return hyperparameters



    model = 'ResNet18'
    hyperparameters = DefaultHP()
    hyperparameters['model_params'] = [True]
    hyperparameters['learning_rate'] = 1e-4
    hyperparameters['weight_decay'] = 1e-2
    Train()
