
import random
import pandas as pd
import numpy as np
import torch
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)

'''
ALSO, ESPECIALLY FOR SKLEARN, ADD THIS NUMBER TO RANDOM_STATE TO ENSURE REPRODUCIBILITY
'''

import test_ablation
import test_sequential_supervised_point_detection 
import test_unsupervised_point_detection
import test_supervised_point_detection
import test_unsupervised_classification
import test_gov

import tuning_test_supervised_point
import tuning_test_supervised_all_others

from utils import call_datasets
from utils import data_config


if __name__ == '__main__':


    # testing_df = pd.read_csv("results//sequential_single_point_algorithms.csv")
    # print(testing_df.head())
    # new = data_config.average_according_to_dataset2(testing_df, 'Dataset', 'train_percent')
    # print(new.head())
    # new.to_csv("check.csv")

    # test_supervised_point_detection.supervised_multivariate_tests()
    # checkifsupervisedworksandchangelengthwithruntimes
    # test_sequential_supervised_point_detection.supervised_sequential_multivariate_tests()
    # test_ablation.ablation_test()
    # test_unsupervised_classification.unsupervised_classification_tests()


    # test_gov.run_gov_test()
    # tuning_test_supervised_point.supervised_multivariate_tests()
    tuning_test_supervised_all_others.supervised_multivariate_compare_tests()





