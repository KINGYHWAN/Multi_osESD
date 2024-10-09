
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




def all_scores(y_true, y_pred):
    unique_labels = len(set(y_true))
    average_method = 'binary' if unique_labels == 2 else 'weighted'
    
    a = round(accuracy_score(y_true, y_pred), 3)
    p = round(precision_score(y_true, y_pred, average=average_method), 3)
    r = round(recall_score(y_true, y_pred, average=average_method), 3)
    f = round(f1_score(y_true, y_pred, average=average_method), 3)
    
    return [a, p, r, f]
# ValueError: Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted']