


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report,roc_auc_score
from models import gov_CNN
from utils import parameters
from utils import plotting_modules
from utils import call_datasets
from utils import data_config
from utils import scoring

def run_gov_test():

    gov_df = call_datasets.call_gov_dataset()
    train_df = gov_df[:50000]
    x_cols = gov_df.columns[:-1]
    y_col =  gov_df.columns[-1]

    x_data = gov_df[x_cols]
    y_data = gov_df[y_col]
    y_data = data_config.convert_labels_to_int(y_data)

    input_data = x_data
    input_y = y_data
    train_labels, train_preds, test_labels, test_preds, trained_model = gov_CNN.run_CNN(
        input_data, input_y, 0.05, parameters.CNN_parameters)
    print(classification_report(train_labels, train_preds))
    print(classification_report(test_labels, test_preds))


