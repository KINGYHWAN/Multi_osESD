
class osESD_supervised :
    rwin_size = 10
    dwin_size = 10
    init_size = 100
    alpha = 0.05
    maxr = 5
    # train_size = 0.2
    epochs = 10
    early_stop = 3
    total_change_rate = 0.00001    # 보통은 0.0001
    total_o_change_rate = 0.005   # 보통은 0.001


# 20	20	50	0.05	10	0.0001	0.0001

class osESD_supervised_ARIMA :
    rwin_size = 20
    dwin_size = 20
    init_size = 50
    alpha = 0.05
    maxr = 10
    # train_size = 0.2
    epochs = 10
    early_stop = 3
    total_change_rate = 0.0001   # 보통은 0.0001
    total_o_change_rate = 0.0001 # 보통은 0.001

# 20	20	100	0.05	10	1.00E-05	0.0001
# 20	20	100	0.05	10	1.00E-05	0.0001

class osESD_supervised_seasonal :
    rwin_size = 20
    dwin_size = 20
    init_size = 100
    alpha = 0.05
    maxr = 10
    # train_size = 0.2
    epochs = 10
    early_stop = 3
    total_change_rate = 0.00005    # 보통은 0.0001
    total_o_change_rate = 0.0001   # 보통은 0.001
# 5	2	50	0.05	10	1.00E-05	0.005

# 10	10	50	0.05	10	1.00E-05	0.0001
class osESD_supervised_yahoo :
    rwin_size = 10
    dwin_size = 10
    init_size = 50
    alpha = 0.05
    maxr = 10
    # train_size = 0.2
    epochs = 10
    early_stop = 3
    total_change_rate = 1.00E-05    # 보통은 0.0001
    total_o_change_rate = 0.0001  # 보통은 0.001

# 20	20	50	0.001	10	0.0005	0.005

class osESD_supervised_NAB1:
    rwin_size = 20
    dwin_size = 20
    init_size = 100
    alpha = 0.001
    maxr = 10
    # train_size = 0.2
    epochs = 10
    early_stop = 3
    total_change_rate = 0.0005    # 보통은 0.0001
    total_o_change_rate = 0.005   # 보통은 0.001

# 20	20	50	0.01	10	0.0005	0.005

class osESD_supervised_NAB2:
    rwin_size = 20
    dwin_size = 20
    init_size = 50
    alpha = 0.01
    maxr = 10
    # train_size = 0.2
    epochs = 10
    early_stop = 3
    total_change_rate = 0.0005    # 보통은 0.0001
    total_o_change_rate = 0.005   # 보통은 0.001


class osESD_unsupervised:
    rwin_size = 10
    dwin_size = 10
    init_size = 100
    alpha = 0.05
    maxr = 5
    # train_size = 0.2
    epochs = 10
    early_stop = 3
    total_change_rate = 0.001    # 보통은 0.0001
    total_o_change_rate = 0.001   # 보통은 0.001

class KNN_parameters :
    neighbors = 3

class IF_parameters:
    n_estimators = 1000
    max_samples = 50
    contamination = 0.03
    plot = True

class rrcf_parameters:
    num_tree = 40
    shingle_size = 4
    tree_size = 256
    plot = True

class ae_parameters:
    lr = 0.0003
    batch_size = 128
    num_epochs = 50
    plot = False
    sequence_len = 20

class vae_parameters:
    lr = 0.0003
    batch_size = 64
    num_epochs = 50
    plot = False
    sequence_len = 10

class lstmed_parameters:
    lr = 0.0003
    batch_size = 128
    num_epochs = 50
    plot = False
    sequence_len = 10

# ### Example usage
# class testing_classifier_parameters:
#     window_length = 50
#     batch_size = 32
#     epochs = 10
#     lr = 0.0003


### Example usage
class CNN_parameters:
    window_length = 20
    batch_size = 64
    epochs = 30
    lr = 0.0003
    delay_y = 1
    retrain_frequency = 5000

class MLP_parameters:
    window_length = 20
    batch_size = 64
    epochs = 30
    lr = 0.0003
    delay_y = 1
    retrain_frequency=5000

class TabNet_params:
    window_length = 20
    batch_size = 64
    num_epochs = 10
    eval_metric = 'accuracy'
    y_index = 1


class Utime_params:
    window_length = 200
    num_epochs = 100



# class osESD_parameters:
#     size = 100
#     dwin = 2
#     rwin = 5
#     maxr = 10
#     alpha = 0.001
#     plot = True
#     condition = True


