
import math
from math import e
import numpy as np
import pandas as pd


from .main_logistic_regression import LogisticRegressionSequential
from .main_osESD_components import osESD
# from osESD_components import SESD_tcha
# from osESD_components import SESD_tres
# from osESD_components import TCHA
# from osESD_components import TRES
from utils.data_config import index_to_preds
# from sklearn.metrics import f1_score, classification_report
from utils.scoring import f1_score
from utils.data_config import anom_replace



import math
from math import e
import numpy as np
import pandas as pd


from .main_logistic_regression import LogisticRegressionSequential
from .main_osESD_components import osESD
# from osESD_components import SESD_tcha
# from osESD_components import SESD_tres
# from osESD_components import TCHA
# from osESD_components import TRES
from utils.data_config import index_to_preds
# from sklearn.metrics import f1_score, classification_report
from utils.scoring import f1_score
from utils.data_config import anom_replace


def multi_osESD_supervised_no_backpropogation(x_data, y_data, params, train_percent, testing_size):

    rwin_size = params.rwin_size
    dwin_size = params.dwin_size
    init_size = params.init_size
    alpha = params.alpha
    maxr = params.maxr
    epochs = params.epochs
    early_stop = params.early_stop
    total_change_rate = params.total_change_rate
    total_o_change_rate =params.total_o_change_rate

    if train_percent>1:
        train_size = train_percent
    else:
        train_size = int(len(x_data) * train_percent)
    cols = [col for col in x_data.columns]

    if testing_size >1 :
        test_return_size = len(x_data)-testing_size
    else:
        test_return_size = int(len(x_data) * round(1 - testing_size, 5))


    # print(round(1 - testing_size, 2))
    # adasd
    uni_col_preds = []
    # rwin_size, dwin_size = 10, 10
    for col in cols:
        col_df = list(x_data[col])
        col_class = osESD(data=col_df, dwins=dwin_size, rwins=rwin_size, init_size=init_size, alpha=alpha, maxr=maxr,
                          condition=True)
        # col_class.initiate()
        col_index = col_class.predict_all()
        uni_col_preds.append(index_to_preds(col_index, len(col_df)))

    Col_num = len(cols) * 4
    train_online_len = train_size - init_size
    transformed_data = [[-1 for _ in range(Col_num)] for _ in range(train_online_len)]


    log_lr = [total_change_rate for _ in range(Col_num + 1)]
    log_lr_array = np.array(log_lr).reshape(1, -1)
    log_lr_array_transposed = log_lr_array.T
    lr_sum = sum(log_lr)

    log_online_lr = [total_o_change_rate for _ in range(Col_num + 1)]
    log_online_lr_array = np.array(log_online_lr).reshape(1, -1)
    log_online_lr_array_transposed = log_online_lr_array.T
    online_lr_sum = sum(log_online_lr)

    best_f1 = -1  ### For comparing for early stop
    best_anoms = [0 for _ in range(len(x_data)-train_online_len-init_size)]
    early_stop_count = 0   ### For counting consecutive stops.

    past_i_f1_cols = [-1 for _ in range(len(cols))] ### for saving past epoch f1 scores
    lr_update = [1 for _ in range(len(cols))] ### for setting direction in moving next gradient
    col_idx_list = [i for i in range(len(past_i_f1_cols))]

    # k = math.log(2)/train_size ### for decaying lr, making it to 0.5 at the end
    for train_epoch in range(epochs):

        CLASSES = {}
        for col in cols:
            col_df = list(x_data[col])
            col_class = osESD(data=col_df, dwins=dwin_size, rwins=rwin_size, init_size=init_size, alpha=alpha,
                              maxr=maxr, condition=True)
            # col_class.initiate()
            CLASSES[col] = col_class
        # print(train_online_len)
        # adadassd
        for i in range(train_online_len):
            for idx, col in enumerate(cols):  # cols
                col_class = CLASSES[col]
                c_val, r_val, c_anom, r_anom = col_class.test_values(i)
                transformed_data[i][idx * 4] = c_val
                transformed_data[i][idx * 4 + 1] = r_val
                transformed_data[i][idx * 4 + 2] = c_anom
                transformed_data[i][idx * 4 + 3] = r_anom
                anom_val = col_class.check_values(c_anom, r_anom)

        train_x = []
        train_y = []

        for idx in range(train_online_len):
            vals = transformed_data[idx]
            if sum(vals) == 0:
                if np.random.uniform(0, 1) > 0.1:
                    continue
            train_x.append(vals)
            train_y.append(y_data[idx + init_size])

        log_model = LogisticRegressionSequential()
        log_model.train(np.array(train_x), np.array(train_y).astype(int).reshape(-1, 1), num_iterations=100,
                        lrs=log_lr_array_transposed)

        # anom_idx = []
        anoms = []

        for i in range(train_online_len, len(x_data) - init_size):
            row_vals = []
            real_y = y_data[i + init_size]
            for idx, col in enumerate(cols):  # cols
                col_class = CLASSES[col]
                results = col_class.test_values(i)  # c_val, r_val, c_anom, r_anom
                row_vals.append(results[0])
                row_vals.append(results[1])
                row_vals.append(results[2])
                row_vals.append(results[3])

            row_pred = log_model.predict(row_vals)[0]
            anoms.append(row_pred)

            # print([row_vals])
            # print([[real_y]])
            # adsad
            # print([row_vals])
            # adasd
            log_model.train_incremental(np.array([row_vals]), np.array([[real_y]]), lrs=log_online_lr_array_transposed)
            # print(row_pred)
            # if i==train_online_len+10:
            #     adasdsa

            if row_pred == 1:
                skip_next = False
                for single_anom_idx, single_anom_val in enumerate(row_vals):
                    if skip_next:
                        skip_next=False
                        continue
                    if single_anom_val == 1:
                        skip_next=True
                        col_idx = single_anom_idx // 4
                        col_class = CLASSES[cols[col_idx]]
                        # print(col_class.r_ins.tres)
                        # print(col_class.c_ins.tcha)
                        # adssads
                        if single_anom_idx % 2 == 0:
                            anom_val = col_class.check_values(row_vals[single_anom_idx], row_vals[single_anom_idx + 1])
                        else:
                            anom_val = col_class.check_values(row_vals[single_anom_idx - 1], row_vals[single_anom_idx])

        # print(sum(anoms))
        # print(sum(anoms))
        # adsasdsad

        return anoms[-test_return_size:]
    
        # check = pd.DataFrame({'y': y_data[train_size:len(y_data)], 'preds': anoms})

        # recent_f1 = f1_score(check['y'], check['preds'])
        # # print(log_lr_array)
        # # print("New F1 : {:.4f}, Best F1 : {:.4f}".format(recent_f1,best_f1))
        # if best_f1>recent_f1:  ### Case where next epoch yielded worse results than final epoch.
        #     early_stop_count +=1
        #     if early_stop_count==early_stop:
        #         # print("Ended early with only {} epochs".format(train_epoch))
        #         # print(len(anoms))
        #         # print(test_return_size)
        #         print(1)
        #         print(test_return_size)
        #         return best_anoms[-test_return_size:]
            

        # else: ### New epoch is best.
        #     early_stop_count = 0
        #     best_f1 = recent_f1
        #     best_anoms = anoms

        # ### End of training
        # if train_epoch==epochs-1:
        #     # print(len(anoms))
        #     # print(test_return_size)
        #     print(2)
        #     return best_anoms[-test_return_size:]



        # ### 여기 아래 부분을 따로 function으로 만들든지 하는게 더 깔끔할듯?

        # ### Strategy 1 : 역수.  폐기

        # ### Strategy 2 : 1-f1_score 만큼 일단 추가하는데, 전보다 낮아지면 오히려 마이너스로.
        # ### 그리고 이게 첫 f1_score은 -1이기 때문에 무조건 플러스 함.

        # ### 지금 이게 1-X 로 하니까 다 너무 비슷비슷하다.
        # check_f1s = []
        # i_col_f1s = []
        # for idx, col in enumerate(cols):
        #     uni_preds = uni_col_preds[idx][train_online_len + init_size:]
        #     i_col_f1s.append(1-f1_score(uni_preds,anoms))
        #     check_f1s.append(f1_score(uni_preds, anoms))
        #     # i_col_f1s.append(1 / f1_score(uni_preds, anoms))
        # i_f1_sum = sum(i_col_f1s)
        # # print(check_f1s)
        # # print(sum(anoms))
        # for idx, past, now in zip(col_idx_list, past_i_f1_cols, i_col_f1s):
        #     if now<past:
        #         lr_update[idx]*=(-1)

        # past_i_f1_cols = i_col_f1s
        # i_f1_sum = sum(i_col_f1s)
        # d_lr = [total_change_rate * i / i_f1_sum for i in i_col_f1s]
        # d_o_lr = [total_o_change_rate * i / i_f1_sum for i in i_col_f1s]
        # # print(d_lr)
        # for col_idx in range(len(cols)):
        #     for times in range(4):
        #         log_lr[col_idx * 4 + times] += d_lr[col_idx] * lr_update[col_idx]
        #         log_online_lr[col_idx * 4 + times] += d_o_lr[col_idx] * lr_update[col_idx]

        # new_sum = sum(log_lr)
        # log_lr = [i*lr_sum/new_sum for i in log_lr]
        # new_o_sum = sum(log_online_lr)
        # log_online_lr = [i*lr_sum/new_o_sum for i in log_online_lr]

        # log_lr_array = np.array(log_lr).reshape(1, -1)
        # log_lr_array_transposed = log_lr_array.T
        # log_online_lr_array = np.array(log_online_lr).reshape(1, -1)
        # log_online_lr_array_transposed = log_online_lr_array.T

        # # total_o_change_rate *= e**(-k*i)




def multi_osESD_supervised_no_backpropogation_del(x_data, y_data, params,train_percent):

    rwin_size = params.rwin_size
    dwin_size = params.dwin_size
    init_size = params.init_size
    alpha = params.alpha
    maxr = params.maxr
    epochs = params.epochs
    early_stop = params.early_stop
    total_change_rate = params.total_change_rate
    total_o_change_rate =params.total_o_change_rate

    if train_percent>1:
        train_size = train_percent
    else:
        train_size = int(len(x_data) * train_percent)
    cols = [col for col in x_data.columns]

    uni_col_preds = []  
    # rwin_size, dwin_size = 5,5
    for col in cols:
        col_df = list(x_data[col])
        col_class = osESD(data=col_df, dwins=dwin_size, rwins=rwin_size, init_size=init_size, alpha=alpha, maxr=maxr,
                          condition=True)
        # col_class.initiate()
        col_index = col_class.predict_all()
        uni_col_preds.append(index_to_preds(col_index, len(col_df)))

    Col_num = len(cols) * 4
    train_online_len = train_size - init_size
    transformed_data = [[-1 for _ in range(Col_num)] for _ in range(train_online_len)]


    log_lr = [total_change_rate for _ in range(Col_num + 1)]
    log_lr_array = np.array(log_lr).reshape(1, -1)
    log_lr_array_transposed = log_lr_array.T
    lr_sum = sum(log_lr)

    log_online_lr = [total_o_change_rate for _ in range(Col_num + 1)]
    log_online_lr_array = np.array(log_online_lr).reshape(1, -1)
    log_online_lr_array_transposed = log_online_lr_array.T
    online_lr_sum = sum(log_online_lr)

    best_f1 = -1  ### For comparing for early stop
    best_anoms = [0 for _ in range(len(x_data)-train_online_len-init_size)]
    early_stop_count = 0   ### For counting consecutive stops.

    past_i_f1_cols = [-1 for _ in range(len(cols))] ### for saving past epoch f1 scores
    lr_update = [1 for _ in range(len(cols))] ### for setting direction in moving next gradient
    col_idx_list = [i for i in range(len(past_i_f1_cols))]

    # k = math.log(2)/train_size ### for decaying lr, making it to 0.5 at the end
    # for train_epoch in range(1):

    CLASSES = {}
    for col in cols:
        col_df = list(x_data[col])
        col_class = osESD(data=col_df, dwins=dwin_size, rwins=rwin_size, init_size=init_size, alpha=alpha,
                            maxr=maxr, condition=True)
        # col_class.initiate()
        CLASSES[col] = col_class
    # print(train_online_len)
    # adadassd
    for i in range(train_online_len):
        for idx, col in enumerate(cols):  # cols
            col_class = CLASSES[col]
            c_val, r_val, c_anom, r_anom = col_class.test_values(i)
            transformed_data[i][idx * 4] = c_val
            transformed_data[i][idx * 4 + 1] = r_val
            transformed_data[i][idx * 4 + 2] = c_anom
            transformed_data[i][idx * 4 + 3] = r_anom
            anom_val = col_class.check_values(c_anom, r_anom)

    train_x = []
    train_y = []

    for idx in range(train_online_len):
        vals = transformed_data[idx]
        if sum(vals) == 0:
            if np.random.uniform(0, 1) > 0.1:
                continue
        train_x.append(vals)
        train_y.append(y_data[idx + init_size])

    log_model = LogisticRegressionSequential()
    log_model.train(np.array(train_x), np.array(train_y).astype(int).reshape(-1, 1), num_iterations=100,
                    lrs=log_lr_array_transposed)

    # anom_idx = []
    anoms = []

    for i in range(train_online_len, len(x_data) - init_size):
        row_vals = []
        real_y = y_data[i + init_size]
        for idx, col in enumerate(cols):  # cols
            col_class = CLASSES[col]
            results = col_class.test_values(i)  # c_val, r_val, c_anom, r_anom
            row_vals.append(results[0])
            row_vals.append(results[1])
            row_vals.append(results[2])
            row_vals.append(results[3])

        row_pred = log_model.predict(row_vals)[0]
        anoms.append(row_pred)

        # print([row_vals])
        # print([[real_y]])
        # adsad
        # print([row_vals])
        # adasd
        log_model.train_incremental(np.array([row_vals]), np.array([[real_y]]), lrs=log_online_lr_array_transposed)
        # print(row_pred)
        # if i==train_online_len+10:
        #     adasdsa

        if row_pred == 1:
            skip_next = False
            for single_anom_idx, single_anom_val in enumerate(row_vals):
                if skip_next:
                    skip_next=False
                    continue
                if single_anom_val == 1:
                    skip_next=True
                    col_idx = single_anom_idx // 4
                    col_class = CLASSES[cols[col_idx]]
                    # print(col_class.r_ins.tres)
                    # print(col_class.c_ins.tcha)
                    # adssads
                    if single_anom_idx % 2 == 0:
                        anom_val = col_class.check_values(row_vals[single_anom_idx], row_vals[single_anom_idx + 1])
                    else:
                        anom_val = col_class.check_values(row_vals[single_anom_idx - 1], row_vals[single_anom_idx])
    return anoms

