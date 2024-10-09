from .main_osESD_components import SESD_tcha
from .main_osESD_components import SESD_tres
from .main_osESD_components import TCHA
from .main_osESD_components import TRES

def orig_osESD(data, dwins, rwins, train_size, alpha, maxr, condition, time=None):
    if time is None:
        time = list(range(1, len(data) + 1))
    offline_data = data[:train_size]
    online_data = data[train_size:len(data)]
    offline_time = time[:train_size]
    online_time = time[train_size:len(data)]
    c_ins = TCHA(data = offline_data, time = offline_time, wins = dwins)
    r_ins = TRES(data = offline_data, time = offline_time, wins = rwins)
    SESD_TCHA = SESD_tcha(data = c_ins.tcha.copy(), alpha = alpha, maxr = maxr)
    SESD_TRES = SESD_tres(data = r_ins.tres.copy(), alpha = alpha, maxr = maxr)
    anomaly_index = []
    for i in range(len(online_data)):
        canom = SESD_TCHA.test(c_ins.update(online_data[i], online_time[i]))
        ranom = SESD_TRES.test(r_ins.update(online_data[i], online_time[i]))
        if condition: function_ = (canom and ranom)
        else: function_ = (canom or ranom)
        if function_ :
            anomaly_index.append(i+train_size)
            D = r_ins.data.copy()
            T = r_ins.time.copy()
            del D[rwins-1]
            del T[rwins-1]
            x_bar = ((rwins*r_ins.x_bar) - r_ins.time[rwins-1]) / (rwins-1)
            y_bar = ((rwins*r_ins.y_bar) - r_ins.data[rwins-1]) / (rwins-1)
            beta_ = sum((T-x_bar)*(D-y_bar))/sum((T-x_bar)**2)
            alpha_ = y_bar - beta_*x_bar
            rep = alpha_ + beta_*T[rwins-2]
            c_ins.replace(rep)
            r_ins.replace(rep)
    return anomaly_index












