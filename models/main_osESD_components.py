import numpy as np
import scipy.stats as stats

class TRES:
    def __init__(self, data, time=None, wins=None):
        self.data = data[:wins]
        self.time = list(range(1, wins+1)) if time is None else time[:wins]
        self.original_data = data
        self.original_time = time
        self.tres = []
        self.x_bar = np.mean(self.time)
        self.y_bar = np.mean(self.data)
        self.wins = wins
        self._initialize()

    def _initialize(self):
        beta = sum([(self.time[i]-self.x_bar)*(self.data[i]-self.y_bar) for i in range(self.wins)]) / sum([(t-self.x_bar)**2 for t in self.time])
        alpha = self.y_bar - beta * self.x_bar
        self.tres.append(self.data[self.wins-1] - (alpha + beta * self.time[self.wins-1]))
        for i in range(self.wins, len(self.original_data)):
            self.data.pop(0)
            self.time.pop(0)
            # print(self.data)
            # print(self.original_data[i])
            self.data.append(self.original_data[i])
            self.time.append(self.original_time[i])
            self.x_bar -= (self.original_time[i-self.wins] - self.original_time[i]) / self.wins
            self.y_bar -= (self.original_data[i-self.wins] - self.original_data[i]) / self.wins
            beta = sum([(self.time[j]-self.x_bar)*(self.data[j]-self.y_bar) for j in range(self.wins)]) / sum([(t-self.x_bar)**2 for t in self.time])
            alpha = self.y_bar - beta * self.x_bar
            self.tres.append(self.data[-1] - (alpha + beta * self.time[-1]))

    def update(self, ond, ont=None):
        first_data = self.data.pop(0)
        first_time = self.time.pop(0)
        self.data.append(ond)
        self.time.append(self.time[-1]+1 if ont is None else ont)
        self.x_bar -= (first_time - self.time[-1]) / self.wins
        self.y_bar -= (first_data - ond) / self.wins
        beta = sum([(self.time[i]-self.x_bar)*(self.data[i]-self.y_bar) for i in range(self.wins)]) / sum([(t-self.x_bar)**2 for t in self.time])
        alpha = self.y_bar - beta * self.x_bar
        tres_ = ond - (alpha + beta * self.time[-1])
        # F = self.tres.pop(0)
        self.tres.append(tres_)
        return tres_

    def replace(self,rep):
        prev = self.data[self.wins-1]
        self.data[self.wins-1] = rep
        self.y_bar -= (prev-rep)/self.wins


class TCHA:
    def __init__(self, data, wins , time=None):
        if time is None:
            time = list(range(1,len(data)+1))
        self.data = data[(len(data) - wins):]
        self.time = time[(len(time) - wins):]
        tcha_data = []
        for x,y in zip(data[wins-1:],data[:len(data)-wins+1]):
            tcha_data.append(x-y)
        tcha_time = []
        for x,y in zip(time[wins-1:],time[:len(time)-wins+1]):
            tcha_time.append(x-y)
        tcha=[]
        for x,y in zip(tcha_data,tcha_time):
            tcha.append(x/y)
        self.tcha = tcha
        self.wins = wins

    def update(self, ond, ont=None):
        if ont is None:
            ont = self.time[self.wins] + 1
        self.data = self.data[1:] + [ond]
        self.time = self.time[1:] + [ont]
        tcha_ = (self.data[self.wins-1] - self.data[0]) / (self.time[self.wins-1] - self.time[0])
        self.tcha = self.tcha[1:] + [tcha_]
        return tcha_

    def replace(self, rep):
        self.data[self.wins - 1] = rep



class SESD_tres:
    def __init__(self, data=None, alpha=0.01, maxr=10):
        self.mean = 0
        self.sqsum = 0
        self.alpha = alpha
        self.maxr = maxr
        self.data = data
        self.size = len(data)
        self.mean = np.mean(data)
        self.sqsum = np.sum(np.square(data))
        lambdas = [0,0,0]
        for i in range(3,len(data)+1):
            lambdas.append(self.get_lambda(alpha, i))
        self.lambdas = lambdas

    def test(self, on):
        out = self.data[0]
        self.data = np.append(self.data[1:], on)
        self.mean += -(out - on) / self.size
        self.sqsum += (-out ** 2) + (on ** 2)
        mean_ = self.mean
        sqsum_ = self.sqsum
        size_ = self.size
        data_ = self.data
        sd_ = np.sqrt((sqsum_ - size_ * (mean_ ** 2) + 1e-8) / (size_ - 1))
        ares = np.abs((data_ - mean_) / sd_)
        esd_index = np.argmax(ares)
        esd = ares[esd_index]
        # print(self.lambdas[size_])
        try:
            # if esd > self.get_lambda(self.alpha,size_):
            if esd > self.lambdas[size_]:
                if esd_index == size_ - 1:
                    return esd
            else:
                # print(1)
                return 0
        except:
            return 0

        for i in range(2, self.maxr + 1):
            # print(i)
            size_ -= 1
            mean_ = ((size_ + 1) * mean_ - data_[esd_index]) / size_
            sqsum_ -= data_[esd_index] ** 2
            sd_ = np.sqrt((sqsum_ - size_ * mean_ ** 2 + 1e-8) / (size_ - 1))

            data_ = np.delete(data_, esd_index)
            ares = np.abs((data_ - mean_) / sd_)
            esd_index = np.argmax(ares)
            esd = ares[esd_index]
            try:
                # if esd > self.get_lambda(self.alpha,size_):
                if esd > self.lambdas[size_]:
                    if esd_index == size_ - 1:
                        return esd
                else:
                    # print(1)
                    return 0
            except:
                return 0
        return 0

    def get_lambda(self, alpha, size):
        t = stats.t.ppf(1 - alpha / (2 * size), size - 2)
        lmbda = t * (size - 1) / np.sqrt((size + t ** 2) * size)
        return lmbda


class SESD_tcha:
    def __init__(self, data=None, alpha=0.01, maxr=10):
        self.mean = 0
        self.sqsum = 0
        self.alpha = alpha
        self.maxr = maxr
        self.data = data
        self.size = len(data)
        self.mean = np.mean(data)
        self.sqsum = np.sum(np.square(data))
        lambdas = [0,0,0]
        for i in range(3,len(data)+1):
            lambdas.append(self.get_lambda(alpha, i))
        self.lambdas = lambdas

    def test(self, on):
        out = self.data[0]
        self.data = np.append(self.data[1:], on)
        self.mean += -(out - on) / self.size
        self.sqsum += (-out**2) + (on**2)
        mean_ = self.mean
        sqsum_ = self.sqsum
        size_ = self.size
        data_ = self.data
        sd_ = np.sqrt((sqsum_ - size_ * (mean_**2) + 1e-8)/(size_-1))
        ares = np.abs((data_-mean_)/sd_)
        esd_index = np.argmax(ares)
        esd = ares[esd_index]
        try:
            # if esd > self.get_lambda(self.alpha,size_):
            if esd > self.lambdas[size_]:
                if esd_index == size_ - 1:
                    return esd
            else:
                # print(1)
                return 0
        except:
            return 0

        for i in range(2, self.maxr+1):
            # print(i)
            size_ -= 1
            mean_ = ((size_+1)*mean_ - data_[esd_index])/size_
            sqsum_ -= data_[esd_index]**2
            sd_ = np.sqrt((sqsum_ - size_ * mean_**2 + 1e-8)/(size_-1))
            data_ = np.delete(data_, esd_index)
            ares = np.abs((data_-mean_)/sd_)
            esd_index = np.argmax(ares)
            esd = ares[esd_index]
            try:
                # if esd > self.get_lambda(self.alpha,size_):
                if esd > self.lambdas[size_]:
                    if esd_index == size_ - 1:
                        return esd
                else:
                    # print(1)
                    return 0
            except:
                return 0
        # return False
        return 0

    def get_lambda(self, alpha, size):
        t = stats.t.ppf(1 - alpha / (2 * size), size - 2)
        lmbda = t * (size - 1) / np.sqrt((size + t ** 2) * size)
        return lmbda


class osESD:
    def __init__(self, data, dwins, rwins, init_size, alpha, maxr, condition, time=None ):

        if time is None:
            time = list(range(1, len(data) + 1))

        self.init_data = data[:init_size]
        self.online_data = data[init_size:]
        self.init_time = time[:init_size]
        self.online_time = time[init_size:]

        self.dwins = dwins
        self.rwins = rwins
        self.init_size = init_size
        self.alpha = alpha
        self.maxr = maxr
        self.condition = condition
        self.initiate()

    def initiate(self):
        c_ins = TCHA(data = self.init_data, time = self.init_time, wins = self.dwins)
        r_ins = TRES(data = self.init_data, time = self.init_time, wins = self.rwins)
        # print(c_ins.data)
        # print(r_ins.data)
        self.SESD_TCHA = SESD_tcha(data = c_ins.tcha.copy(), alpha = self.alpha, maxr = self.maxr)
        self.SESD_TRES = SESD_tres(data = r_ins.tres.copy(), alpha = self.alpha, maxr = self.maxr)
        self.c_ins = c_ins
        self.r_ins = r_ins

    def test_values(self,idx):
        c_val = self.SESD_TCHA.test(self.c_ins.update(self.online_data[idx], self.online_time[idx]))
        r_val = self.SESD_TRES.test(self.r_ins.update(self.online_data[idx], self.online_time[idx]))
        c_anom = 0 if c_val == 0 else 1
        r_anom = 0 if r_val == 0 else 1
        return c_val, r_val, c_anom, r_anom

    def check_values(self, c_anom, r_anom):
        if self.condition: function_ = (c_anom and r_anom)
        else: function_ = (c_anom or r_anom)
        if function_ :
            # anomaly_index.append(i+self.train_size)
            D = self.r_ins.data.copy()
            T = self.r_ins.time.copy()
            del D[self.rwins-1]
            del T[self.rwins-1]
            x_bar = ((self.rwins*self.r_ins.x_bar) - self.r_ins.time[self.rwins-1]) / (self.rwins-1)
            y_bar = ((self.rwins*self.r_ins.y_bar) - self.r_ins.data[self.rwins-1]) / (self.rwins-1)
            beta_ = sum((T-x_bar)*(D-y_bar))/sum((T-x_bar)**2)
            alpha_ = y_bar - beta_*x_bar
            rep = alpha_ + beta_*T[self.rwins-2]
            self.c_ins.replace(rep)
            self.r_ins.replace(rep)
            return 1
        return 0

    def predict_idx(self,idx): ### index is based on online_data! not total data
        canom = self.SESD_TCHA.test(self.c_ins.update(self.online_data[idx], self.online_time[idx]))
        ranom = self.SESD_TRES.test(self.r_ins.update(self.online_data[idx], self.online_time[idx]))
        if self.condition: function_ = (canom and ranom)
        else: function_ = (canom or ranom)
        if function_ :
            # anomaly_index.append(i+self.train_size)
            D = self.r_ins.data.copy()
            T = self.r_ins.time.copy()
            del D[self.rwins-1]
            del T[self.rwins-1]
            x_bar = ((self.rwins*self.r_ins.x_bar) - self.r_ins.time[self.rwins-1]) / (self.rwins-1)
            y_bar = ((self.rwins*self.r_ins.y_bar) - self.r_ins.data[self.rwins-1]) / (self.rwins-1)
            beta_ = sum((T-x_bar)*(D-y_bar))/sum((T-x_bar)**2)
            alpha_ = y_bar - beta_*x_bar
            rep = alpha_ + beta_*T[self.rwins-2]
            self.c_ins.replace(rep)
            self.r_ins.replace(rep)
            return 1
        return 0


    def predict_all(self):
        anomaly_index = []
        for i in range(len(self.online_data)):
            canom = self.SESD_TCHA.test(self.c_ins.update(self.online_data[i], self.online_time[i]))
            ranom = self.SESD_TRES.test(self.r_ins.update(self.online_data[i], self.online_time[i]))
            if self.condition: function_ = (canom and ranom)
            else: function_ = (canom or ranom)
            if function_ :
                anomaly_index.append(i+self.init_size)
                D = self.r_ins.data.copy()
                T = self.r_ins.time.copy()
                del D[self.rwins-1]
                del T[self.rwins-1]
                x_bar = ((self.rwins*self.r_ins.x_bar) - self.r_ins.time[self.rwins-1]) / (self.rwins-1)
                y_bar = ((self.rwins*self.r_ins.y_bar) - self.r_ins.data[self.rwins-1]) / (self.rwins-1)
                beta_ = sum((T-x_bar)*(D-y_bar))/sum((T-x_bar)**2)
                alpha_ = y_bar - beta_*x_bar
                rep = alpha_ + beta_*T[self.rwins-2]
                self.c_ins.replace(rep)
                self.r_ins.replace(rep)
        return (anomaly_index)

