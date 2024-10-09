import numpy as np
import pandas as pd
from collections import Counter

def index_to_preds(index_val, length):
    preds = [0 for _ in range(length)]
    for i in index_val:
        preds[i]=1
    return preds


def multi_df(datasets,y_col):
    data_frames = []
    for i, df in enumerate(datasets):
        data_frames.append(df['value'].rename(f'X{i+1}'))
    new_df = pd.concat(data_frames, axis=1)
    anomaly_column = pd.DataFrame([df[y_col] for df in datasets]).any(axis=0).astype(int)
    new_df[y_col] = anomaly_column
    return new_df

def bernoulli(l):
    return np.random.choice([-1, 1], l, p=[0.5, 0.5])

def uniform(l):
    return np.random.uniform(-1, 1, l)

def linear(l):
    x = np.arange(1, l + 1)
    probs = np.linspace(0, 1, l)
    probs = probs / np.sum(probs)
    samples = np.random.choice(x / l, size=l, replace=True, p=probs)
    signs = bernoulli(l)
    return samples * signs

def quadratic(l):
    x = np.arange(1, l + 1)
    probs = np.linspace(0, 1, l)
    probs = (probs ** 2) / np.sum(probs ** 2)
    samples = np.random.choice(x / l, size=l, replace=True, p=probs)
    signs = bernoulli(l)
    return samples * signs

### Made to allow measuring of unlabeled anomaly datasets.
### Not cluster, just point anomalies.

def anomaly_injector(df,percent,std):
    L = len(df)
    y = [0 for _ in range(L)]
    temp_df = df.copy()
    Bernoulli = np.array(bernoulli(L))
    Linear = np.array(linear(L))
    Quadratic = np.array(quadratic(L))
    Uniform = np.array(uniform(L))
    randoms = np.random.uniform(0,1,L)
    for i in range(L):
        random_val = randoms[i]
        if random_val<=percent:
            y[i]=1
            # anom_type = np.random.randint(0,4,1)
            anom_type=3
            if anom_type==1:
                add_val = np.random.choice(Bernoulli,1)
            elif anom_type==2:
                add_val = np.random.choice(Linear,1)
            elif anom_type==3:
                add_val = np.random.choice(Quadratic,1)
            elif anom_type==4:
                add_val = np.random.choice(Uniform,1)
            temp_df.iloc[i] += add_val*std
    return temp_df, np.array(y)


def inject_then_sum(dfs,percent,std):
    num_dfs = len(dfs)
    cols = ['X'+str(i+1) for i in range(num_dfs)]
    cols.append('anomaly')
    return_df = pd.DataFrame(columns=cols)
    ys = [0 for _ in range(len(dfs[0]))]
    for idx,df in enumerate(dfs):
        anom_df,y = anomaly_injector(df,percent,std)
        return_df['X'+str(idx+1)] = anom_df
        ys = [a+b for a,b in zip(ys,y)]
    new_ys = [1 if x>0 else 0 for x in ys]
    return_df['anomaly']=new_ys
    return return_df



def sum_then_inject(dfs,percent,std):
    num_dfs = len(dfs)
    L = len(dfs[0])
    cols = ['X'+str(i+1) for i in range(num_dfs)]
    cols.append('anomaly')
    return_df = pd.DataFrame(columns=cols)
    randoms = np.random.uniform(0,1,L)

    for idx, df in enumerate(dfs):
        temp_df = df.copy()
        Bernoulli = np.array(bernoulli(L))
        Linear = np.array(linear(L))
        Quadratic = np.array(quadratic(L))
        Uniform = np.array(uniform(L))
        for i in range(L):
            random_val = randoms[i]
            if random_val<=percent:
                # anom_type = np.random.randint(0,4,1)
                anom_type=4
                if anom_type==1:
                    add_val = np.random.choice(Bernoulli,1)
                elif anom_type==2:
                    add_val = np.random.choice(Linear,1)
                elif anom_type==3:
                    add_val = np.random.choice(Quadratic,1)
                elif anom_type==4:
                    add_val = np.random.choice(Uniform,1)

                temp_df.iloc[i] += add_val*std*5
            # print(len(temp_df))
            return_df['X'+str(idx+1)] = temp_df

    y = (randoms < percent).astype(int)
    return_df['anomaly'] = y
    return return_df


def anom_replace(orig_data, anom_preds):
    new_df = orig_data.copy()
    for col in new_df.columns:
        col_data = new_df[col]
        for i, pred in enumerate(anom_preds):
            if pred == 1: 
                j = i + 1
                while j < len(anom_preds) and anom_preds[j] != 0:
                    j += 1  
                if i > 0 and j < len(col_data): 
                    mean_val = (col_data[i - 1] + col_data[j]) / 2
                    col_data = col_data.copy()
                    col_data[i] = mean_val
        new_df[col] = col_data
    return new_df


def convert_labels_to_int(y_data):
    counters = Counter(y_data)
    sorted_counters = [label for label, _ in counters.most_common()]
    new_dic = {label: i for i, label in enumerate(sorted_counters)}
    new_y = [new_dic[label] for label in y_data]
    return pd.Series(new_y)

def get_freq(df,time_col):
    return (df[time_col][len(df)-1]-df[time_col][0])/len(df)

def shorten_df(df,ratio):
    return df[::ratio].reset_index(drop=True)



# def average_according_to_dataset(df,col_name):
#     df['type'] = df[col_name].apply(lambda x:x.split("_")[0])
#     grouped_df = df.groupby('type').mean()
#     grouped_df = grouped_df.round(3)
#     return grouped_df

# def average_according_to_dataset_sequential(df, data_col, split_col):
#     df['type'] = df[data_col].apply(lambda x: x.split("_")[0])
#     grouped_df = df.groupby(['type', split_col]).mean()
#     grouped_df = grouped_df.round(3)
#     return grouped_df



def average_according_to_dataset(df,col_name):
    df['Type'] = df[col_name].apply(lambda x: x.split('_')[0])
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    agg_cols = [col for col in numeric_cols if col not in [col_name, 'Type']]
    agg_dict = {col: ['mean', 'std'] for col in agg_cols}
    grouped_df = df.groupby('Type').agg(agg_dict)
    grouped_df = grouped_df.round(3)
    new_columns = {}
    for col in agg_cols:
        combined_col = col + '_mean_std'
        grouped_df[combined_col] = grouped_df[col]['mean'].astype(str) + "_" + grouped_df[col]['std'].astype(str)
        new_columns[col] = combined_col
    grouped_df = grouped_df[list(new_columns.values())]
    return grouped_df

def average_according_to_dataset_sequential(df, data_col, split_col):
    df['type'] = df[data_col].apply(lambda x: x.split("_")[0])
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    agg_cols = [col for col in numeric_cols if col not in ['type', data_col, split_col]]
    agg_dict = {col: ['mean', 'std'] for col in agg_cols}
    grouped_df = df.groupby(['type', split_col]).agg(agg_dict)
    grouped_df = grouped_df.round(3)
    new_columns = {}
    for col in agg_cols:
        combined_col = col + '_mean_std'
        grouped_df[combined_col] = grouped_df[col]['mean'].astype(str) + "_" + grouped_df[col]['std'].astype(str)
        new_columns[col] = combined_col
    grouped_df = grouped_df[list(new_columns.values())]
    return grouped_df


