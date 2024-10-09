import numpy as np

def make_window(x, y, window_length, y_index=-1):
    x_len, x_cols = x.shape
    try : x = x.values
    except: pass
    windows = []
    if y_index == -1:
        y_index = window_length // 2
    for row_idx in range(x_len-window_length+1):
        windows.append(x[row_idx:row_idx+window_length].reshape(1, window_length, x_cols))
    return np.array(windows), y[window_length-y_index:-y_index].to_numpy()

def make_lin_window(x,y,window_length, y_index=-1):
    x_len, x_cols = x.shape
    try : x = x.values
    except: pass
    windows = []
    if y_index == -1:
        y_index = window_length // 2
    for row_idx in range(x_len-window_length+1):
        windows.append(x[row_idx:row_idx+window_length].flatten())
    return np.array(windows), y[window_length-y_index:x_len-y_index].to_numpy()

def shuffle_windows(windows, labels):
    perm = np.random.permutation(len(labels))
    shuffled_windows = windows[perm]
    shuffled_labels = labels[perm]
    return shuffled_windows, shuffled_labels

def split_train_test(windows, labels, train_ratio=0.8):
    split_idx = int(len(labels) * train_ratio)
    train_windows = windows[:split_idx]
    test_windows = windows[split_idx:]
    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]
    return train_windows, train_labels, test_windows, test_labels
