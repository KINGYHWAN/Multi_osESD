import matplotlib.pyplot as plt
import os
import pandas as pd

def save_multivariate_plots(full_df,anoms,save_dir):

    # save_dir = "save"
    # labels = anoms
    for i in range(len(full_df.columns)):
        data = full_df.iloc[:,i]
        plt.figure(figsize=(15, 7))
        fig, ax = plt.subplots()
        new_df = pd.DataFrame({"timestamps":[i for i in range(len(data))],"value":data,"anomaly":anoms})
        l = ax.plot(new_df['timestamps'], new_df['value'])
        IS = new_df[new_df["anomaly"] > 0]
        ax.set_xlabel("Timestamps")
        ax.set_ylabel("Value")
        ax.set_title(str(i) + '_plot')
        ax.plot(IS['timestamps'], IS['value'], "ro", markersize=4)
        plt.show()
        save_path = os.path.join(save_dir, f"plot_{i}.png")
        plt.savefig(save_path)
        plt.close()






# def save_plot(data_path, column_name, df, save_path, model_name):
#     fig, ax = plt.subplots()
#     l = ax.plot(df['timestamps'], df['value'])
#     IS = df[df[column_name] > 0]
#     ax.plot(IS['timestamps'], IS['value'], "ro")
#     ax.set_xlabel("Timestamps")
#     ax.set_ylabel("Value")
#     ax.set_title(column_name + '_plot')
#     plt.savefig(fname=save_path + data_path[:-4] +'_'+ model_name + '.png')
#     plt.close()
#
#
#
# def plot_specific(df,x_col, plot_value):
#     plt.figure(figsize=(10, 6))
#     plt.plot(df.index, df[x_col], color='blue', label='Value')
#     plt.scatter(df.index[df['anomaly'] == plot_value], df[x_col][df['anomaly'] == plot_value],
#                 color='red', label='Anomaly', zorder=5)
#     plt.xlabel('Timestamp')
#     plt.ylabel('Value')
#     plt.title('Time Series with Anomalies')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
