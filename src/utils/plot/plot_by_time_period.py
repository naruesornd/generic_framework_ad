import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
class PlotByTimePeriod:
    def __init__(self, df):
        self.df = df
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y%m%d %H:%M:%S")


    def plot_by_month(self,month, column_name):
        df = self.df
        df['Month'] = df['timestamp'].dt.to_period('M').astype(str)
        df_filtered = df[df['Month'].isin(month)]
        #sort by timestamp
        df_filtered = df_filtered.sort_values(by='timestamp')

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        for i, month in enumerate(month):
            df_month = df_filtered[df_filtered['Month'] == month]

            axes[i].plot(df_month['timestamp'], df_month[column_name], color='tab:green')
            axes[i].set_title(f'{month} - {column_name}')
            axes[i].set_ylabel(column_name)
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            axes[i].grid(True)
        # 总体图修饰
        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        plt.show()


