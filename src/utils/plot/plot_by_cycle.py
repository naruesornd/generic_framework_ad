from matplotlib import pyplot as plt
from data_processor import CycleProcessor
import ipywidgets as widgets
from IPython.display import display

import numpy as np
def plot_by_cycle(df, column_name, plot_num=10):
    unique_cycles = df['cycle_id'].unique()
    unique_cycles = [c for c in unique_cycles if c != -1]

    if len(unique_cycles) > 10:
        np.random.seed(42)  # 可选：保证每次选到的周期一样，便于调试
        selected_cycles = np.random.choice(unique_cycles, size=plot_num, replace=False)
    else:
        selected_cycles = unique_cycles

    fig, axes = plt.subplots(plot_num, 1, figsize=(10, 20))

    for i,cycle_id in enumerate(selected_cycles):
        cycle_data = df[df['cycle_id'] == cycle_id][column_name]
        axes[i].plot(cycle_data)
        axes[i].set_title(f"Cycle {cycle_id}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel(column_name)
    plt.legend()
    plt.tight_layout()
    plt.show()


def interactive_cycle_plot(df, plot_func):
    # 筛选可选列（数值型）
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    column_selector = widgets.Dropdown(
        options=numeric_columns,
        description='Column:',
        value=numeric_columns[0]
    )

    plot_num_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=min(20, len(df['cycle_id'].unique()) - 1),
        step=1,
        description='Num of cycles:'
    )

    ui = widgets.VBox([column_selector, plot_num_slider])

    out = widgets.interactive_output(
        plot_func,
        {'df': widgets.fixed(df), 'column_name': column_selector, 'plot_num': plot_num_slider}
    )

    display(ui, out)