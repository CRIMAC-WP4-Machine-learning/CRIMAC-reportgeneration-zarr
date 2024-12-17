import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def process_report_csv(path_to_csv_report, path_to_STOX):
    df = pd.read_csv(path_to_csv_report)

    # Reading transect information
    try:
        transect_information = pd.read_csv(f'{path_to_STOX}/PSUByTime.txt',
                                           sep="\t", quotechar='"')
    except:
        transect_information = pd.read_csv(f'{path_to_STOX}/2017843_transects.csv')

    df['ping_start'] = pd.to_datetime(df['ping_start'])
    df['ping_end'] = pd.to_datetime(df['ping_end'])
    transect_information['StartDateTime'] = pd.to_datetime(transect_information['StartDateTime'])
    transect_information['StopDateTime'] = pd.to_datetime(transect_information['StopDateTime'])

    # Convert StartDateTime and StopDateTime to naive datetime (removing timezone)
    transect_information['StartDateTime'] = transect_information['StartDateTime'].dt.tz_localize(None)
    transect_information['StopDateTime'] = transect_information['StopDateTime'].dt.tz_localize(None)

    # Ensure deneme datetime columns are also timezone-naive
    df['ping_start'] = pd.to_datetime(df['ping_start']).dt.tz_localize(None)
    df['ping_end'] = pd.to_datetime(df['ping_end']).dt.tz_localize(None)

    # Initialize an empty list to collect filtered chunks
    filtered_chunks = []

    # Loop over smaller transect_information DataFrame
    for _, row in transect_information.iterrows():
        start_time = row['StartDateTime']
        stop_time = row['StopDateTime']

        # Filter 'deneme' where ping_start and ping_end fall within the time range
        filtered_chunk = df[
            (df['ping_start'] >= start_time) &
            (df['ping_end'] <= stop_time)
            ]

        # Append the filtered chunk to the list
        filtered_chunks.append(filtered_chunk)

    # Concatenate all filtered chunks into a single DataFrame
    filtered_data = pd.concat(filtered_chunks, ignore_index=True)
    filtered_data = filtered_data.sort_values(by='ping_start', ascending=True).reset_index(drop=True)

    # Group by the pair (ping_start, ping_end) and sum the sa_value
    result = filtered_data.groupby(['ping_start', 'ping_end'])['sa_value'].sum().reset_index()
    result['sa_value'] = 10 * result['sa_value']

    return result


def generate_line_plots(result_1, result_2, result_3, result_4, savename, dpi=400):
    """
    Generates line plots comparing sa_values from multiple reports while skipping zero values.

    Inputs:
        - result_1, result_2, result_3, result_4: DataFrames with 'ping_start' and 'sa_value' columns.
        - savename: Path to save the output plot (as JPG).
        - dpi: Resolution for the saved figure.
    """

    # Nested function to plot with markers, skipping sa_value == 0
    def plot_with_markers(ax, df1, df2, title, label):
        ax.plot(df1["ping_start"], df1["sa_value"], linestyle="-", marker=".",
                markersize=4, label="Report 1", markevery=(df1["sa_value"] != 0))
        ax.plot(df2["ping_start"], df2["sa_value"], linestyle="-", marker=".",
                markersize=4, alpha=0.4, label=label, markevery=(df2["sa_value"] != 0))
        ax.set_title(title)
        ax.set_ylabel("sa value")
        ax.legend(loc='upper right')

    # Create figure with 3 rows and 1 column
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 15), sharex=True)

    # Plot Report 1 vs Report 2
    plot_with_markers(axes[0], result_1, result_2, "Report 1 vs Report 2", 'Report 2')

    # Plot Report 1 vs Report 3
    plot_with_markers(axes[1], result_1, result_3, "Report 1 vs Report 3", 'Report 3')

    # Plot Report 1 vs Report 4
    plot_with_markers(axes[2], result_1, result_4, "Report 1 vs Report 4", 'Report 4')
    axes[2].set_xlabel("ping time")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(savename, dpi=dpi, format='jpg')
    plt.show()
    print(f"Plot saved as {savename}")


def generate_hexbin_plots(result_1, result_2, result_3, result_4, savename, dpi=400):
    """
    Generates scatter and hexbin density plots comparing sa_values from multiple reports.
    Adds a y = x reference line and computes R² values.

    Inputs:
        - result_1, result_2, result_3, result_4: DataFrames with 'sa_value' column.
        - savename: Path to save the output plot (as JPG).
        - dpi: Resolution for the saved figure.
    """

    # Nested function to create hexbin density plots
    def hexbin_density(ax, x, y, title, cmap="viridis"):
        hb = ax.hexbin(x, y, gridsize=75, cmap=cmap, mincnt=1, norm=colors.LogNorm())
        ax.set_title(title)
        ax.set_ylabel("sa_value (label)")
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label("log(Counts)")

    # Function to add a y = x line
    def add_y_equals_x_line(ax, x, y):
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', linewidth=1)

    # Function to compute R²
    def compute_r2(x, y):
        return r2_score(x, y)

    # Create figure with 3 rows and 2 columns
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 18), sharex=True, sharey=True)

    # First Row: Report 1 vs Report 2
    r2_2 = compute_r2(result_1["sa_value"], result_2["sa_value"])
    axes[0, 0].scatter(result_1["sa_value"], result_2["sa_value"], s=10, alpha=0.5, color='orange')
    axes[0, 0].set_title(f"Report 1 vs Report 2 (Scatter)\nR²: {r2_2:.3f}")
    axes[0, 0].set_ylabel("sa_value (Report 2)")
    add_y_equals_x_line(axes[0, 0], result_1["sa_value"], result_2["sa_value"])
    hexbin_density(axes[0, 1], result_1["sa_value"], result_2["sa_value"], "Report 1 vs Report 2 (Density)", cmap='jet')

    # Second Row: Report 1 vs Report 3
    r2_3 = compute_r2(result_1["sa_value"], result_3["sa_value"])
    axes[1, 0].scatter(result_1["sa_value"], result_3["sa_value"], s=10, alpha=0.5, color="orange")
    axes[1, 0].set_title(f"Report 1 vs Report 3 (Scatter)\nR²: {r2_3:.3f}")
    axes[1, 0].set_ylabel("sa_value (Report 3)")
    add_y_equals_x_line(axes[1, 0], result_1["sa_value"], result_3["sa_value"])
    hexbin_density(axes[1, 1], result_1["sa_value"], result_3["sa_value"], "Report 1 vs Report 3 (Density)", cmap="jet")

    # Third Row: Report 1 vs Report 4
    r2_4 = compute_r2(result_1["sa_value"], result_4["sa_value"])
    axes[2, 0].scatter(result_1["sa_value"], result_4["sa_value"], s=10, alpha=0.5, color="orange")
    axes[2, 0].set_title(f"Report 1 vs Report 4 (Scatter)\nR²: {r2_4:.3f}")
    axes[2, 0].set_ylabel("sa_value (Report 4)")
    axes[2, 0].set_xlabel("sa_value (Report 1)")
    add_y_equals_x_line(axes[2, 0], result_1["sa_value"], result_4["sa_value"])
    hexbin_density(axes[2, 1], result_1["sa_value"], result_4["sa_value"], "Report 1 vs Report 4 (Density)", cmap="jet")
    axes[2, 1].set_xlabel("sa_value (Report 1)")

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(savename, dpi=dpi, format='jpg')
    plt.show()
    print(f"Plot saved as {savename}")
