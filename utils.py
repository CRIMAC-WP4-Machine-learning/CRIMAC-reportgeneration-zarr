import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def process_report_csv(path_to_csv_report, path_to_STOX, survey_code):
    df = pd.read_csv(path_to_csv_report)

    # Reading transect information
    try:
        transect_information = pd.read_csv(f'{path_to_STOX}/PSUByTime.txt',
                                           sep="\t", quotechar='"')
    except:
        transect_information = pd.read_csv(f'{path_to_STOX}/{survey_code}_transects.csv')

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
    # Averaging over transect
    filtered_averaged_chunks = []

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

        # Compute the mean for numeric columns
        mean_numeric = filtered_chunk.select_dtypes(include='number').mean()

        # Combine results
        filtered_averaged_chunk = pd.DataFrame([mean_numeric], columns=mean_numeric.index)
        filtered_averaged_chunk['ping_start'] = filtered_chunk['ping_start'].min()
        filtered_averaged_chunk['ping_end'] = filtered_chunk['ping_end'].max()

        filtered_averaged_chunk['dist_start (nm)'] = filtered_chunk['dist_start (nm)'].min()
        filtered_averaged_chunk['dist_end (nm)'] = filtered_chunk['dist_end (nm)'].max()

        filtered_averaged_chunks.append(filtered_averaged_chunk)

    # Concatenate all filtered chunks into a single DataFrame
    filtered_data = pd.concat(filtered_chunks, ignore_index=True)
    filtered_data = filtered_data.sort_values(by='ping_start', ascending=True).reset_index(drop=True)

    # Group by the pair (ping_start, ping_end) and sum the sa_value
    result = filtered_data.groupby(['ping_start', 'ping_end'])['sa_value'].sum().reset_index()
    #result['sa_value'] = 10 * result['sa_value']

    # Concatenate all filtered chunks into a single DataFrame
    filtered_averaged_data = pd.concat(filtered_averaged_chunks, ignore_index=True)
    filtered_averaged_data = filtered_averaged_data.sort_values(by='ping_start', ascending=True).reset_index(drop=True)

    # Group by the pair (ping_start, ping_end) and sum the sa_value
    result_averaged = filtered_averaged_data.groupby(['ping_start', 'ping_end'])['sa_value'].sum().reset_index()
    #result_averaged['sa_value'] = 10 * result_averaged['sa_value']

    return result, result_averaged


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
        NASC_multiplier = 4 * np.pi * (1852 ** 2)
        ax.plot(df1["ping_start"], df1["sa_value"] * NASC_multiplier, linestyle="-", marker=".",
                markersize=4, label="Report 1", markevery=(df1["sa_value"] != 0))
        ax.plot(df2["ping_start"], df2["sa_value"] * NASC_multiplier, linestyle="-", marker=".",
                markersize=4, alpha=0.4, label=label, markevery=(df2["sa_value"] != 0))
        ax.set_title(title)
        ax.set_ylabel(r'NASC $(\mathrm{m}^2 \, \mathrm{nmi}^{-2})$')
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
    plt.close()
    print(f"Plot saved as {savename}")


def generate_PSU_transect_line_plots(result_1, result_2, result_3, result_4, savename, dpi=400):
    """
    Generates line plots comparing sa_values from multiple reports while skipping zero values.

    Inputs:
        - result_1, result_2, result_3, result_4: DataFrames with 'ping_start' and 'sa_value' columns.
        - savename: Path to save the output plot (as JPG).
        - dpi: Resolution for the saved figure.
    """

    # Nested function to plot with markers, skipping sa_value == 0
    def plot_with_markers(ax, df1, df2, title, label):
        NASC_multiplier = 4 * np.pi * (1852 ** 2)
        ax.plot(range(1, len(df1) + 1), df1["sa_value"] * NASC_multiplier, linestyle="-", marker=".",
                markersize=4, label="Report 1", markevery=(df1["sa_value"] != 0))
        ax.plot(range(1, len(df2) + 1), df2["sa_value"] * NASC_multiplier, linestyle="-", marker=".",
                markersize=4, alpha=0.4, label=label, markevery=(df2["sa_value"] != 0))
        ax.set_title(title)
        ax.set_ylabel(r'NASC $(\mathrm{m}^2 \, \mathrm{nmi}^{-2})$')
        ax.legend(loc='upper right')

    # Create figure with 3 rows and 1 column
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 15), sharex=True)

    # Plot Report 1 vs Report 2
    plot_with_markers(axes[0], result_1, result_2, "Report 1 vs Report 2", 'Report 2')

    # Plot Report 1 vs Report 3
    plot_with_markers(axes[1], result_1, result_3, "Report 1 vs Report 3", 'Report 3')

    # Plot Report 1 vs Report 4
    plot_with_markers(axes[2], result_1, result_4, "Report 1 vs Report 4", 'Report 4')
    axes[2].set_xlabel("PSU")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(savename, dpi=dpi, format='jpg')
    plt.close()
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
    plt.close()
    print(f"Plot saved as {savename}")


def generate_boxplot(result_1, result_2, result_3, result_4, savename):
    """
    Generates a single boxplot showing the errors of sa_values between
    result_2, result_3, and result_4 compared to result_1.

    Inputs:
        - result_1, result_2, result_3, result_4: DataFrames with 'sa_value' column.
        - savename: Path to save the output plot (as JPG).
        - dpi: Resolution for the saved figure.
    """
    # Calculate errors
    errors_2 = result_2["sa_value"] - result_1["sa_value"]
    errors_3 = result_3["sa_value"] - result_1["sa_value"]
    errors_4 = result_4["sa_value"] - result_1["sa_value"]

    # Combine errors into a list for boxplot
    errors = [errors_2, errors_3, errors_4]
    labels = ["Report 2 - Report 1", "Report 3 - Report 1", "Report 4 - Report 1"]

    # Create the boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(errors, labels=labels, patch_artist=True)
    plt.title("Error Distribution (sa_value Differences)")
    plt.ylabel("Error (sa_value)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the figure
    plt.savefig(savename, dpi=400, format='jpg')
    plt.close()
    print(f"Boxplot saved as {savename}")


def plot_worst_best_examples(sv, bottom, predictions_1, predictions_2, deneme_1, deneme_2, threshold_value, survey_code, name_, savename):
    # Averaging over 0.1 nmiles
    #deneme_1 = deneme_1.groupby(['ping_start', 'ping_end', 'dist_start (nm)', 'dist_end (nm)'],
    #                            as_index=False).sum('sa_value')
    #deneme_2 = deneme_2.groupby(['ping_start', 'ping_end', 'dist_start (nm)', 'dist_end (nm)'],
    #                            as_index=False).sum('sa_value')

    # Calculate absolute error
    deneme_1["absolute_error"] = (deneme_1["sa_value"] - deneme_2["sa_value"]).abs()

    # Keep sa_value columns from both DataFrames
    deneme_1["sa_value_new"] = deneme_2["sa_value"]  # Add sa_value column from deneme_2

    # Find top 5 maximum and minimum errors
    worst = deneme_1.nlargest(5, "absolute_error")
    ara = deneme_1[(deneme_1['sa_value'] > 10**-7) & (deneme_1['sa_value_new'] > 10**-7)]
    best = ara.nsmallest(5, "absolute_error")

    if name_ == 'best':
        selected_df = best
    elif name_ == 'worst':
        selected_df = worst

    for i in range(len(selected_df)):
        target_start_time = pd.to_datetime(selected_df.ping_start.values[i])  # Convert target ping_start to Timestamp
        target_end_time = pd.to_datetime(selected_df.ping_end.values[i])  # Convert target ping_end to Timestamp
        ping_times = pd.to_datetime(sv.ping_time.values)  # Convert sv.ping_time to datetime64[ns]

        # Find the index of the closest ping_start
        closest_start_index = np.abs(ping_times - target_start_time).argmin()
        closest_start_time = ping_times[closest_start_index]  # Get the corresponding ping_start time

        # Find the index of the closest ping_end
        closest_end_index = np.abs(ping_times - target_end_time).argmin()
        closest_end_time = ping_times[closest_end_index]  # Get the corresponding ping_end time

        ping_slice = slice(closest_start_index, closest_end_index)
        range_slice = slice(0, 500)

        sv_portion = sv.sv.sel(frequency=200000).isel(ping_time=ping_slice, range=range_slice)
        bottom_portion = bottom.bottom_range.isel(ping_time=ping_slice, range=range_slice)

        bottom_okuma_denemesi = bottom_portion.values
        seabed_slice_pad = np.zeros_like(bottom_okuma_denemesi).copy()
        seabed_slice_pad[:, 10:] = bottom_okuma_denemesi[:, :-10]
        bottom_portion_10_below = bottom_portion.copy()
        bottom_portion_10_below = bottom_portion_10_below.copy(data=seabed_slice_pad)

        # identifying the bottom line
        # Example input array
        data = bottom_portion.T.values
        first_one_mask = np.cumsum(data == 1, axis=0) == 1
        bottom_line = data * first_one_mask
        # Find the row index of the first 1 in each column
        first_one_indices = np.argmax(bottom_line == 1, axis=0)

        predictions_1_portion = predictions_1.annotation.sel(category=27).isel(ping_time=ping_slice, range=range_slice)
        predictions_2_portion_ = predictions_2.annotation.sel(category=27).isel(ping_time=ping_slice, range=range_slice)
        predictions_2_portion = (predictions_2_portion_.where(bottom_portion_10_below != 1) >= threshold_value)

        # Extracting the coordinate values
        ping_time = sv_portion.ping_time.values
        range_values = sv_portion.range.values

        y_coords = range_values[first_one_indices]

        # Plotting the figures in a single plot with 3 rows and 1 column
        fig, axs = plt.subplots(3, 1, figsize=(6, 12), sharex=True)

        # First plot
        im1 = axs[0].imshow(10 * np.log10(sv_portion).T, cmap='viridis', aspect='auto',
                            extent=[ping_time[0], ping_time[-1], range_values[-1], range_values[0]])

        axs[0].grid(False)
        axs[0].set_title(f"Sv in 200 kHz ({closest_start_time})\n {name_}{i + 1} ({survey_code})")
        axs[0].set_ylabel("range (m)")

        # Second plot
        im2 = axs[1].imshow(predictions_2_portion.T, cmap='seismic', vmin=0, vmax=1, aspect='auto',
                            extent=[ping_time[0], ping_time[-1], range_values[-1], range_values[0]])
        axs[1].plot(ping_time, y_coords, color='grey', linewidth=2, label='Bottom Line')  # bottom
        axs[1].grid(False)
        axs[1].set_title(f"Predictions (sa={selected_df['sa_value_new'].values[i]})")
        axs[1].set_ylabel("range (m)")

        # Third plot
        im3 = axs[2].imshow(predictions_1_portion.T, cmap='seismic', vmin=0, vmax=1, aspect='auto',
                            extent=[ping_time[0], ping_time[-1], range_values[-1], range_values[0]])
        axs[2].plot(ping_time, y_coords, color='grey', linewidth=2, label='Bottom Line')  # bottom
        axs[2].grid(False)
        axs[2].set_title(f"Labels (sa={selected_df['sa_value'].values[i]})")
        axs[2].set_xlabel("ping time")
        axs[2].set_ylabel("range (m)")

        plt.xticks(rotation=45)

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(f'{savename}_{name_}{i + 1}.jpg', dpi=400, format='jpg')
        plt.close()
        print(f"Plot saved as {savename}_{name_}{i + 1}")

        plt.show()

