import s3fs
import xarray as xr
import numpy as np
import pandas as pd

# Reads files directly from S3
host = 'https://s3.hi.no'
access_key = 'crimac'
secret_key = '9!%L*h7Q'
bucketname = 'crimac-scratch'
region = 'us-east-1'

fs = s3fs.S3FileSystem(
    key=access_key,
    secret=secret_key,
    client_kwargs={
        'endpoint_url': host,
        'region_name': region})

# Surveys and years arrays
years = [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
survey_codes = [2007205, 2008205, 2009107, 2010205, 2011206, 2013842, 2014807, 2015837, 2016837, 2017843, 2018823,
                2019847, 2020821]

for k in range(len(years)):

    # Data Array Definitions & Filtering Based on Labels

    print(f'Calculating the Stox Input for survey = {survey_codes[k]} ...')

    base_uri = f's3://crimac-scratch/gpfs0-crimac-scratch/{years[k]}/S{survey_codes[k]}/ACOUSTIC'
    s3_sv_uri = base_uri + f'/GRIDDED/S{survey_codes[k]}_sv.zarr'
    labels_uri = base_uri + f'/GRIDDED/S{survey_codes[k]}_labels.zarr'

    # Load data from the server
    sv_s3 = s3fs.S3Map(s3_sv_uri, s3=fs)
    sv_data_from_server = xr.open_zarr(sv_s3)
    sv_data = sv_data_from_server.sv.sel(frequency=38000).drop('raw_file')

    # Load labels from the server
    labels_s3 = s3fs.S3Map(labels_uri, s3=fs)
    labels_from_server = xr.open_zarr(labels_s3)

    # TODO: FOR PREDICTION FROM ML MODELS: REPLACE labels_sandeel AS THE BINARY PREDICTIONS FROM THE ML MODEL
    labels_sandeel = labels_from_server.annotation.sel(category=27)

    # TODO: TALK TO NILS OLAV: PING TIMES ARE NOT THE SAME AS IN LABELS AS IN SV DATA!
    labels_sandeel['ping_time'] = sv_data.ping_time.values

    # Filtering on labels
    filtered_sv_data = sv_data.where(labels_sandeel == 1.0)

    # Distance Calculation

    distance_array = sv_data_from_server.distance.values

    # Initialize variables for labeling each segment based on 0.1 increments
    labels = np.zeros(len(distance_array), dtype=int)  # Array to hold labels
    current_label = 1  # Start labeling from 1
    start_index = 0  # Starting index for each new segment

    # Loop until all elements are labeled
    while start_index < len(distance_array):
        cumulative_increment = 0  # Reset cumulative increment for the new segment

        # Find the end of the current segment
        for i in range(start_index + 1, len(distance_array)):
            cumulative_increment += distance_array[i] - distance_array[i - 1]

            # If cumulative increment reaches approximately 0.1, label the segment
            if cumulative_increment >= 0.1:
                labels[start_index:i] = current_label  # Label all elements in this segment
                current_label += 1  # Move to the next label for the next segment
                start_index = i  # Start the next segment from this point
                break
        else:
            # If the loop completes without reaching 0.1, label remaining elements
            labels[start_index:] = current_label
            break

    # Calculation of start/stop values for distances, ping_time, latitude, and longitude

    # Definition of Arrays
    ping_time_array = sv_data.ping_time.values
    latitude_array = sv_data.latitude.values
    longitude_array = sv_data.longitude.values

    # Calculating start and end indices
    unique_labels, start_indices = np.unique(labels, return_index=True)
    end_indices = np.r_[start_indices[1:] - 1, len(labels) - 1]

    # Creating final arrays
    start_distances = distance_array[start_indices]
    end_distances = distance_array[end_indices]

    start_pings = ping_time_array[start_indices]
    end_pings = ping_time_array[end_indices]

    start_latitudes = latitude_array[start_indices]
    end_latitudes = latitude_array[end_indices]

    start_longitudes = longitude_array[start_indices]
    end_longitudes = longitude_array[end_indices]

    # Calculating average range resolution and number of range bins to sum for each unit

    average_range_res = np.diff(sv_data_from_server.range.values).mean()
    n_range_to_sum = int(np.round(10 / average_range_res))
    # Summing on range axis
    rescaled_range = filtered_sv_data.coarsen(range=n_range_to_sum, boundary='pad').sum(skipna=True)

    # Calculating start and end range values for each bin

    # Original range values
    original_range = filtered_sv_data["range"].values

    # Compute the start and end indices for each new bin
    start_ranges = original_range[::n_range_to_sum]  # Start values
    end_ranges = original_range[n_range_to_sum - 1::n_range_to_sum]  # End values

    # Handle any partial bins at the end (padding due to 'boundary="pad"')
    if len(start_ranges) > len(end_ranges):
        end_ranges = np.append(end_ranges, original_range[-1])

    # Averaging on custom distance bins
    print(f'Calculating the averages along custom distance bins for survey = {survey_codes[k]} ...')
    rescaled_dist = rescaled_range.assign_coords(segment=('ping_time', labels))  # assigning the coordinate
    averaged_sv_data = rescaled_dist.groupby('segment').mean(dim='ping_time')  # groupby to average

    ##### Calculating the final Array for StoX

    print(f'Reading the final calculated sa array for survey = {survey_codes[k]} ...')
    nupy_averaged_sv_data = averaged_sv_data.values.T

    # TODO: TALK TO ARNE JOHANNES - MULTIPLYING WITH RANGE RESOLUTION!
    nupy_averaged_sv_data = average_range_res * nupy_averaged_sv_data

    print(f'Calculating the final array for StoX in survey = {survey_codes[k]} ...')

    ############### Arranging the final array for StoX Input ##################
    # Create a grid of indices for rows and columns
    n_rows, n_cols = nupy_averaged_sv_data.shape
    row_indices, col_indices = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing='ij')

    # Create arrays for each variable with broadcasting
    distance_start = np.repeat(start_distances, n_rows).reshape(n_cols, n_rows).T
    distance_end = np.repeat(end_distances, n_rows).reshape(n_cols, n_rows).T

    ping_start = np.repeat(start_pings, n_rows).reshape(n_cols, n_rows).T
    ping_end = np.repeat(end_pings, n_rows).reshape(n_cols, n_rows).T

    longitude_start = np.repeat(start_longitudes, n_rows).reshape(n_cols, n_rows).T
    longitude_end = np.repeat(end_longitudes, n_rows).reshape(n_cols, n_rows).T

    latitude_start = np.repeat(start_latitudes, n_rows).reshape(n_cols, n_rows).T
    latitude_end = np.repeat(end_latitudes, n_rows).reshape(n_cols, n_rows).T

    range_start = np.tile(start_ranges, (n_cols, 1)).T
    range_end = np.tile(end_ranges, (n_cols, 1)).T

    # Flatten the data arrays
    flattened_data = {
        'dist_start (nm)': distance_start.flatten(),
        'dist_end (nm)': distance_end.flatten(),
        'ping_start': ping_start.flatten(),
        'ping_end': ping_end.flatten(),
        'longitude_start': longitude_start.flatten(),
        'longitude_end': longitude_end.flatten(),
        'latitude_start': latitude_start.flatten(),
        'latitude_end': latitude_end.flatten(),
        'range_start (m)': range_start.flatten(),
        'range_end (m)': range_end.flatten(),
        'sa_value': nupy_averaged_sv_data.flatten(),
    }

    # Create the DataFrame directly from the flattened arrays
    df_final = pd.DataFrame(flattened_data)

    df_final.to_csv(f'{survey_codes[k]}_StoX_output.csv', index=False)
    print(f'StoX input array for survey = {survey_codes[k]} is saved!')
