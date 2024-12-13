import os
import xarray as xr
import numpy as np
import pandas as pd
import time

# Script to generate the reports from the sv+label for the Sand eel series

# Threshold Definitions

thresholds = {
    "report_1": 1.0,
    "report_2": 0.951172,
    "report_3": 0.917090,
    "report_4": 0.900195,
}

#crimacscratch = os.getenv('CRIMACSCRATCH')
crimacscratch = '/data/crimac-scratch/'
dataout = '/data/crimac-scratch/staging/'

# Sand eel surveys
cs = ['2005205', '2006207', '2007205', '2008205', '2009107',
      '2010205', '2011206', '2013842', '2014807', '2015837',
      '2016837', '2017843', '2018823', '2019847', '2020821',
      '2021847', '2022206', '2022611', '2023006009', '2024002006']

# Predictions/labels vs reports
pr = [['labels.zarr', 'report_1.zarr'],
      ['predictions_2.zarr', 'report_2.zarr'],
      ['predictions_3.zarr', 'report_3.zarr'],
      ['predictions_4.zarr', 'report_4.zarr']]


def generate_report(sv_zarr, labels_zarr, bottom_zarr, threshold,
                    report_csv_save_directory, sa_zarr_save_directory):
    """
  Generate the input csv file for StoX and corresponding sa.zarr file by processing sv data and labels/or predictions.

  Parameters:
  sv_zarr (zarr): Zarr data containing sv data.
  labels_zarr (zarr): Zarr data containing labels or predictions.
  bottom_zarr (stzarrr): Zarr data containing bottom data.
  threshold (float): Threshold value for filtering based on labels/predictions.For labels, it is 1, for predictions,
                     it is the value maximizing F1 score on training data.
  csv_savename (str): Name of the CSV file to save the final StoX input data.
  sa_zarr_save_directory (str): Name of the Zarr file to save the averaged SV data.

  Outputs:
  - A Zarr file containing the final sa data.
  - A CSV file containing the StoX-compatible input data.

  Description:
  This function loads sv, label(or predictions), and bottom data, filters the SV data based on labels and seabed
  proximity, and performs distance-based averaging and range bin summation. It calculates metrics such as start and
  end distances, pings, latitudes, longitudes, and range values for each distance bin (0.1 nm). The averaged sv data
  is saved as a Zarr file, and a StoX input CSV file is generated with all the required fields for further analysis.
  """

    start_time = time.time()

    # Selecting sv data and labels arrays from zarr
    labels_sandeel = labels_zarr.annotation.sel(category=27)
    sv_data = sv_zarr.sv.sel(frequency=38000).drop('raw_file')
    bottom_data = bottom_zarr.bottom_range.drop('frequency')

    # TODO: TALK TO NILS OLAV: PING TIMES ARE NOT THE SAME AS IN LABELS AS IN SV DATA!
    labels_sandeel['ping_time'] = sv_data.ping_time.values
    bottom_data['ping_time'] = sv_data.ping_time.values

    # Filtering on labels
    filtered_sv_data = sv_data.where(labels_sandeel >= threshold)

    # Filtering on bottom (10 pixels below)
    print(f'Filtering the data on bottom...')
    # Reading bottom data
    bottom_okuma_denemesi = bottom_data.values
    seabed_pad = 10
    seabed_slice_pad = np.zeros_like(bottom_okuma_denemesi).copy()
    seabed_slice_pad[:, seabed_pad:] = bottom_okuma_denemesi[:, :-seabed_pad]
    deneme = bottom_data.copy()
    deneme = deneme.copy(data=seabed_slice_pad)  # Final shifted bottom data - below bottom is 1

    # Filtering on bottom
    filtered_sv_data = filtered_sv_data.where(deneme != 1.0)

    # Distance Calculation

    distance_array = sv_zarr.distance.values

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

    average_range_res = np.diff(sv_zarr.range.values).mean()
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
    print(f'Calculating the averages along custom distance bins...')
    rescaled_dist = rescaled_range.assign_coords(segment=('ping_time', labels))  # assigning the coordinate
    averaged_sv_data = rescaled_dist.groupby('segment').mean(dim='ping_time')  # groupby to average

    ##### Calculating the final Array for StoX

    print(f'Reading the final calculated sa array...')
    nupy_averaged_sv_data = averaged_sv_data.values.T

    # TODO: TALK TO ARNE JOHANNES - MULTIPLYING WITH RANGE RESOLUTION!
    nupy_averaged_sv_data = average_range_res * nupy_averaged_sv_data

    # Saving final sa values

    # Create a new xarray Dataset with only segment and range coordinates
    averaged_sv_data_ds = xr.Dataset(
        {
            "averaged_sv_data": (("segment", "range"), nupy_averaged_sv_data.T)
        },
        coords={
            "segment": averaged_sv_data.coords["segment"],
            "range": averaged_sv_data.coords["range"],
        }
    )

    averaged_sv_data_ds = averaged_sv_data_ds.drop_vars(["channel_id", "frequency", "category"])

    # Chunk the dataset for better Zarr compatibility
    averaged_sv_data_ds = averaged_sv_data_ds.chunk({"segment": 1000, "range": 20})

    # Save as Zarr
    averaged_sv_data_ds.to_zarr(
        f'{sa_zarr_save_directory}',
        mode="w",
        encoding={"averaged_sv_data": {"chunks": (1000, 20)}}
    )

    print(f'Calculating the final array for StoX in survey...')

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

    df_final.to_csv(f'{report_csv_save_directory}', index=False)

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f'StoX input array for survey is saved!')
    print(f"Execution time: {execution_time_minutes:.2f} minutes")


# This function prioritizes the staging data over the production data
def prodstage(crimacscratch, _cs, zarrstore):
    prod = os.path.join(crimacscratch, _cs[0:4], 'S' + _cs, zarrstore)
    staging = os.path.join(crimacscratch, 'staging', _cs[0:4],
                           'S' + _cs, zarrstore)
    if os.path.exists(staging):
        d = staging
    elif os.path.exists(prod):
        d = prod
    else:
        d = None
    return d


# This is Ahmet's playground:
def runcruise(cruise):
    try:
        sv_zarr = xr.open_zarr(cruise['sv_file'])
        label_zarr = xr.open_zarr(cruise['pred_file'])
        bottom_zarr = xr.open_zarr(cruise['bottom_file'])
        # print(sv_zarr)
        # print(label_zarr)
        # print(bottom_zarr)

        # Check if reports file already exists
        reports_zarr_dir = cruise['report_file']
        reports_csv_dir = cruise['report_file'][:-4] + 'csv'

        if os.path.exists(reports_zarr_dir):
            print(f'Report already exists...{reports_zarr_dir}')
        else:
            # Extracting report number
            report_name = reports_zarr_dir[-13:-4]

            # Assigning the threshold value accordingly
            threshold = thresholds[report_name]

            print('Export reports.zarr to ', reports_zarr_dir)  # zarr name
            print('Export reports.csv to ', reports_csv_dir)  # csv name

            # Generating reports
            generate_report(sv_zarr, label_zarr, bottom_zarr, threshold,
                            reports_csv_dir, reports_zarr_dir)
    except:
        print(cruise)
        print('Failed!!!!')


def runcruises(files):
    for _files in files:
        sv_exist = files[_files]['sv_file'] != None
        bottom_exist = files[_files]['bottom_file'] != None
        pred_exist = files[_files]['pred_file'] != None
        if sv_exist & bottom_exist & pred_exist:
            # print ('Existing data for ', files[_files]['report_file'])
            # Run processing
            runcruise(files[_files])
        else:
            print('Missing data for  ', files[_files]['report_file'])
            # print ('sv data     : ', str(sv_exist))
            # print ('Bottom data : ', str(bottom_exist))
            # print ('Pred data   : ', str(pred_exist))


files = {}
for _cs in cs:
    # Sv file
    _sv_file = os.path.join('ACOUSTIC', 'GRIDDED', 'S' + _cs + '_sv.zarr')
    sv_file = prodstage(crimacscratch, _cs, _sv_file)
    # Bottom file
    _bottom_file = os.path.join('ACOUSTIC', 'GRIDDED', 'S' + _cs + '_bottom.zarr')
    bottom_file = prodstage(crimacscratch, _cs, _sv_file)

    # Loop over pairs of predictions abd reports
    for _pr in pr:
        # Prediction
        if _pr[0].split('.')[0] == 'labels':
            pl = 'GRIDDED'
        else:
            pl = 'PREDICTIONS'
        _pred_file = os.path.join('ACOUSTIC', pl, 'S' + _cs + '_' + _pr[0])
        pred_file = prodstage(crimacscratch, _cs, _pred_file)
        # Report
        _report_file = os.path.join('ACOUSTIC', 'REPORTS', 'S' + _cs + '_' + _pr[1])
        report_file = os.path.join(dataout, _cs[0:4], 'S' + _cs, _report_file)
        files['S' + _cs + '_' + _pr[1]] = ({'sv_file': sv_file, 'bottom_file': bottom_file,
                                            'pred_file': pred_file, 'report_file': report_file})

runcruises(files)

'''
os.path.exists('/data/crimac-scratch/staging/2022/S2022206/ACOUSTIC/PREDICTIONS/S2022206_predictions_4.zarr')
nils = xr.open_zarr('/data/crimac-scratch/staging/2022/S2022206/ACOUSTIC/PREDICTIONS/S2022206_predictions_4.zarr')

os.path.exists('/data/crimac-scratch/staging/2021/S2021847/ACOUSTIC/PREDICTIONS/S2021847_predictions_4.zarr')
nils = xr.open_zarr('/data/crimac-scratch/staging/2021/S2021847/ACOUSTIC/PREDICTIONS/S2021847_predictions_4.zarr')


\\ces.hi.no\crimac-scratch\staging\2021\S2021847\ACOUSTIC\PREDICTIONS\S2021847_predictions_4.zarr

import zarr
store = zarr.open('/data/crimac-scratch/staging/2022/S2022206/ACOUSTIC/PREDICTIONS/S2022206_predictions_4.zarr', mode='r')
print('.zmetadata' in store)

'''
