import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from utils import (process_report_csv, generate_line_plots, generate_hexbin_plots, generate_PSU_transect_line_plots,
                   generate_boxplot, plot_worst_best_examples)
import time

start_time = time.time()

# Script to generate the reports from the sv+label for the Sand eel series

crimacscratch = os.getenv('CRIMACSCRATCH')
dataout = os.path.join(crimacscratch, 'tmp', 'sand_eel_results')

# Sand eel surveys
cs = ['2005205', '2006207', '2007205', '2008205', '2009107',
      '2010205', '2011206', '2013842', '2014807', '2015837',
      '2016837', '2017843', '2018823', '2019847', '2020821',
      '2021847', '2022206', '2022611', '2023006009', '2024002006']

# Predictions/labels vs reports
pr = ['report_1.csv', 'report_2.csv', 'report_3.csv', 'report_4.csv']



'''
thresholds = {  # mean values of the training years
    "report_1": 1.0,
    "report_2": 0.967480,
    "report_3": 0.900195,
    "report_4": 0.896094,
}
'''

thresholds = {  # median values of the training years
    "report_1": 1.0,
    "report_2": 0.963378906,
    "report_3": 0.905761719,
    "report_4": 0.914550781,
}

# This function prioritizes the staging data over the production data
def prodstage(crimacscratch,_cs,zarrstore):
  prod = os.path.join(crimacscratch, _cs[0:4], 'S'+_cs, zarrstore)
  staging = os.path.join(crimacscratch, 'staging', _cs[0:4],
                      'S'+_cs, zarrstore)
  if os.path.exists(staging):
    d = staging
  elif os.path.exists(prod):
    d = prod
  else:
    d = None
  return d


# Initialize output CSV file
output_csv = f'{dataout}/sa_sum_values_summary.csv'
if not os.path.exists(output_csv):
    pd.DataFrame(columns=['Year', 'Report_1', 'Report_2', 'Report_3', 'Report_4']).to_csv(output_csv, index=False)



# Plot the figures as a function of time
for _cs in cs:

    report_files = []
    missing_dirs = []

    # Build report_files list and check if prodstage returns None
    for i in range(4):
        report_path = os.path.join('ACOUSTIC', 'REPORTS', f'S{_cs}_{pr[i]}')
        result = prodstage(crimacscratch, _cs, report_path)
        if result:  # Valid path
            report_files.append(result)
        else:  # If prodstage returns None, add to missing_dirs
            missing_dirs.append(report_path)

    # Path to STOX
    path_to_STOX = f'{crimacscratch}{_cs[0:4]}/S{_cs}/STOX'
    if not os.path.exists(path_to_STOX):
        missing_dirs.append(path_to_STOX)

    # Check for missing paths and continue if any are absent
    if missing_dirs:
        print(f"\nMissing directories for survey {_cs}:", *missing_dirs, sep="\n")
        continue
    print(f'\nCreating plots for survey {_cs}.')

    result_1, result_1_averaged = process_report_csv(report_files[0], path_to_STOX, f'{_cs}')
    result_2, result_2_averaged = process_report_csv(report_files[1], path_to_STOX, f'{_cs}')
    result_3, result_3_averaged = process_report_csv(report_files[2], path_to_STOX, f'{_cs}')
    result_4, result_4_averaged = process_report_csv(report_files[3], path_to_STOX, f'{_cs}')

    # APPENDING THE sa SUMS TO THE CSV FILE - GENERAL COMPARISON AMONG YEARS
    sa_sums = {
        'Year': _cs[:4],
        'Report_1': result_1_averaged['sa_value'].sum() if result_1_averaged is not None else None,
        'Report_2': result_2_averaged['sa_value'].sum() if result_2_averaged is not None else None,
        'Report_3': result_3_averaged['sa_value'].sum() if result_3_averaged is not None else None,
        'Report_4': result_4_averaged['sa_value'].sum() if result_4_averaged is not None else None,
    }

    # Append the data to the CSV
    pd.DataFrame([sa_sums]).to_csv(output_csv, mode='a', header=False, index=False)


    # Reading sv, labes, and predictions

    report_path = os.path.join('ACOUSTIC', 'REPORTS', f'S{_cs}_{pr[i]}')
    result = prodstage(crimacscratch, _cs, report_path)

    zarrs = ['sv.zarr', 'labels.zarr', 'predictions_2.zarr', 'predictions_3.zarr', 'predictions_4.zarr']
    for i in range(len(zarrs)):
        report_path = os.path.join('ACOUSTIC', 'GRIDDED', f'S{_cs}_{zarrs[i]}')
        result = prodstage(crimacscratch, _cs, report_path)

    # Attach sv, labels, bottom, and predictions
    zarrstore = f'ACOUSTIC/GRIDDED/S{_cs}_sv.zarr'
    sv_f = prodstage(crimacscratch,_cs, zarrstore)
    sv = xr.open_zarr(sv_f)
    bottom_f = prodstage(crimacscratch, _cs, f'ACOUSTIC/GRIDDED/S{_cs}_bottom.zarr')
    bottom = xr.open_zarr(bottom_f)
    labels_f = prodstage(crimacscratch, _cs, f'ACOUSTIC/GRIDDED/S{_cs}_labels.zarr')
    predictions_1 = xr.open_zarr(labels_f)
    predictions_2 = xr.open_zarr(
        f'{crimacscratch}/staging/{_cs[0:4]}/S{_cs}/ACOUSTIC/PREDICTIONS/S{_cs}_predictions_2.zarr')
    predictions_3 = xr.open_zarr(
        f'{crimacscratch}/staging/{_cs[0:4]}/S{_cs}/ACOUSTIC/PREDICTIONS/S{_cs}_predictions_3.zarr')
    predictions_4 = xr.open_zarr(
        f'{crimacscratch}/staging/{_cs[0:4]}/S{_cs}/ACOUSTIC/PREDICTIONS/S{_cs}_predictions_4.zarr')

    # Visualizing best and worst examples from survey
    # Predictions_2
    plot_worst_best_examples(sv, bottom, predictions_1, predictions_2, result_1,
                             result_2, thresholds['report_2'], f'S{_cs}',
                             'worst', f'{dataout}/examples/S{_cs}_pred_2_examples')
    plot_worst_best_examples(sv, bottom, predictions_1, predictions_2, result_1,
                             result_2, thresholds['report_2'], f'S{_cs}',
                             'best', f'{dataout}/examples/S{_cs}_pred_2_examples')

    # Predictions_3
    plot_worst_best_examples(sv, bottom, predictions_1, predictions_3, result_1,
                             result_3, thresholds['report_3'], f'S{_cs}',
                             'worst', f'{dataout}/examples/S{_cs}_pred_3_examples')
    plot_worst_best_examples(sv, bottom, predictions_1, predictions_3, result_1,
                             result_3, thresholds['report_3'], f'S{_cs}',
                             'best', f'{dataout}/examples/S{_cs}_pred_3_examples')

    # Predictions_4
    plot_worst_best_examples(sv, bottom, predictions_1, predictions_4, result_1,
                             result_4, thresholds['report_4'], f'S{_cs}',
                             'worst', f'{dataout}/examples/S{_cs}_pred_4_examples')
    plot_worst_best_examples(sv, bottom, predictions_1, predictions_4, result_1,
                             result_4, thresholds['report_4'], f'S{_cs}',
                             'best', f'{dataout}/examples/S{_cs}_pred_4_examples')


    # Line plots on each 0.1 nm
    generate_line_plots(result_1, result_2, result_3, result_4, f'{dataout}/S{_cs}_line_plots_sa.jpg')

    # Line plots on each transect
    generate_PSU_transect_line_plots(result_1_averaged, result_2_averaged, result_3_averaged, result_4_averaged,
                        f'{dataout}/S{_cs}_line_plots_sa_transect_averaged.jpg')

    # Scatter and density (hexbin) plots
    generate_hexbin_plots(result_1_averaged, result_2_averaged, result_3_averaged, result_4_averaged,
                          f'{dataout}/S{_cs}_scatters_sa_comparison.jpg')

    # Generate boxplots for sa comparison
    generate_boxplot(result_1_averaged, result_2_averaged, result_3_averaged, result_4_averaged,
                     f'{dataout}/S{_cs}_boxplots_sa_comparison.jpg')

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"\n### Execution time: {execution_time_minutes:.2f} minutes")

