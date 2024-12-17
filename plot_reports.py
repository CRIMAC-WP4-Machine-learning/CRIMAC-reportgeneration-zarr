import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from utils import process_report_csv, generate_line_plots, generate_hexbin_plots
import time

start_time = time.time()

# Script to generate the reports from the sv+label for the Sand eel series

#crimacscratch = os.getenv('CRIMACSCRATCH')
crimacscratch = '/scratch/disk5/ahmet/remote_data/'
dataout = '/scratch/disk5/ahmet/testing'

# Sand eel surveys
cs = ['2005205', '2006207', '2007205', '2008205', '2009107',
      '2010205', '2011206', '2013842', '2014807', '2015837',
      '2016837', '2017843', '2018823', '2019847', '2020821',
      '2021847', '2022206', '2022611', '2023006009', '2024002006']

# Predictions/labels vs reports
pr = ['report_1.csv', 'report_2.csv', 'report_3.csv', 'report_4.csv']


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
        print(f"Missing directories for survey {_cs}:", *missing_dirs, sep="\n")
        continue
    print(f'Creating plots for survey = {_cs}')

    result_1 = process_report_csv(report_files[0], path_to_STOX, f'{_cs}')
    result_2 = process_report_csv(report_files[1], path_to_STOX, f'{_cs}')
    result_3 = process_report_csv(report_files[2], path_to_STOX, f'{_cs}')
    result_4 = process_report_csv(report_files[3], path_to_STOX, f'{_cs}')

    # Example call with your DataFrames:
    generate_line_plots(result_1, result_2, result_3, result_4, f'{dataout}/S{_cs}_sa_line_plots.jpg')

    # Example call with your DataFrames:
    generate_hexbin_plots(result_1, result_2, result_3, result_4, f'{dataout}/S{_cs}_sa_comparison_scatters.jpg')

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"### Execution time: {execution_time_minutes:.2f} minutes")

