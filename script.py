import os
import zarr
import xarray as xr
import numpy as np
import pandas as pd
import time
from report_generation import generate_report

# Script to generate the reports from the sv+label for the Sand eel series

# Threshold Definitions
thresholds = {
    "report_1": 1.0,
    "report_2": 0.967480,
    "report_3": 0.900195,
    "report_4": 0.896094,
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
        # Check if reports file already exists
        reports_zarr_dir = cruise['report_file']
        reports_csv_dir = cruise['report_file'][:-4] + 'csv'

        if os.path.exists(reports_zarr_dir):
            print(f'Report already exists...{reports_zarr_dir}')
        else:
            # Extracting report number
            report_name = reports_zarr_dir[-13:-5]

            # Assigning the threshold value accordingly
            threshold = thresholds[report_name]

            print('Export reports.zarr to ', reports_zarr_dir)  # zarr name
            print('Export reports.csv to ', reports_csv_dir)  # csv name

            sv_zarr = xr.open_zarr(cruise['sv_file'])
            label_zarr = xr.open_zarr(cruise['pred_file'])
            bottom_zarr = xr.open_zarr(cruise['bottom_file'])

            # Generating reports
            generate_report(sv_zarr, label_zarr, bottom_zarr, threshold,
                            reports_csv_dir, reports_zarr_dir)
    except Exception as e:
        print(cruise)
        print('An error occured:')
        print(e)


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
            continue
            # print('Missing data for  ', files[_files]['report_file'])
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
    bottom_file = prodstage(crimacscratch, _cs, _bottom_file)

    # Loop over pairs of predictions abd reports
    for _pr in pr:
        # Prediction
        if _pr[0].split('.')[0] == 'labels':
            pl = 'GRIDDED'
        else:
            pl = 'PREDICTIONS'
        _pred_file = os.path.join('ACOUSTIC', pl, 'S' + _cs + '_' + _pr[0])
        pred_file = prodstage(crimacscratch, _cs, _pred_file)

        if pred_file is None:
            print(f"Prediction file for {_pr[0]} is missing.")
        else:
            print(f"Prediction file for {_pr[0]}: {pred_file}")

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
