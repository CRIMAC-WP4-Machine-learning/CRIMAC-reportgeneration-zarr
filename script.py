import os
import xarray as xr

# Script to generate the reports from the sv+label for the Sand eel series

#crimacscratch = os.getenv('CRIMACSCRATCH')
crimacscratch = '/data/crimac-scratch/'
dataout = '/mnt/c/DATAscratch/crimac-scratch/testing'

# Sand eel surveys
cs = ['2005205', '2006207', '2007205', '2008205', '2009107',
      '2010205', '2011206', '2012837', '2013842', '2014807',
      '2015837', '2016837', '2017843', '2018823', '2019847',
      '2020821', '2021847', '2022206', '2022611', '2023006009',
      '2024002006']

# Predictions/labels vs reports
pr = [['labels.zarr', 'report_1.zarr'],
      ['predictions_2.zarr', 'report_2.zarr'],
      ['predictions_3.zarr', 'report_3.zarr'],
      ['predictions_4.zarr', 'report_4.zarr']]


# This function prioritizes the staging data over the production data
def prodstage(crimacscratch,_cs,zarrstore):
  prod= os.path.join(crimacscratch, _cs[0:4], 'S'+_cs, zarrstore)
  staging = os.path.join(crimacscratch, 'staging', _cs[0:4],
                      'S'+_cs, zarrstore)
  if os.path.exists(staging): 
    d = staging
  elif os.path.exists(prod):
    d = prod
  else:
    d = None
  return d


def runcruise(cruise):
  try:
    sv_zarr = xr.open_zarr(cruise['sv_file'])
    label_zarr = xr.open_zarr(cruise['pred_file'])
    bottom_zarr = xr.open_zarr(cruise['bottom_file'])
    print(sv_zarr)
    print(label_zarr)
    print(bottom_zarr)
    print('Export results to ',cruise['report_file'])
  except:
    print(cruise)
    print('Failed!!!!')


def runcruises(files):
  for _files in files:
    sv_exist = files[_files]['sv_file']!=None
    bottom_exist = files[_files]['bottom_file']!=None
    pred_exist = files[_files]['pred_file']!=None
    if sv_exist & bottom_exist & pred_exist:
      #print ('Existing data for ', files[_files]['report_file'])
      # Run processing
      runcruise(files[_files])
    else:
      print ('Missing data for  ', files[_files]['report_file'])
      #print ('sv data     : ', str(sv_exist))
      #print ('Bottom data : ', str(bottom_exist))
      #print ('Pred data   : ', str(pred_exist))


files = {}

for _cs in cs:
  # Sv file
  _sv_file = os.path.join('ACOUSTIC','GRIDDED','S'+_cs+'_sv.zarr')
  sv_file = prodstage(crimacscratch,_cs,_sv_file)
  # Bottom file
  _bottom_file = os.path.join('ACOUSTIC','GRIDDED','S'+_cs+'_bottom.zarr')
  bottom_file = prodstage(crimacscratch,_cs,_sv_file)

  # Loop over pairs of predictions abd reports
  for _pr in pr:
    # Prediction
    _pred_file = os.path.join('ACOUSTIC','PREDICTIONS','S'+_cs+'_'+_pr[0])
    pred_file = prodstage(crimacscratch,_cs,_pred_file)
    # Report
    _report_file = os.path.join('ACOUSTIC','REPORTS','S'+_cs+'_'+_pr[1])
    report_file = os.path.join(dataout, _cs[0:4], 'S'+_cs,_report_file)
    files['S'+_cs+'_'+_pr[1]] = ({'sv_file':sv_file, 'bottom_file':bottom_file,
                       'pred_file':pred_file, 'report_file':report_file})

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
