import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Script to generate the reports from the sv+label for the Sand eel series

#crimacscratch = os.getenv('CRIMACSCRATCH')
crimacscratch = '/data/crimac-scratch/'
dataout = '/mnt/c/DATAscratch/MLpreds/'

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
  # Loop over pairs of predictions abd reports
  ex = False
  fig, ax = plt.subplots()
  for _pr in reversed(pr):
    # Report
    _report_file = os.path.join('ACOUSTIC', 'REPORTS', 'S'+_cs+'_'+_pr[1])
    report_file = prodstage(crimacscratch,_cs,_report_file)
    if report_file is not None:
      report_zarr = xr.open_zarr(report_file)
      ex=True
      sa = (10*report_zarr.averaged_sv_data.sum(dim='range'))
      ax.plot(sa.segment, sa, label=_pr[1])
  if ex:
    ax.legend()
    ax.set_title(_cs)
    output_filename = dataout+'S'+_cs+'_sa.png'
    plt.savefig(output_filename, dpi=500, bbox_inches="tight")
    plt.close()


# Plot the scatterplot
for _cs in cs:
  # Loop over pairs of predictions abd reports
  _report1_file = os.path.join('ACOUSTIC', 'REPORTS', 'S'+_cs+'_report_1.zarr')
  report1_file = prodstage(crimacscratch,_cs,_report1_file)
  print(report1_file)
  if report1_file is not None:
    report1_zarr = xr.open_zarr(report1_file)
    sa0 = (10*report1_zarr.averaged_sv_data.sum(dim='range'))
    print('sa0: '+str(len(sa0)))
    fig, ax = plt.subplots()
    for _pr in pr[1:]:
      # Report
      _report_file = os.path.join('ACOUSTIC', 'REPORTS', 'S'+_cs+'_'+_pr[1])
      report_file = prodstage(crimacscratch,_cs,_report_file)
      if report_file is not None:
        report_zarr = xr.open_zarr(report_file)
        sa = (10*report_zarr.averaged_sv_data.sum(dim='range'))
        print('sa : '+str(len(sa)))
        ax.scatter(sa0, sa, label=_pr[1])
    ax.set_xlabel('report_1')
    ax.legend()
    ax.set_title(_cs)
    output_filename = dataout+'S'+_cs+'_sa_scatter.png'
    plt.savefig(output_filename, dpi=500, bbox_inches="tight")
    plt.close()
