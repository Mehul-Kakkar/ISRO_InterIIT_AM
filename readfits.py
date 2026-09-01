from pathlib import Path
import glob
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

folder_path1 = "./Task1/A/cla"

fits_files1 = glob.glob(f"{folder_path1}/*.fits")

p01 = np.zeros(2048, dtype=float)
p1 = list(enumerate(p01))
df1 = pd.DataFrame(p1, columns=['CHANNEL', 'COUNTS'])
# print(df1)
l1=[]
for file_path in fits_files1:
    path_obj = Path(file_path)
    try:
        with fits.open(file_path) as hdul:
            if len(hdul) > 1 and hdul[1].data is not None:
                df = Table(hdul[1].data).to_pandas()
                d = hdul[1].header
                l1.append((d['SAT_LAT'], d['SAT_LON']))
                df1['COUNTS']=df1['COUNTS']+df['COUNTS']
                
            else:
                print('No Data')
                
    except Exception as e:
        print(f"Error reading {path_obj.name}: {e}")

folder_path2 = "./Task1/B/cla"

fits_files2 = glob.glob(f"{folder_path2}/*.fits")

p02 = np.zeros(2048, dtype=float)
p2 = list(enumerate(p02))
df2 = pd.DataFrame(p2, columns=['CHANNEL', 'COUNTS'])
# print(df2)
l2=[]
for file_path in fits_files2:
    path_obj = Path(file_path)
    try:
        with fits.open(file_path) as hdul:
            if len(hdul) > 1 and hdul[1].data is not None:
                df = Table(hdul[1].data).to_pandas()
                d = hdul[1].header
                l2.append((d['SAT_LAT'], d['SAT_LON']))
                df2['COUNTS']=df2['COUNTS']+df['COUNTS']
                
            else:
                print('No Data')
                
    except Exception as e:
        print(f"Error reading {path_obj.name}: {e}")
t1 = len(fits_files1)*8
t2 = len(fits_files2)*8
print(l1)
print(l2)
# print(df2)
plt.figure(figsize=(16,9))
plt.title(
    "Chandrayaan-2 CLASS L1 X-ray Spectra — Footprints A and B",
    fontsize=18
)
plt.plot(df1['CHANNEL']*13.5/1000, df1['COUNTS']/t1, color='g', label="Footprint A")
plt.plot(df2['CHANNEL']*13.5/1000, df2['COUNTS']/t2, color='r', label="Footprint B")
plt.xlim(0.5, 10)
plt.xlabel("ENERGY (keV)")
plt.ylabel("COUNTS (per sec)")
plt.legend()
plt.show()