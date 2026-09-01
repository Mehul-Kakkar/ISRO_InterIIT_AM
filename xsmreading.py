# from astropy.io import fits
# import pandas as pd

# with fits.open("Task1/A/xms/ch2_xsm_20200529_v1_level1.fits") as hdul:
#     data = hdul[1].data

# # Normal columns
# df = pd.DataFrame({
#     "Time": data["Time"],
#     "UTCString": data["UTCString"],
#     "FrameNumber": data["FrameNumber"],
#     "BDHTime": data["BDHTime"],
#     "XSMTime": data["XSMTime"],
#     "DecodingStatusFlag": data["DecodingStatusFlag"]
# })

# # Expand DataArray into 2048 columns
# data_array_df = pd.DataFrame(
#     data["DataArray"],
#     columns=[f"DataArray_{i}" for i in range(2048)]
# )

# # Combine them
# df = pd.concat([df, data_array_df], axis=1)

# # Save
# df.to_csv("xsm_data.csv", index=False)

from astropy.io import fits
import pandas as pd

# =========================
# 1. Open the FITS file
# =========================
fits_file = "Task1/A/xms/ch2_xsm_20200529_v1_level1.fits"

with fits.open(fits_file) as hdul:
    data = hdul[1].data

# =========================
# 2. Create DataFrame
# =========================
df = pd.DataFrame({
    "Time": data["Time"],
    "UTCString": data["UTCString"],
    "FrameNumber": data["FrameNumber"],
    "BDHTime": data["BDHTime"],
    "XSMTime": data["XSMTime"],
    "DecodingStatusFlag": data["DecodingStatusFlag"]
})

# =========================
# 3. Expand DataArray
# =========================
data_array_df = pd.DataFrame(
    data["DataArray"],
    columns=[f"DataArray_{i}" for i in range(2048)]
)

df = pd.concat([df, data_array_df], axis=1)

# =========================
# 4. Convert time to datetime
# =========================
df["UTC_datetime"] = pd.to_datetime(
    df["UTCString"].astype(str)
)

# =========================
# 5. Desired time range
# =========================
start_time = pd.Timestamp("2020-05-29 10:40:00")
end_time = pd.Timestamp("2020-05-29 10:43:00")

# =========================
# 6. Find closest available
#    data points to boundaries
# =========================
start_idx = (df["UTC_datetime"] - start_time).abs().idxmin()
end_idx = (df["UTC_datetime"] - end_time).abs().idxmin()

# Make sure start comes before end
if start_idx > end_idx:
    start_idx, end_idx = end_idx, start_idx

# Extract the selected range
df1 = df.loc[start_idx:end_idx].copy()

# =========================
# 7. Save selected data
# =========================
# output_file = "xsm_10_40_to_10_43.csv"

# df_selected.to_csv(output_file, index=False)

# =========================
# 8. Print information
# =========================
# print("Data successfully extracted!")
# print(f"Number of rows: {len(df_selected)}")
# print()
# print("Requested time range:")
# print(f"{start_time} → {end_time}")
# print()
# print("Actual selected time range:")
# print(f"{df_selected['UTC_datetime'].iloc[0]} → "
#       f"{df_selected['UTC_datetime'].iloc[-1]}")
# print()
# print(f"Saved as: {output_file}")

# ACTUAL SHIT

for i in df1:
    pass