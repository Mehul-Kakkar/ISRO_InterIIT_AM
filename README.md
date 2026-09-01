# Task 1 — Part 1: CLASS Raw Spectrum Visualization

This repository contains the code and CLASS data used to visualize and compare the raw L1 X-ray spectra for Footprints A and B.

## Contents

- `Task1/A/cla/` — CLASS L1 FITS data for Footprint A
- `Task1/B/cla/` — CLASS L1 FITS data for Footprint B
- `Task1/A/xms/` — XSM Level-1 data for Footprint A
- Python scripts — Read and combine CLASS spectra, convert channel to energy, and generate the comparison plot.

## Method

1. Read the `COUNTS` column from each CLASS L1 FITS file.
2. Sum the 8-second spectra for each footprint.
3. Convert channel to energy using:

   `Energy (keV) = CHANNEL × 13.5 / 1000`

4. Normalize the summed counts by the total integration time.
5. Plot both spectra over the 0.5–10 keV range recommended for lunar spectral analysis.

## Output

The final plot compares the CLASS X-ray spectra of Footprints A and B, with energy in keV and count rate in counts/s.
