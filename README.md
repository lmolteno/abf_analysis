# ABF Analysis
This is a project for Shane Ohline to help analyse AHP (after hyperpolarisation) data recorded 
through a patch-clamping rig at the University of Otago. This code:
1. Matches all the pre-AP (action potential baselines with each other (multiple measurements per cell)
2. Averages and convolves the resulting sweeps (this is exported per-cell).
3. Numerically integrates the area between the averaged, convolved sweep and the x-axis from both
the first crossing of 0V (relative to the baseline) to the 2000ms mark. This value is given in mVs per cell.


### Requirements
- csv of
  - for each file a row containing the area under zero  
      - from 500ms -> 2s  
      - from first 0 crossing after the action potential -> 2s  
  - figure exported  
- average line for each age group (excluding outliers)
