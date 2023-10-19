# ABF Analysis
This is a project for Shane Ohline to help analyse AHP (after hyperpolarisation) data recorded 
through a patch-clamping rig at the University of Otago. This code:
1. Matches all the pre-AP (action potential baselines with each other (multiple measurements per cell)
2. Averages and convolves the resulting sweeps (this is exported per-cell).
3. Numerically integrates the area between the averaged, convolved sweep and the x-axis from both
the first crossing of 0V (relative to the baseline) to the 2000ms mark, and from 500ms to 2000ms. This value is given in mVs for each cell.
4. These calculations are repeated for the averaged data across the cells in each group, providing a group average.
