# Density-of-state-simulation
Quantum Band Diagram & DOS Visualizer
This Python script processes simulation outputs (CSV files) from quantum well or heterostructure models and produces insightful visualizations including band diagrams, wavefunctions, and Density of States (DOS) plots.

# Code description
Parses band structure and density profile data from band_data.csv and density_profile.csv.
Extracts energy levels, wavefunctions, potentials, and Fermi energy.
Computes and plots 1D Density of States at different spatial positions.
Generates intuitive visual plots:
Band diagram with potential, wavefunctions, and eigenenergies
Local DOS curves at selected positions

# Input Files
Place these CSV files in the working directory:

band_data.csv: Contains position-resolved potential, wavefunction, and energy data.
density_profile.csv: (Optional, currently unused) Contains carrier density profile.


Parameters (in-script)
You can adjust these directly in the script:

num_ext: Number of regions/extensions (e.g., materials).
m_DOS, m_conf: Effective mass parameters for DOS and confinement.
mid_gap_energy: Used to separate valence and conduction states.
eng_min_max: Energy window for DOS.
points: Positions (in Å) at which DOS is evaluated.
Output
Band Diagram with:
Fermi level
Potential
First few wavefunctions
Corresponding eigenenergies
DOS Plots across specified positions
Inset plot highlighting zoomed-in band edge
# Notes
DOS smoothing is applied using a Gaussian filter
