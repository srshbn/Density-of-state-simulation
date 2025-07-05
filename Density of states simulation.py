import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
import os

# Physical constants
Ha = 27.211386245988  # Hartree energy in eV
C = 1.60218E-19       # Coulomb constant
Bohr = 0.529177E-10   # Bohr radius in meters
eps_0 = 8.8541878128E-12  # Vacuum permittivity
J_to_Ha = 1 / (C * Ha)

# ---------------------------------------------
# Function to extract band structure parameters
# ---------------------------------------------
def extract_parameters(path, num_ext, mid_gap_energy, m_DOS, m_conf, verbose=False):
    band = [{'col_num': 0, 'potential': [], 'max_eng': 0, 'min_eng': 0,
             'engs': [], 'wfs': [], 'm_DOS': [], 'm_conf': []} for _ in range(num_ext)]
    band_diag = {'num_ext': 0, 'num_states': 0, 'mid_gap_energy': 0,
                 'density_profile': [], 'position': [], 'fermi': [], 'bands': []}

    band_data_path = os.path.join(path, 'band_data.csv')
    density_profile_path = os.path.join(path, 'density_profile.csv')

    # Determine number of states
    with open(band_data_path) as f:
        for _ in range(2):
            next(f)
        row = next(f).split()
        num_states = int(((len(row) - 2) / num_ext - 1) / 2)

    if verbose:
        print("Row length:", len(row))
        print("Number of states:", num_states)

    # Load data
    band_data_df = pd.read_csv(band_data_path, delim_whitespace=True, skiprows=2, header=None)
    density_profile_df = pd.read_csv(density_profile_path, usecols=[0, 1])

    band_data = band_data_df.to_numpy()
    density_profile = density_profile_df.to_numpy()

    # Assign general quantities
    band_diag['num_states'] = num_states
    band_diag['num_ext'] = num_ext
    band_diag['mid_gap_energy'] = mid_gap_energy
    band_diag['position'] = band_data[:, 0]
    band_diag['fermi'] = band_data[:, 1]
    band_diag['density_profile'] = density_profile

    pos_point = band_diag['position'].shape[0]

    # Extract per-extension band data
    for ii in range(num_ext):
        band[ii]['m_DOS'] = m_DOS[ii]
        band[ii]['m_conf'] = m_conf[ii]

        col_num = 2 + ii * (2 * num_states + 1)
        band[ii]['col_num'] = col_num
        band[ii]['potential'] = band_data[:, col_num]

        wfs = np.zeros((pos_point, num_states))
        engs = np.zeros((pos_point, num_states))
        for jj in range(num_states):
            wfs[:, jj] = band_data[:, col_num + 1 + jj * 2]
            engs[:, jj] = band_data[:, col_num + 2 + jj * 2]

        band[ii]['wfs'] = wfs
        band[ii]['engs'] = engs
        band[ii]['max_eng'] = np.max(engs[0, :])
        band[ii]['min_eng'] = np.min(engs[0, :])

    band_diag['bands'] = band
    return band_diag

# ----------------------------------
# Main script for plotting analysis
# ----------------------------------
if __name__ == '__main__':
    path = os.getcwd()
    num_ext = 2
    mid_gap_energy = 0.0
    m_DOS = [0.59, 0.7]
    m_conf = [0.59, 0.54]

    band_diag = extract_parameters(path, num_ext, mid_gap_energy, m_DOS, m_conf, verbose=True)
    bands = band_diag['bands']
    position = band_diag['position']

    # Create figure and layout
    fig1 = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 2, figure=fig1)
    ax11 = fig1.add_subplot(gs[0, :])
    #ax12 = fig1.add_subplot(gs[1, 0])
    ax13 = fig1.add_subplot(gs[1, :])
    axins = inset_axes(ax13, width=2, height=1.5)

    # Plot band diagram and Fermi energy
    ax11.plot(position, band_diag['fermi'], 'k--', label='Fermi Energy')
    for ii in range(num_ext):
        ax11.plot(position, bands[ii]['potential'], 'b-', label=f'Potential {ii}')
        axins.plot(position, bands[ii]['potential'], 'b-')

    # Plot wavefunctions and eigenenergies
    nstates_to_plot = 5
    for ii in range(num_ext):
        for jj in range(nstates_to_plot):
            ax11.plot(position, bands[ii]['wfs'][:, jj], 'g:', alpha=0.7)
            ax11.plot(position, bands[ii]['engs'][:, jj], 'r-.', alpha=0.7)
            if jj == 0:
                axins.plot(position, bands[ii]['wfs'][:, jj], 'g:')

    # Plot density profile
    #ax12.plot(bands['density_profile'][:, 0], bands['density_profile'][:, 1], label='Carrier Density')
    #ax12.plot(position, bands[ii]['wfs'][:, 2], 'g:', alpha=0.7)

    # Function to calculate DOS
    def get_dos(band_diag, x_point, num_pts, eng_min_max, nstates_to_inc):
        HBAR2OVERM0 = 7.61996163
        num_ext = band_diag['num_ext']
        mid_gap_eng = band_diag['mid_gap_energy']

        dos = np.zeros((2, num_pts))
        for eng_pt in range(num_pts):
            energy = eng_min_max[0] + eng_pt * (eng_min_max[1] - eng_min_max[0]) / num_pts
            dos[0, eng_pt] = energy

            for ext in range(num_ext):
                bands = band_diag['bands'][ext]
                dos_fac = np.sqrt(bands['m_DOS']) * np.sqrt(2.) / np.pi / np.sqrt(HBAR2OVERM0)
                m_avg = np.sqrt(bands['m_DOS'] * bands['m_conf'])
                dos_fac_2D = m_avg / (np.pi * HBAR2OVERM0)

                for state in range(nstates_to_inc):
                    engs = bands['engs']
                    wfs = bands['wfs']
                    if engs[x_point, state] > mid_gap_eng:
                        if energy > engs[x_point, state]:
                            normed_den = ((wfs[x_point, state] - engs[x_point, state]) / 10) ** 2
                            dos[1, eng_pt] += normed_den * dos_fac / np.sqrt(energy - engs[x_point, state])
                        if energy > np.max(bands['potential']):
                            dos[1, eng_pt] += dos_fac_2D
                    else:
                        if energy < engs[x_point, state]:
                            normed_den = ((wfs[x_point, state] - engs[x_point, state]) / 10) ** 2
                            dos[1, eng_pt] += normed_den * dos_fac / np.sqrt(engs[x_point, state] - energy)
                        if energy < np.min(bands['potential']):
                            dos[1, eng_pt] += dos_fac_2D

        return dos

    # Compute and plot DOS at selected x points
    eng_min_max = [-2, 2]
    num_pts = 100
    delta_x = position[-1] - position[-2]
    points = [int(x / delta_x) for x in [450, 455, 460, 465, 470]]

    for x_point in points:
        dos = get_dos(band_diag, x_point, num_pts, eng_min_max, band_diag['num_states'])
        dos[1, :] = gaussian_filter1d(dos[1, :], 2)
        ax13.plot(dos[0, :], dos[1, :], '-', label=f'{position[x_point]:.1f} Å')
        axins.plot([position[x_point], position[x_point]], [-2, 1.5], '-')

    # Final formatting
    ax11.set_xlabel('$x$ position (Å)')
    ax11.set_ylabel('Energy (eV)')
    ax11.set_ylim(-2, 1)
    ax11.set_xlim(0, 600)

    #ax12.set_xlabel('$x$ position (Å)')
    #ax12.set_ylabel('Density (1/cm$^2$)')

    ax13.set_xlabel('Energy (eV)')
    ax13.set_ylabel('DOS')
    ax13.set_ylim(0, 0.5)

    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_ylim(-2, 1.5)
    axins.set_xlim(400, 500)

    # Legend for energy plots
    cust_lines = [
        Line2D([0], [0], color='k', linestyle='--'),
        Line2D([0], [0], color='b', linestyle='-'),
        Line2D([0], [0], color='r', linestyle='-.'),
        Line2D([0], [0], color='g', linestyle=':')
    ]
    ax11.legend(cust_lines, ['Fermi Eng.', 'Potential', 'Energy', 'Wavefunction'])

    # Add subplot labels
    ax11.set_title('(a)', loc='left', x=-0.12)
    #ax12.set_title('(b)', loc='left', x=-0.24)
    ax13.set_title('(c)', loc='left', x=-0.26)

    fig1.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.35)
    plt.show()
