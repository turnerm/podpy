"""
podpy is an implementatin of the pixel optical depth method as described in 
Turner et al. 2014, MNRAS, 445, 794, and Aguirre et al. 2002, ApJ, 576, 1. 
Please contact the author (Monica Turner) at turnerm@mit.edu if you have 
any questions, comment or issues. 
"""

"""
This routine demonstrates how to use the POD code on an example 
spectrum. It first initalizes the spectrum, plots it, and then 
calls the POD routine for various ions. Parameter descrptions
are given in the documentation, which can be found using the help()
command.  
"""

from podpy import Spectrum, Pod, TauBinned
import numpy as np
import matplotlib.pyplot as plt
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Plots the recovered (corrected) tau against the original. For more
# details on the flag values, please see the Pod.Pod documentation. 
def plot_tau_rec(spec, ion):
	pobj = vars(spec)[ion]	
	bad_idx = np.where(pobj.flag % 2 != 0)
	sat_idx = np.where(pobj.flag == Pod.Pod.FLAG_SAT) 
	corr_idx = np.where((pobj.flag == Pod.Pod.FLAG_SAT + Pod.Pod.FLAG_REP) | 
						(pobj.flag == Pod.Pod.FLAG_REP))  
	fig, ax = plt.subplots()
	alpha = 0.5
	ax.plot(pobj.tau, pobj.tau_rec, 'k.', alpha = alpha)
	ax.plot(pobj.tau[bad_idx], pobj.tau_rec[bad_idx], 'r.', alpha = alpha, label = "bad/contaminated")
	ax.plot(pobj.tau[sat_idx], pobj.tau_rec[sat_idx], 'b.', alpha = alpha, label = "saturated")
	ax.plot(pobj.tau[corr_idx], pobj.tau_rec[corr_idx], 'g.', alpha = alpha, label = "corrected")
	ax.legend()
	ax.set_xlabel("Log original optical depth")
	ax.set_ylabel("Log Corrected optical depth")
	ax.set_title("Corrected vs original optical depth for " + ion)
	ax.set_xlim(-6.2, 4.2)
	ax.set_ylim(-6.2, 4.2)
	ax.minorticks_on()
	fig.show()


# Parameters for initializing a KODIAQ spectrum
object_name = "J010311+131617"
z_qso = 2.710
filepath = __location__ + "/spectrum/"

# Initialize Spectrum object, with option to manually mask bad regions as well
# as DLAs. See Spectrum.Spectrum and Spectrum.Kodiaq for details. 
spec = Spectrum.KodiaqFits(z_qso, 
						object_name, 
						filepath = filepath, 
						mask_badreg = True, 
						mask_dla = True)
#spec.plot_spectrum()

# Recover HI with the fiduial input parameters. See Pod.LymanAlpha and Pod.Pod
# documentation for the description of these.
spec.get_tau_rec_h1()
plot_tau_rec(spec, "h1")


# For recovering CIV, an automatic continuum fitting redward of the 
# QSO Lya emission is applied. 
spec.fit_continuum()
spec.get_tau_rec_c4()
plot_tau_rec(spec, "c4")


# For recovering OVI, it is best to unmask any DLAs in the Lya forest region  
spec_nodlamask = Spectrum.KodiaqFits(z_qso, 
							object_name, 
							filepath = filepath, 
							mask_badreg = True, 
							mask_dla = False)
spec_nodlamask.get_tau_rec_h1()
spec_nodlamask.get_tau_rec_o6()
plot_tau_rec(spec_nodlamask, "o6")


A = TauBinned.TauBinned(spec.h1, spec.c4)
fig, ax = plt.subplots()

ax.errorbar(A.tau_binned_x, A.tau_binned_y, yerr = A.tau_binned_err,
	marker='o', linestyle='-', color='black', linewidth=1)
ax.axhline(A.tau_min, c = 'k', ls = ":")
ax.set_xlabel(r"$\log_{10} \tau$ HI")
ax.set_ylabel(r"median $\log_{10} \tau$ CIV")
ax.set_xlim(-3.5, 1.5)
ax.set_ylim(-3.5, -1.0)
ax.minorticks_on()

fig.show()




