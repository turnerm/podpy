"""
podpy is an implementatin of the pixel optical depth method as described in 
Turner et al. 2014, MNRAS, 445, 794, and Aguirre et al. 2002, ApJ, 576, 1. 
Please contact the author (Monica Turner) at turnerm@mit.edu if you have 
any questions, comment or issues. 
"""

import numpy as np
import scipy.interpolate as intp
import universe as un


class Pod:
	"""
	Class attributes
	---------------
	TAU_MIN: Value for pixels with negative optical depth
	TAU_MAX: Value for pixels with saturated optical depth 
	N_SIGMA: Sigma multiplier used in numerous routines
	DIST_QSO: The distance blueward of the QSO Lya emission that defines the 
		redshift range (in km/s)
	FLAG_DISCARD: Bad pixel flag -- do not use these for your final results!
	FLAG_SAT: Pixel is saturated, value replaced by TAU_MAX
	FLAG_NEGTAU: Pixel is negative, value replaced by TAU_MIN 
	FLAG_REP: Pixel value was replaced either because it was saturated and a 
		higher order Lyman series line was available (if HI), or by weaker 
		doublet component that is well detected and has lower scaled
		optical depth (if metal ion).

	Instance attributes
	-------------------
	spectrum: the spectrum object used to construct the Pod object 
	g: oscillator strength 
	lambda_Z: ion rest wavelength
	mean_signal_to_noise: mean signal to noise of recovery region 	

	The following arrays all have the same length, and correspond to the region
	of the spectrum where the optical depth was recovered for a particular ion: 
	lambdaa: wavelengths
	z: redshifts 
	flux: fluxes 
	sigma_noise: error
	tau: raw optical depths
	tau_rec: recovered optical depths
	flag: flag values 	
	"""

	TAU_MIN = 10**(-6.0)
	TAU_MAX = 10**(4.0)
	N_SIGMA = 3.0
	DIST_QSO = 3000 # km/s
	FLAG_DISCARD = 1
	FLAG_SAT = 2
	FLAG_NEGTAU = 4
	FLAG_REP = 8

	def __init__(self):
		self._get_z() 
		self._get_flux()
		self._get_tau()
		try:
			self.z[0]
		except IndexError:
			print "ERROR: No wavelength coverage for HI Lya"
		# 
		print "*** Finding tau_rec for", self.ion, "***"
		print "z range:", self.z[0], self.z[-1]
		print "lambda range:", self.lambdaa[0], self.lambdaa[-1]
		print "Total number of pixels:", len(self.lambdaa)
		print "After removing bad pixels from spectrum:", len(np.where(
			self.flag % 2 == 0)[0]) 
			

	def _get_fiducial_z(self):
		z_qso = self.spectrum.z_qso
		z = self.spectrum.lambdaa / un.lambda_h1[0]- 1.0
		z_beta = (1.0 + z_qso) * (un.lambda_h1[1]) / (un.lambda_h1[0]) - 1.0
		z_max = z_qso - (1. + z_qso) * Pod.DIST_QSO / un.c
		idx_lya = np.where((z > z_beta) & (z < z_max))
		return z[idx_lya]	

	def _get_tau(self):
		negative_flux = self.flux <= 0
		saturated = self.flux <= self.nsigma_sat * self.sigma_noise
		self.tau = np.where(negative_flux, Pod.TAU_MAX, -np.log(self.flux))	
		self.tau[self.bad_pixel] = Pod.TAU_MIN	
		# tau_rec takes saturation and negative pixels 
		self.tau_rec = self.tau.copy()
		self.tau_rec[saturated] =  Pod.TAU_MAX	
		negative_tau = np.where(self.tau_rec <= 0)
		self.tau_rec[negative_tau] =  Pod.TAU_MIN
		# make a flag array and flag bad, saturated, negative pixels 
		self.flag = np.zeros(len(self.z), dtype = 'int')	
		self.flag[self.bad_pixel] += Pod.FLAG_DISCARD 
		self.flag[saturated] += Pod.FLAG_SAT
		self.flag[negative_tau] += Pod.FLAG_NEGTAU
		# calc s/n of the recovered region
		self.mean_signal_to_noise = np.mean(1.0/self.sigma_noise[self.flag % 2 == 0])

	@staticmethod
	def find_near_bad_pixel(lambdaa, flag, lambda_search):
		# Flag any pixels that were bad in the spectrum
		index_right = lambdaa.searchsorted(lambda_search, side = 'right')
		index_left = index_right - 1
		near_bad_pixel = (flag[index_right] % 2 == 1) | (flag[index_left] % 2 == 1)
		return(near_bad_pixel)

	@staticmethod
	def _get_z_range_from_ion(z_min, z_max, ion, lambda_Z, z_qso):
		# Only use OVI bluewards of Lya forest (i.e. starting in Lyb forest)
		if ion == "o6":
			z_max = min(z_max, (1. + z_qso) * un.lambda_h1[1] / lambda_Z[1] - 1.)
		# Only use CIV, SiIV, OI and CII redward of the QSO lya emission
		elif ion in ["c4", "si4", "o1", "c2"]: 
			z_min = max(z_min, (1. + z_qso) * un.lambda_h1[0] / lambda_Z[0] - 1.)	
		# Only use SiIII redward of QSO LyB (i.e. use only in the Lya forest)
		elif ion == "si3":
			z_min = max(z_min, (1. + z_qso) * un.lambda_h1[1] / lambda_Z[0] - 1.)
		# Only use NV and SiII blueward of QSO Lya emission
		elif ion in ["n5", "si2"]:
			z_max = min(z_max, (1. + z_qso) * un.lambda_h1[0] / lambda_Z[1] - 1.)
		return z_min, z_max
	
	def _log_all_taus(self):
		self.tau[self.tau <= 0] = Pod.TAU_MIN 
		self.tau_rec[self.tau_rec <= 0] = Pod.TAU_MIN 
		self.tau = np.log10(self.tau)
		self.tau_rec = np.log10(self.tau_rec)

	def print_to_ascii(self,
						path, 
						label = "",
						print_bad = False, 
						print_header = False):
		filename = path + self.spectrum.object_name + "_tau_" + self.ion +\
			label + ".dat"
		f = open(filename, 'w')
		if print_bad: 
			index_good = np.arange(len(self.tau_rec), dtype = int)
		else:
			index_good = np.where(self.flag % 2 == 0)[0]
		if print_header:
			print >> f, "# redshift      tau   flux         sigma_noise"
		for i in index_good:	
			if output_log:
				print >> f, "%9.7f  %10.7f %11.9f  %11.9f" % (self.z[i], 
					self.tau_rec[i], self.flux[i], self.sigma_noise[i])
			else:
				print >> f, "%9.7f  %20.7f  %11.9f %11.9f" % (self.z[i], 
					self.tau_rec[i], self.flux[i], self.sigma_noise[i])
	
# Subclasses for different lines

class LymanAlpha(Pod):

	"""
	Parameters
	----------
	spectrum: object from the Spectrum class 
	n_higher_order: int, default = 15 
		The number of higher-order Lyman series lines used to correct saturated HI
	nsigma_sat: boolean, default = N_SIGMA
		Noise array multiplier to determine pixel saturation. 
	correct_contam: int, default = 1
		If 0, do not search for or correct contaminated HI pixels.
		If 1, flag these pixels as FLAG_DISCARD, but do not attempt to correct.
		If 2, try to correct the contamination.  
	output_log: boolean, default = True
		Output the log of the optical depths.	
	"""
	
	def __init__(self, 
			spectrum, 
			n_higher_order = 15,
			nsigma_sat = Pod.N_SIGMA,
			correct_contam = 1,
			output_log = True 
			):
		self.lambda_Z = un.lambda_h1
		self.g = un.g_h1
		self.spectrum = spectrum
		self.nsigma_sat = nsigma_sat
		self.ion = "h1"
		Pod.__init__(self)
		# Keep track of higher order line parameters
		index_contam = np.zeros(len(self.lambdaa))	
		tau_higher_order_full_flag = 1E4
		tau_higher_order_full = tau_higher_order_full_flag *\
			 np.ones((n_higher_order, len(self.lambdaa)))  
		for iho in range(n_higher_order):
			# Index for accessing un.h1 params   
			ih1 = iho + 1
			# Wavelength array of higher order line
			lambda_higher_order = un.lambda_h1[ih1]*(self.z + 1.0)
			# List of indices that are within the spectral range, of length N_g 
			# (call the full length of pixels N)
			within_spectrum_range = np.where((lambda_higher_order >= 
				self.spectrum.lambdaa[0]) & (lambda_higher_order <= 
				self.spectrum.lambdaa[-1]))[0]
			# now only work with these pixels  
			lambda_higher_order = lambda_higher_order[within_spectrum_range]
			# Looking for pixels to flag: first, bad pixels in both main 
			# and higher order 
			bad_pixel = self.flag[within_spectrum_range] % 2 != 0
			bad_pixel_higher_order = self.find_near_bad_pixel(self.spectrum.lambdaa, 
				self.spectrum.flag, lambda_higher_order)
			# higher order observed flux, and tau and sigma
			flux_higher_order = self.spectrum.flux_function(lambda_higher_order)
			negative_flux = (flux_higher_order <= 0)
			tau_higher_order = np.where(negative_flux, Pod.TAU_MAX, 
				-np.log(flux_higher_order) * un.g_h1[0] / un.g_h1[ih1])
			sigma_noise_higher_order = (
				self.spectrum.sigma_noise_function(lambda_higher_order))
			if correct_contam:
				# Use lya to estimate expected higher order flux  
				# If the pixel is saturated, set it to three times the noise 
				flux_expected_higher_order = np.max((self.flux[within_spectrum_range],
					self.nsigma_sat * self.sigma_noise[within_spectrum_range]),
					axis = 0)
				# Then scale by the transition strength 
				flux_expected_higher_order = (flux_expected_higher_order **
					(un.g_h1[ih1] / un.g_h1[0]))
				# A pixel is considered contaminated if it's higher order flux is 
				# above the expected flux by N_sigma * noise_ho 
				contam = (flux_higher_order - self.nsigma_sat * 
					sigma_noise_higher_order > flux_expected_higher_order) 
				# The pixel is marked as contaminatd if it meets the contamination 
				# criterea as well as is not a bad pixel in either the main or 
				# higher order line. However this array is of length N_g and not 
				# the full length  
				index_contam_within_spectrum_range = np.where((contam) & (~bad_pixel) &
					(~bad_pixel_higher_order))
				# Now use this to make an array of indices that can be applied to 
				# the full range, and mark it as contaminated in the full array  
				index_full_contam = within_spectrum_range[
					index_contam_within_spectrum_range] 
				index_contam[index_full_contam] = 1
				if correct_contam == 2:
					# For the contaminated pixels, add tau_higher_order to 
					# the full array
					tau_higher_order_full[iho][index_full_contam] = (
						tau_higher_order[index_contam_within_spectrum_range])
			# Also, the saturated pixels have to be replaced
			# -- only use "well detected" pixels 
			saturated = self.flag[within_spectrum_range] == Pod.FLAG_SAT 
			lhs = flux_higher_order >= self.nsigma_sat * sigma_noise_higher_order 
			rhs = (flux_higher_order <= 1.0 - self.nsigma_sat * 
				sigma_noise_higher_order)
			if correct_contam == 1:	
				index_sat_within_spectrum_range = np.where((saturated) & (~bad_pixel) &
					(~bad_pixel_higher_order) & (lhs) & (rhs) & (~contam))
			else:
				index_sat_within_spectrum_range = np.where((saturated) & (~bad_pixel) &
					(~bad_pixel_higher_order) & (lhs) & (rhs))
			index_full_sat = within_spectrum_range[index_sat_within_spectrum_range] 
			tau_higher_order_full[iho][index_full_sat] = (
				tau_higher_order[index_sat_within_spectrum_range])
		# Collapse the full tau_higher_order array to the minimum values
		if n_higher_order > 0:
			tau_higher_order_full = np.min(tau_higher_order_full, axis = 0)
			idx_replace = np.where(tau_higher_order_full != tau_higher_order_full_flag)
			self.tau_rec[idx_replace] = tau_higher_order_full[idx_replace]
			self.flag[idx_replace] += Pod.FLAG_REP
			self.index_contam = index_contam
			if correct_contam == 1:
				self.flag[np.where(index_contam == 1)] += Pod.FLAG_DISCARD
		self._print_stats()
		if output_log:
			self._log_all_taus()
		else:
			self.tau[self.tau <= 0] = Pod.TAU_MIN 
			self.tau_rec[self.tau_rec <= 0] = Pod.TAU_MIN 
		print "*** Done ***\n"
	
	def _get_z(self):
		self.z = self._get_fiducial_z()

	def _get_flux(self):
		lambda_rest = self.lambda_Z[0]
		lambdaa = lambda_rest * (1.0 + self.z) 
		self.idx = np.where((self.spectrum.lambdaa >= lambdaa[0]) &
			(self.spectrum.lambdaa <= lambdaa[-1]))
		self.lambdaa = self.spectrum.lambdaa[self.idx]
		self.flux = self.spectrum.flux[self.idx]
		self.sigma_noise = self.spectrum.sigma_noise[self.idx]
		self.bad_pixel = self.sigma_noise <= 0
	

	def _print_stats(self):
		print "Pixels analyzed:", len(self.tau)
		print "Number of saturated pixels:", len(np.where((self.flag == Pod.FLAG_SAT) | 
		(self.flag == Pod.FLAG_SAT + Pod.FLAG_REP))[0]) 
		print "Number of these pixels that have lower optical depth in higher order lines:"
		print len(np.where(self.flag == Pod.FLAG_SAT + Pod.FLAG_REP)[0])

class Metal(Pod):
	
	"""
	Parameters
	----------
	spectrum: object from the Spectrum class
	ion: str
		The ion for which to recover the optical depth
	correct_h1: boolean
		Whether to subtract the HI Lyb series. Usually done for OVI and CIII.
	correct_self: boolean
		Whether to perform the self-contamination correction. Usually done for CIV.
	take_min_doublet: boolean
		Whether to take the minimum optical depth of the doublet. Usually done for 
		NV, OVI, and SiIV.
	nsigma_sat: boolean, default = N_SIGMA
		Noise array multiplier to determine pixel saturation
	nsigma_dm: boolean, default = N_SIGMA
		Noise array multiplier to determine when the weaker doublet component 
		should be used.
		Only required if take_min_doublet = True.
	nsigma_contam: boolean, default = N_SIGMA
		Noise array multiplier used to determine if pixel is contaminated. 
		Only required if correct_self = True.
	n_higher_order: int, default = 5
		The number of higher-order Lyman series lines used to subtract HI.
		Only rquired if correct_h1 = True.
	output_log: boolean, default = True
		Output the logs of the optical depths.	
	"""

	def __init__(self, 
			spectrum, 
			ion, 
			correct_h1, 
			correct_self, 
			take_min_doublet, 
			nsigma_sat = Pod.N_SIGMA,
			nsigma_dm = Pod.N_SIGMA,
			nsigma_contam = Pod.N_SIGMA,
			n_higher_order = 5,
			output_log = True, 
			):
		self.spectrum = spectrum
		self.ion = ion
		self.nsigma_sat = nsigma_sat
		self.lambda_Z, self.g = self._get_ion_properties(self.ion)
		Pod.__init__(self)
		if take_min_doublet: 
			self._get_weaker_tau()
		# Part A
		if correct_h1:
			print "Part A: correct for higher order Lyman contamination"
			if self.ion == "o6":
				self.h1_contam_flag = np.zeros(len(self.z))	
			self._correct_for_h1(self.tau, self.tau_rec, self.lambdaa, self.flag,
				sigma_noise = self.sigma_noise, n_higher_order = n_higher_order)
			# If taking doublet minimum, need to apply HI correction to weaker 
			# component
			if take_min_doublet: 
				self._correct_for_h1(self.tau_w, self.tau_w_rec, self.lambda_w, 
					self.flag_w, sigma_noise = self.sigma_noise_w, 
					n_higher_order = n_higher_order) 
			if self.ion == "o6":
				h1_contam = np.where(self.h1_contam_flag == 2)
				self.flag[h1_contam] += Pod.FLAG_DISCARD 
		# Part B
		if correct_self:
			print "Part B: correct for self contamination"
			self._correct_for_self(nsigma_contam = nsigma_contam)
		# Part C
		if take_min_doublet:
			print "Part C: Take doublet minimum"
			self._take_min_of_doublet(nsigma_dm = nsigma_dm)
		# fix up some stuff...
		if output_log:
			self._log_all_taus()
		print "*** Done ***\n"

	@staticmethod
	def _get_ion_properties(ion):
		lambda_Z = vars(un)["lambda_" + ion]
		g_Z = vars(un)["g_" + ion] 
		return(lambda_Z.copy(), g_Z)

	def _get_z(self):
		z_qso = self.spectrum.z_qso
		z = self._get_fiducial_z()
		# Get z constraints depending on the ion 
		z_min, z_max = self._get_z_range_from_ion(z[0], z[-1], self.ion, 
			self.lambda_Z, z_qso)
		index_z_min = z.searchsorted(z_min, side = 'left')
		index_z_max = z.searchsorted(z_max, side = 'right') 
		self.z = z[index_z_min:index_z_max]

	
	def _get_flux(self):
		lambda_rest = self.lambda_Z[0]
		lambdaa = lambda_rest * (1.0 + self.z) 
		within_spectral_range = np.where((lambdaa >= self.spectrum.lambdaa[0])
			& (lambdaa <= self.spectrum.lambdaa[-1]))
		self.z = self.z[within_spectral_range]
		self.lambdaa = lambdaa[within_spectral_range]
		self.flux = self.spectrum.flux_function(self.lambdaa)
		self.sigma_noise = self.spectrum.sigma_noise_function(self.lambdaa)
		self.bad_pixel = self.find_near_bad_pixel(self.spectrum.lambdaa, 
				self.spectrum.flag, self.lambdaa)
	

	def _get_weaker_tau(self):
		# Find the weaker line
		self.lambda_w = self.lambda_Z[1] * (self.z + 1.0)
		self.flux_w = self.spectrum.flux_function(self.lambda_w)
		self.sigma_noise_w = self.spectrum.sigma_noise_function(self.lambda_w)	
		# Find saturated and bad pixels
		negative_flux = self.flux_w <= 0
		saturated = self.flux_w < self.nsigma_sat * self.sigma_noise_w
		bad_pixel = self.find_near_bad_pixel(self.spectrum.lambdaa, self.spectrum.flag,
			self.lambda_w)
		# Interpolate over flux
		self.tau_w = np.where(negative_flux, Pod.TAU_MAX / 2., 
			-np.log(self.flux_w))	
		self.tau_w[bad_pixel]= Pod.TAU_MIN / 2. 
		# For tau_w_rec take saturation into consideration 
		self.tau_w_rec = self.tau_w.copy() 
		self.tau_w_rec[saturated] = Pod.TAU_MAX  / 2.
		# Set the flags
		self.flag_w = np.zeros(len(self.z))	
		self.flag_w[saturated] += Pod.FLAG_SAT
		self.flag_w[bad_pixel] += Pod.FLAG_DISCARD
		# For the weaker tau, we not set pixels with negative optical depths to 
		# TAU_MIN. This is because for the doublet minimum, we want to keep the 
		# original value of the weaker component so that when the noise comparison 
		# condition is used, the weaker value isn't artifically inflated by being 
		# set to a positive value.  
	
	def _correct_for_h1(self,
					tau,
					tau_rec,
					lambdaa,
					flag,
					sigma_noise,
					n_higher_order):
		tau_rec_h1_function = self.spectrum.interp_f_lambda(self.spectrum.h1.lambdaa, 
			10**(self.spectrum.h1.tau_rec))
		print "Subtracting", n_higher_order, "HI lines"
		for j in range(n_higher_order):
			order = j + 1 # avoid lya
			index_lambdas, lambdas = self._get_h1_correction_lambdas(tau_rec, 
				lambdaa, flag, order)
			tau_rec[index_lambdas] -= (un.g_h1[order] / un.g_h1[0] * 
				tau_rec_h1_function(lambdas))
		# Special case: dealing with saturated pixels 
		index_saturated = np.where(flag == Pod.FLAG_SAT)[0]
		for isat in index_saturated: 
			order_array = np.arange(n_higher_order) + 1
			lambdas = lambdaa[isat] * un.lambda_h1[0] / un.lambda_h1[order_array]
			index_lambdas = np.where((lambdas >= self.spectrum.h1.lambdaa[0]) & 
				(lambdas <= self.spectrum.h1.lambdaa[-1]))[0]
			# Only take the orders within the wavelength range
			order_array = order_array[index_lambdas]
			lambdas = lambdas[index_lambdas]
			total_h1_od = np.sum(un.g_h1[order_array] / un.g_h1[0] 
				 * tau_rec_h1_function(lambdas))
			if np.exp(-total_h1_od) < Pod.N_SIGMA * sigma_noise[isat]:
				if self.ion == "c3": 
					flag[isat] += Pod.FLAG_DISCARD	
				elif self.ion == "o6": 
					self.h1_contam_flag[isat] += Pod.FLAG_DISCARD 
			
	def _get_h1_correction_lambdas(self, tau, lambdaa, flag, order):
		lambdas = lambdaa * un.lambda_h1[0] / un.lambda_h1[order]
		index_lambdas = np.arange(len(lambdas))
		# Make sure lambdas lies within range of calculated tau_rec    
		lambda_max = self.spectrum.h1.lambdaa[-1] 
		index_lambdas = index_lambdas[(lambdas >= self.spectrum.h1.lambdaa[0]) & 
			(lambdas <= lambda_max)] 
		# Make sure tau is not saturated 
		index_lambdas = index_lambdas[flag[index_lambdas] != Pod.FLAG_SAT]	 
		# Make sure the HI optical depth isn't near a bad pixel 
		bad_pixel = self.find_near_bad_pixel(self.spectrum.h1.lambdaa, 
			self.spectrum.h1.flag, lambdas[index_lambdas])
		index_lambdas = index_lambdas[~(bad_pixel)]
		# Make sure that the two h1 pixels being integrated between do not 
		# have tau = max_tau, otherwise don't use them!
		index_right = self.spectrum.h1.lambdaa.searchsorted(lambdas[index_lambdas],
			side = 'right')
		index_left = index_right - 1
		near_saturated_pixel = ((self.spectrum.h1.flag[index_right] == Pod.FLAG_SAT) 
			| (self.spectrum.h1.flag[index_left] == Pod.FLAG_SAT)) 
		index_lambdas = index_lambdas[~near_saturated_pixel]
		lambdas = lambdas[index_lambdas]
		return index_lambdas, lambdas

	def _correct_for_self(self,
						num_iterations = 5,
						nsigma_contam = Pod.N_SIGMA): 		
		# Step B (i)
		print "Part (i)"
		# Make a tau function that spans the full spectral wavelength range 
		saturated = (self.spectrum.flux <= self.nsigma_sat * 
			self.spectrum.sigma_noise)
		tau = np.where(saturated, Pod.TAU_MAX, -np.log(self.spectrum.flux))		
		tau_function = self.spectrum.interp_f_lambda(self.spectrum.lambdaa, tau)
		# Check for contaminated pixels along the c4 range
		contaminated = self._check_if_contaminated(tau_function, self.lambdaa, 
			nsigma_contam)
		# Make sure not to add the odd number to pixels that are already bad from 
		# the spectrum	
		if sum(self.spectrum.sigma_noise) == 0.:
			# Special case in case spectrum is theoretical and has no noise
			self.flag[:] = 0
			contaminated = np.zeros(len(self.flag), dtype = int)
		bad_pixel_from_spectrum = self.flag % 2 == 1
		self.flag[(contaminated != 0) & (~bad_pixel_from_spectrum)] += Pod.FLAG_DISCARD
		print "Number of contaminated / out of range pixels:", len(np.where(
			contaminated)[0]) 
		print "Pixels remaining:", len(np.where(self.flag % 2 == 0)[0])
		# Step B (ii)
		print "Part (ii)"
		# These are the slightly lower wavelength positions to be 
		# subtracted in the self correction
		lambdas = self.lambdaa * self.lambda_Z[0] / self.lambda_Z[1] 
		index_lambdas = np.arange(len(lambdas))
		# Discard those which have parent wavelength that is marked as a bad pixel
		index_lambdas = index_lambdas[(self.flag % 2 == 0)]
		lambdas = lambdas[index_lambdas]
		# Need to split up the lambdas -- some will lie outside of the range of tau_rec
		lambdas_within_range = (lambdas > self.lambdaa[0]) &\
			(lambdas < self.lambdaa[-1])
		index_lambdas_rec = index_lambdas[lambdas_within_range] 
		index_lambdas_full = index_lambdas[~lambdas_within_range] 
		lambdas_rec = lambdas[lambdas_within_range]
		lambdas_full = lambdas[~lambdas_within_range]
		# Also discard any higher order lambdas that are marked as bad
		bad_pixel_rec = self.find_near_bad_pixel(self.lambdaa, self.flag, lambdas_rec)
		index_lambdas_rec = index_lambdas_rec[~(bad_pixel_rec)] 	
		lambdas_rec = lambdas_rec[~(bad_pixel_rec)]
		bad_pixel_full = self.find_near_bad_pixel(self.spectrum.lambdaa, 
			self.spectrum.flag, lambdas_full)	
		index_lambdas_full = index_lambdas_full[~(bad_pixel_full)] 	
		lambdas_full = lambdas_full[~(bad_pixel_full)]
		# Do the correction
		tau_old = self.tau_rec.copy()
		# For the ones outside tau rec, subtract only once 
		subtract = (self.g[1] / self.g[0])* tau_function(lambdas_full)
		self.tau_rec[index_lambdas_full] = tau_old[index_lambdas_full] - subtract	
		# Iterate over the rest of the pixels 
		not_converged = 1
		subtract_old = 0
		while not_converged:
			idx_bad = (self.flag % 2 == 1)
			tau_rec_function = self.spectrum.interp_f_lambda(self.lambdaa[~idx_bad], 
				self.tau_rec[~idx_bad])
			subtract = (self.g[1] / self.g[0])* tau_rec_function(lambdas_rec)
			self.tau_rec[index_lambdas_rec] = tau_old[index_lambdas_rec] - subtract 
			not_converged = max(abs(subtract - subtract_old)) > 1E-4
			subtract_old = subtract
		

	def _check_if_contaminated(self, 
							tau_function, 
							lambda_o,
							n_sigma = Pod.N_SIGMA):
		flag = np.zeros(len(lambda_o))
		z = lambda_o / self.lambda_Z[0] - 1.0  # = self.z for the first case
		lambda_d = self.lambda_Z[1] * (z + 1.0) # > lambda_o
		lambda_s = self.lambda_Z[0] / self.lambda_Z[1] * lambda_o # < lambda_o
		within_spectral_range = ((lambda_d <= self.spectrum.lambdaa[-1]) &
			(lambda_s >= self.spectrum.lambdaa[0]))
		lambda_o = lambda_o[within_spectral_range]	
		lambda_d = lambda_d[within_spectral_range]	
		lambda_s = lambda_s[within_spectral_range]	
		tau_o = tau_function(lambda_o) 
		tau_d = tau_function(lambda_d)
		tau_s = tau_function(lambda_s)	
		sigma_bar = (self.spectrum.sigma_noise_function(lambda_o)**2.0 + 	
			self.spectrum.sigma_noise_function(lambda_d)**2.0 + 
			self.spectrum.sigma_noise_function(lambda_s)**2.0)**0.5 
		lhs = np.exp(-tau_o) + n_sigma * sigma_bar  
		rhs = np.exp(- self.g[1] / self.g[0] * tau_s - self.g[0] / self.g[1] * tau_d)
		contaminated = np.where(lhs < rhs, 1, 0)
		out_of_range = 2 * np.ones(len(np.where(~within_spectral_range)[0]), dtype=int)
		contaminated = np.concatenate((contaminated, out_of_range))
		return (contaminated)

	def _take_min_of_doublet(self,
							nsigma_dm):
		tau1 = self.tau_rec
		sigma1 = self.sigma_noise
		tau2 = self.tau_w_rec
		sigma2 = self.sigma_noise_w 
		lhs = (np.exp(-tau2) - nsigma_dm * sigma2)**(self.g[0] / self.g[1])
		rhs = np.exp(-tau1) + nsigma_dm * sigma1
		replace_with_w = ((lhs > rhs) & (self.flag_w % 2 == 0))
		self.tau_rec = np.where(replace_with_w, tau2, tau1)
		self.flag[replace_with_w] += Pod.FLAG_REP
		print "Replacing", len(np.where(
			replace_with_w)[0]), "pixels with those from the weaker doublet"



