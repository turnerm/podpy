"""
podpy is an implementatin of the pixel optical depth method as described in 
Turner et al. 2014, MNRAS, 445, 794, and Aguirre et al. 2002, ApJ, 576, 1. 
Please contact the author (Monica Turner) at turnerm@mit.edu if you have 
any questions, comment or issues. 
"""

import numpy as np
import numpy.random as rd
import universe as un

class TauBinned:

	"""
	Creates a spectrum object.

	Parameters
	----------
	tau_x_object: instance of the Pod class
		Instance of the Pod class that you would like to use for the x axis (i.e., the 
		optical depths used for binning)
	tau_x_object: instance of the Pod class
		Instance of the Pod class that you would like to use for the y axis (i.e., compute
		the optical depth percentile in each bin of x)
	tau_x_min, tau_x_max: float, optional
		The minimum and maximum optical depths to use x (in log). Defaults are -5.75 and 4.0. 
	bin_size: float, optional
		The bin size to use when binning in x (in log). Default is 0.25. 
	chunk_size: float, optional
		The size of the chunks for bootstrap resampling, in Angstroms. Defaults is 50 A. 
	percentile_value: float, optional
		The percentile to compute in y. Default is 50.

	Instance attributes
	----------
	tau_binned_x: The medians of the binned optical depths in x
	tau_binned_y: The percentile of the binned optical depths in y
	tau_binned_err: The error on the percentile in y 	
	"""


	MIN_BIN_PIXELS = 25
	MIN_CHUNKS_PER_BIN = 5
	MIN_PIX_ABOVE_PERCENTILE = 5
	FLAG = -10


	def __init__(self, 
				tau_x_object, 
				tau_y_object, 
				tau_x_min = -5.75,  
				tau_x_max = 4.0, 
				bin_size = 0.25, 
				chunk_size = 5.0, # Angstroms
				percentile_value = 50,
				bsrs = 1000,
				seed = 12345):
		self.seed = seed
		self.percentile_value = percentile_value
		self.chunk_size = chunk_size
		self.ion_x = tau_x_object.ion
		self.ion_y = tau_y_object.ion
		print "*** Binning optical depths ***"
		# Create the optical depth arrays
		lambda_x, tau_x, tau_y = TauBinned._find_pixel_pairs(tau_x_object, tau_y_object)
		# Get the flat level, tau_min
		self._get_tau_min(tau_x, tau_y)
		print "Binning", self.ion_x, "vs.", self.ion_y
		print "Number of good pixels:", len(lambda_x)
		tau_chunks_x, tau_chunks_y = self._calc_percentiles(lambda_x, tau_x, tau_y, 
															tau_x_min, tau_x_max, bin_size)
		if bsrs:
			self._calc_errors(bsrs, tau_chunks_x, tau_chunks_y)

	@staticmethod
	def _find_pixel_pairs(tau_x_object, tau_y_object):
  		index_x, index_y = TauBinned._find_same_range(tau_x_object.z, tau_y_object.z)
		bad_pixel = ((tau_x_object.flag[index_x] % 2 == 1) | (tau_y_object.flag[index_y] % 2 == 1)) 
		index_x = index_x[~bad_pixel]
		index_y = index_y[~bad_pixel]
		tau_x, tau_y = tau_x_object.tau_rec[index_x], tau_y_object.tau_rec[index_y] 
		lambda_x = tau_x_object.lambdaa[index_x]
		return lambda_x, tau_x, tau_y 

	@staticmethod
	def _find_same_range(array_1, array_2):
		# This is to be used for cases when oVI z range is less than that of 
		# LyA and they need to be made the same. So the input array is either z
		# or lambda, and should be consecutive.    
		error = 1E-10
		index_1 = np.where((array_1 >= array_2[0] - error) & (array_1 <= array_2[-1] + error))[0]
		index_2 = np.where((array_2 >= array_1[0] - error) & (array_2 <= array_1[-1] + error))[0]
		return index_1, index_2



	def _get_tau_min(self, tau_x, tau_y):
		self._get_tau_c()
		index_min = np.where(tau_x < self.tau_c) 
		self.tau_min = np.percentile(tau_y[index_min], self.percentile_value)	
	
	def _get_tau_c(self):
		if self.ion_x in ["h1"]:
			self.tau_c = 0.1
		elif self.ion_x in ["c4", "si4"]:
			self.tau_c = 0.01

	def _calc_percentiles(self, lambda_x, tau_x, tau_y, tau_x_min, tau_x_max, bin_size):
		# prepare chunks for bootstrapping
		edge_chunks = np.arange(lambda_x[0], lambda_x[-1] + self.chunk_size, self.chunk_size)
		num_chunks = len(edge_chunks) - 1 
		# need to use list since elements will be different sizes
		chunk_pixels = [0] * (num_chunks)
		tau_chunks_x = [0] * (num_chunks)
		tau_chunks_y = [0] * (num_chunks)	
		for i in range(num_chunks):		
			chunk_pixels[i] = np.where((lambda_x > edge_chunks[i]) & (lambda_x < edge_chunks[i+1]))[0]
			tau_chunks_x[i] = tau_x[chunk_pixels[i]]
			tau_chunks_y[i] = tau_y[chunk_pixels[i]]
		# create the left and right bin edges
		self.edge_bins = np.arange(tau_x_min - bin_size / 2.0, tau_x_max + bin_size/2.0, bin_size) 	
		num_bins = len(self.edge_bins)-1
		# set up the empty arrays
		tau_binned_x, tau_binned_y = np.empty(num_bins), np.empty(num_bins)
		num_chunks_per_bin = np.zeros(num_bins)
		# calculate the percentile 
		print "Calculating", self.percentile_value, "th percentiles"
		for i in range(num_bins):
			bin_pixels = np.where((tau_x > self.edge_bins[i]) & (tau_x < self.edge_bins[i+1]))[0]		
			# Remove any bins that have less than 5 chunks 
			# Also remove any bins that have less than 25 pixels 
			tmp = [filter(lambda x: x in bin_pixels, sublist) for sublist in chunk_pixels]
			tmp = [len(sublist) for sublist in tmp] 
			tmp = np.array(tmp)	
			num_chunks_per_bin[i] = len(np.where(tmp > 0)[0])
			if len(bin_pixels) > 0:
				ypix = tau_y[bin_pixels] 
				yval = np.percentile(ypix, self.percentile_value)	
				ngt_yval = len(np.where(ypix > yval)[0])				
			else:
				ngt_yval = 0
			if ((len(bin_pixels) < TauBinned.MIN_BIN_PIXELS) or 
				(num_chunks_per_bin[i] < TauBinned.MIN_CHUNKS_PER_BIN) or
				(ngt_yval < TauBinned.MIN_PIX_ABOVE_PERCENTILE)):
				bin_pixels = np.empty(0, dtype=int)
			# Calculate the percentiles		
			tau_binned_x[i] = np.median(tau_x[bin_pixels])
			if (len(bin_pixels) > 0):
				tau_binned_y[i] = yval 
			else:
				tau_binned_y[i] = float('nan')
		# get rid of emtpy bins
		self.nan = np.where(np.isnan(tau_binned_x))
		self.tau_binned_x = np.delete(tau_binned_x, self.nan)
		self.tau_binned_y = np.delete(tau_binned_y, self.nan)
		self.num_chunks_per_bin = np.delete(num_chunks_per_bin, self.nan)
		self.num_chunks = num_chunks
		return tau_chunks_x, tau_chunks_y

	def _calc_errors(self, bsrs, tau_chunks_x, tau_chunks_y):
		# bootstrap resampling
		print "Bootstrap resampling"
		rs = rd.RandomState(self.seed)
		num_bins = len(self.edge_bins) - 1
		val_matrix = np.empty((bsrs, num_bins)) 
		flag_no_pixels = -100
		tau_fake_min = np.empty(bsrs)
		for i in range(bsrs):			
			if (i + 1) % 100 == 0:
				print "...", i + 1
			# Make the fake spectrum of length num_chunks
			tau_fake_x = np.empty(0) 
			tau_fake_y = np.empty(0) 
			for j in range(self.num_chunks):
				index = rs.randint(self.num_chunks)
				tau_fake_x = np.append(tau_fake_x, tau_chunks_x[index])
				tau_fake_y = np.append(tau_fake_y, tau_chunks_y[index])
			# Find the percentile of the optical depth in each bin
			for j in range(num_bins):
				bin_pixels = np.where((tau_fake_x > self.edge_bins[j]) & 
					(tau_fake_x < self.edge_bins[j+1]))[0]
				val_matrix[i][j] = (np.percentile(tau_fake_y[bin_pixels], 
					self.percentile_value) if (len(bin_pixels) > 0) else flag_no_pixels)
			# Also get the errors on tau_min	
			index_min = np.where(tau_fake_x < self.tau_c) 
			tau_fake_min[i] = np.percentile(tau_fake_y[index_min], self.percentile_value)	
		val_matrix = val_matrix.transpose()	
		el = np.empty(num_bins)
		eu = np.empty(num_bins)	
		for i in range(num_bins):	
			tmp = val_matrix[i]
			tmp = tmp[tmp != flag_no_pixels] 
			if len(tmp) > 0:
				el[i] = np.percentile(tmp, un.one_sigma_below)
				eu[i] = np.percentile(tmp, un.one_sigma_above)
		el = np.delete(el, self.nan)
		eu = np.delete(eu, self.nan)
		self.tau_binned_err = (self.tau_binned_y - el, -self.tau_binned_y + eu) 
		self.tau_min_err = (self.tau_min - np.percentile(tau_fake_min, un.one_sigma_below),
							-self.tau_min + np.percentile(tau_fake_min, un.one_sigma_above))


