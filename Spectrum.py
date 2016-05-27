import numpy as np
import scipy.interpolate as intp
import matplotlib.pyplot as plt
from astropy.io import fits
import os
###
import universe as un
reload(un)
import Pod 
reload(Pod)


class Spectrum:

	"""
	Creates a spectrum object.

	Parameters
	----------
	z_qso: float
		Redshift of the quasar
	object_name: str
		Name of the object, important for locating relevant files
	filepath: str, optional
		The directory tree leading to the flux, error array and mask files
	mask_badreg: boolean, optional
		Whether bad regions should be manually masked using a file provided by the user. 
		The file should have the format filepath + object_name + "_badreg.mask", and
		contain two tab-separated text columns, where each row contains the start and end
		wavelength of a region to mask.  
	mask_dla: boolean, optional
		Same as mask_badreg, except for masking DLAs, and the file should be named
		filepath + objecta_name + '_dla.mask'.
	verbose: boolean, optional
	
	Instance attributes
	----------
	lambdaa: wavelength in angstroms
	flux: normalized flux array
	sigma_noise: error array 				
	npix: length of the above arrays 
	"""
	BAD_NOISE = -1 # Value to set bad pixels to

	def __init__(self, 
				z_qso,
				object_name,
				filepath = "", 
				mask_badreg = False,
				mask_dla = False,
				verbose = True):
		print "\n*** Getting spectrum ***"
		self.z_qso = z_qso
		self.object_name = object_name 
		self.filepath = filepath
		self.mask_badreg = mask_badreg
		self.mask_dla = mask_dla
		self.verbose = verbose


	def mask_spectrum(self):
		"""
		Search for mask files and apply the mask if they exist   
		"""
		if self.mask_badreg or self.mask_dla:
			mask_filepath = self.filepath + self.object_name 
			mask_type_list = []
			if self.mask_badreg:
				mask_type_list.append("badreg")
			if self.mask_dla:
 				mask_type_list.append("dla")
			if self.verbose:
				print "Looking for mask files..."
			for mask_type in mask_type_list:
				mask_file = mask_filepath + "_" + mask_type + ".mask" 
				if os.path.isfile(mask_file):
					print "...found", mask_type, "file"
					data = np.loadtxt(mask_file)
					if len(data.shape) == 1:
						data = [data]
					for row in data:
						mask_region_idx = np.where((self.lambdaa >= row[0]) & 
							(self.lambdaa <= row[1])) 
						self.sigma_noise[mask_region_idx] = Spectrum.BAD_NOISE
				else:
					if self.verbose:
						print "...WARNING: no", mask_type, "file found"

 
	def prep_spectrum(self):
		"""
		Prepares the spectrum before implementing POD 
		by setting up the flag array that keeps track of bad / corrected pixels,
		and setting up the optical depth arrays and functions. 
		"""
		# Flag regions where spectrum is no good (noise <= 0)
		# and set noise to -1 exactly for consistency in interpolation
		self.flag = np.zeros(self.npix, dtype = int)
		bad_pix = np.where(self.sigma_noise <= 0)
		self.sigma_noise[bad_pix] = Spectrum.BAD_NOISE
		self.flag[bad_pix] = Pod.Pod.FLAG_DISCARD 
		# Set the flux to tau_min since they are bad regions, should not have signal
		self.flux[bad_pix] = 1.0  
		# Prepare the functions for interpolating	
		self.flux_function = self.interp_f_lambda(self.lambdaa, self.flux)
		self.sigma_noise_function = self.interp_f_lambda(self.lambdaa,self.sigma_noise)
		negflux = (self.flux <= 0) 
		self.tau = np.where(negflux, Pod.Pod.TAU_MAX, -np.log(self.flux))		
		self.tau_function = self.interp_f_lambda(self.lambdaa, self.tau)
	
	@staticmethod 
	def interp_f_lambda(lambdaa, f_lambda):
		"""
		Linearly interpolates a function of lambda

		Parameters
		----------
		lambdaa: array_like
			Wavelength array.
		f_lambda: array_like	
			Whichever quantity is a function of the wavelength. 

		Returns
		-------
		f: function of wavelength  

		"""
		return intp.interp1d(lambdaa, f_lambda, kind='linear')

	def get_tau_rec_h1(self, 
						**kwargs):
		"""
		Get the optical depth of HI, returned as an object named self.h1.    

		Parameters
		----------
		label: str, optional
			Adds a label to the object that is attached to the spectrum. e.g., label = "_A" 
			would result in an object named "h1_A". 	

		See Pod class for remaining input parameters.

		"""
		name = "h1" + kwargs.pop("label", "")
		vars(self)[name] = Pod.LymanAlpha(self,
											**kwargs)

	def get_tau_rec_ion(self, 
						ion,
						*args,
						**kwargs): 
		"""
		Get the optical depth of ion $ion, returned as an object named self.$ion

		Parameters
		----------
		ion: str
			Currently supported: c2, c3, c4, n5, o1, o6, si2, si3, si4 
		label: str, optional
			Adds a label to the object that is attached to the spectrum. e.g., label = "_A" 
			would result in an object named "$ion_A". 	

		See Pod class for remaining input parameters.

		"""
		name = ion + kwargs.pop("label", "")
		tau_rec = Pod.Metal(self, 
								ion,
								*args,
								**kwargs)
		vars(self)[name] = tau_rec

	def get_tau_rec_c2(self,**kwargs):
		"""
		Get the optical depth of CII using the fiducial recovery parameters: 
			correct_h1 = False
			correct_self = False
			take_min_doublet = False	
		Returns an object named self.c2.  

		See Pod class for optional input parameters. 
		"""
		self.get_tau_rec_ion("c2", 0, 0, 0, **kwargs)  

	def get_tau_rec_c3(self, **kwargs): 
		"""
		Get the optical depth of CIII using the fiducial recovery parameters: 
			correct_h1 = True 
			correct_self = False
			take_min_doublet = False	
		Returns an object named self.c3.  

		See Pod class for optional input parameters. 
		"""
		self.get_tau_rec_ion("c3", 1, 0, 0, **kwargs)

	def get_tau_rec_c4(self, **kwargs):
		"""
		Get the optical depth of CIV using the fiducial recovery parameters: 
			correct_h1 = False 
			correct_self = True 
			take_min_doublet = False	
		Returns an object named self.c4.  

		See Pod class for optional input parameters. 
		"""
		self.get_tau_rec_ion("c4", 0, 1, 0, **kwargs)

	def get_tau_rec_n5(self, **kwargs):
		"""
		Get the optical depth of NV using the fiducial recovery parameters: 
			correct_h1 = False 
			correct_self = False 
			take_min_doublet = True	
		Returns an object named self.n5.  

		See Pod class for optional input parameters. 
		"""
		self.get_tau_rec_ion("n5", 0, 0, 1, **kwargs)

	def get_tau_rec_o1(self, **kwargs):
		"""
		Get the optical depth of OI using the fiducial recovery parameters: 
			correct_h1 = False 
			correct_self = False 
			take_min_doublet = False 
		Returns an object named self.o1.

		See Pod class for optional input parameters. 
		"""
		self.get_tau_rec_ion("o1", 0, 0, 0, **kwargs)

	def get_tau_rec_o6(self, **kwargs):
		"""
		Get the optical depth of OVI using the fiducial recovery parameters: 
			correct_h1 = True 
			correct_self = False 
			take_min_doublet = True	
		Returns an object named self.o6.  

		See Pod class for optional input parameters. 
		"""
		self.get_tau_rec_ion("o6", 1, 0, 1, **kwargs)

	def get_tau_rec_si2(self, **kwargs):
		"""
		Get the optical depth of SiII using the fiducial recovery parameters: 
			correct_h1 = False 
			correct_self = False 
			take_min_doublet = False 
		Returns an object named self.si2.  

		See Pod class for optional input parameters. 
		"""
	
		self.get_tau_rec_ion("si2", 0, 0, 0, **kwargs)

	def get_tau_rec_si3(self,  **kwargs):
		"""
		Get the optical depth of SiIII using the fiducial recovery parameters: 
			correct_h1 = False 
			correct_self = False 
			take_min_doublet = False	
		Returns an object named self.si3.  

		See Pod class for optional input parameters. 
		"""
		self.get_tau_rec_ion("si3", 0, 0, 0, **kwargs)

	def get_tau_rec_si4(self, **kwargs):
		"""
		Get the optical depth of SiIV using the fiducial recovery parameters: 
			correct_h1 = False 
			correct_self = False 
			take_min_doublet = True	
		Returns an object named self.si4.  

		See Pod class for optional input parameters. 
		"""
		self.get_tau_rec_ion("si4", 0, 0, 1, **kwargs)

	def fit_continuum(self, 
					bin_size = 20.0,		
					n_cf_sigma = 2.0,
					max_iterations = 20):
		"""
		Automatically re-fit the continuum redward of the QSO Lya emission 
		in order to homogenize continuum fitting errors between different quasars
		"""
		if self.verbose:
			print "Fitting continuum..."
		# Set up the area to fit
		lambda_min = (1.0 + self.z_qso) * un.lambda_h1[0]
		index_correction = np.where(self.lambdaa > lambda_min)[0]	
		if len(index_correction) == 0:
			sys.exit("No pixels redward of LyA!")
		lambdaa = self.lambdaa[index_correction]
		flux = self.flux[index_correction].copy()
		sigma_noise = self.sigma_noise[index_correction].copy()
		flag = self.flag[index_correction].copy()
		# Get the bins 
		bin_size = bin_size * (1. + self.z_qso) 
		lambda_max = lambdaa[-1]
		nbins = np.floor((lambda_max - lambda_min) / bin_size)
		nbins = int(nbins)
		bin_size = (lambda_max - lambda_min) / nbins 
		if self.verbose:
			print "...using", nbins, "bins of", bin_size, "A"
		bin_edges = lambda_min + np.arange(nbins + 1) * bin_size  	
		pixels_list = [None] * nbins
		lambda_centre = np.empty(nbins)
		for i in range(nbins):
			pixels_list[i] = np.where((lambdaa > bin_edges[i]) & 
				(lambdaa <= bin_edges[i+1]))[0]
			lambda_centre[i] = bin_edges[i] + bin_size / 2.
		# Throw this all into a while loop until convergence
		flag_interp = flag.copy()
		flux_interp = flux
		converged = 0
		medians = np.empty(nbins)
		medians_flag = np.zeros(nbins, dtype=int)	
		niter = 0
		lambda_interp = lambda_centre 
		medians_interp = np.empty(nbins)
		while converged == 0 and niter < max_iterations:
			niter += 1
			for i in range(nbins):
				pixels = pixels_list[i] 
				lambda_bin = lambdaa[pixels] 
				flux_bin = flux[pixels]
				medians[i] = np.median(flux_bin[flag_interp[pixels] % 2 == 0])
				if (np.isnan(medians[i])):
					medians_flag[i] = 1
			for i in range(nbins):
				if medians_flag[i] == 1:
					medians[i] = medians[(i-1) % nbins]			
			medians_interp = medians
			# Interpolate
			flux_function = intp.splrep(lambda_interp, medians_interp, k = 3)
			flux_interp = intp.splev(lambdaa, flux_function)
			flag_interp_new = flag.copy()
			bad_pix = np.where((flux_interp - flux) > n_cf_sigma * sigma_noise)[0]
			flag_interp_new[bad_pix] = 1
			# Test for convergence 
			if (flag_interp_new == flag_interp).all():
				converged = 1
			flag_interp = flag_interp_new
		# Divide out the spectrum in the fitted part
		self.flux_old = self.flux.copy()
		self.sigma_noise_old = self.sigma_noise.copy()
		bad_pix = np.where(flag % 2 == 1)
		flux_interp[bad_pix] = flux[bad_pix] 
		self.flux[index_correction] = flux / flux_interp 
		self.sigma_noise[index_correction] = sigma_noise / flux_interp 
		self.flux_function = self.interp_f_lambda(self.lambdaa, self.flux)
		self.sigma_noise_function = self.interp_f_lambda(self.lambdaa, 
			self.sigma_noise)
		if self.verbose:
			print "...done\n"


	def plot_spectrum(self):
		"""	
		Quick view plot of QSO spectrum
		"""
		fig, ax = plt.subplots(figsize = (12, 3))	
		ax.plot(self.lambdaa, self.flux, lw = 0.5, c = 'b')
		ax.plot(self.lambdaa, self.sigma_noise, lw = 0.5, c = 'r')
		ax.axhline(y = 0, ls = ':', c = 'k', lw = 0.5)
		ax.axhline(y = 1, ls = ':', c = 'k', lw = 0.5)
		ax.set_ylim(-0.2, 1.2)
		ax.set_xlabel("$\lambda$ [\AA]")
		ax.set_ylabel("Normalized flux")
		ax.minorticks_on()
		fig.show()

class KodiaqFits(Spectrum):

	"""
	Createa spectrum from KODIAQ fits files. The flux file and error array, 
	respectively, should be located in the files:
	filepath + objectname + _f.fits 
	filepath + objectname + _e.fits 
	"""
	def __init__(self, 
				z_qso,
				object_name,
				**kwargs):
		Spectrum.__init__(self, z_qso, object_name, **kwargs)	
		# Read in flux file
		flux_filepath = self.filepath + object_name + "_f.fits" 
		flux_file = fits.open(flux_filepath)[0]
		self.flux = flux_file.data
		# Get wavelength array
		crpix1 = flux_file.header['CRPIX1'] # reference pixel 
		crval1 = flux_file.header['CRVAL1'] # coordinate at reference pixel
		cdelt1 = flux_file.header['CDELT1'] # coordinate increment per pixel
		length = flux_file.header['NAXIS1'] 
		self.lambdaa = 10**((np.arange(length) + 1.0 - crpix1) * cdelt1 + crval1)
		self.npix = length
		# Read in error file
		err_filepath = self.filepath + object_name + "_e.fits" 
		err_file = pf.open(err_filepath)[0]
		self.sigma_noise = err_file.data
		# Mask any bad regions or dlas
		self.mask_spectrum()
		# Prep spectrum
		self.prep_spectrum()
		print "*** Done ***\n"

class UserInput(Spectrum):
	"""
	Create a spectrum from user input. Unlike the KodiaqFits class, here the
	flux, wavelength and noise arrays need to be dirrectly supplied. 

	Parameters
	----------
	lambdaa: wavelength in angstroms
	flux: normalized flux array
	sigma_noise: error array 				
	"""

	def __init__(self,
				z_qso,
				lambdaa,
				flux,
				sigma_noise,
				**kwargs):
		Spectrum.__init__(self, z_qso, object_name, **kwargs)	
		self.flux = flux
		self.sigma_noise = sigma_noise
		self.lambdaa = lambdaa
		self.npix = len(lambdaa)
		# Mask any bad regions or dlas
		self.mask_spectrum()
		# Prep spectrum
		self.prep_spectrum()
		print "*** Done ***\n"
