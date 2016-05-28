"""
podpy is an implementatin of the pixel optical depth method as described in 
Turner et al. 2014, MNRAS, 445, 794, and Aguirre et al. 2002, ApJ, 576, 1. 
Please contact the author (Monica Turner) at turnerm@mit.edu if you have 
any questions, comment or issues. 
"""

import numpy as np

# Rest wavelength source: Morton 2003, ApJS, 149, 205
# Oscillator strength source: Morton 2003, ApJS, 149, 205 
#                                Verner et al 1994, AAPS, 108, 287

# Lyman lines
lambda_h1 = np.array([1215.6701, 1025.7223, 972.5368, 949.7431, 
       937.8035, 930.7483, 926.2257, 923.1504, 
       920.9631, 919.3514, 918.1294, 917.1806, 
       916.429, 915.824, 915.329, 914.919, 
       914.576, 914.286, 914.039, 913.826, 
       913.641, 913.480, 913.339, 913.215, 
       913.104, 913.006, 912.918, 912.839, 
       912.768, 912.703, 912.645]) 
f_h1 = np.array([0.416400, 0.079120, 0.029000, 0.013940, 
       0.007799, 0.004814, 0.003183, 0.002216, 
       0.001605, 0.00120,  0.000921, 7.226e-4,
       0.000577, 0.000469, 0.000386, 0.000321, 
       0.000270, 0.000230, 0.000197, 0.000170, 
       0.000148, 0.000129, 0.000114, 0.000101, 
       0.000089, 0.000080, 0.000071, 0.000064, 
       0.000058, 0.000053, 0.000048]) 
g_h1 = f_h1 * lambda_h1

lambda_h1_limit = 911.8

# Metals
# OVI
lambda_o6 = np.array([1031.927, 1037.616])
f_o6 = np.array([0.132900, 0.066090])
g_o6 = f_o6 * lambda_o6
# NV
lambda_n5 = np.array([1238.821, 1242.804])
f_n5 = np.array([0.157000, 0.078230])
g_n5 = f_n5 * lambda_n5
# CIV
lambda_c4 = np.array([1548.195, 1550.770])
f_c4 = np.array([0.190800, 0.095220])
g_c4 = f_c4 * lambda_c4
# CIII
lambda_c3 =  np.array([977.020])
f_c3 = np.array([0.7620])	
g_c3 = f_c3 * lambda_c3
# CII
lambda_c2 = np.array([1334.5323, 1036.3367]) 
f_c2 = np.array([ 0.1278, 0.1231])  
g_c2 = f_c2 * lambda_c2
# SiIV
lambda_si4 = np.array([1393.755, 1402.770])
f_si4 = np.array([0.5140, 0.2553])
g_si4 = f_si4 * lambda_si4
# SiIII
lambda_si3 = np.array([1206.500])	
f_si3 = np.array([1.669000])
g_si3 = f_si3 * lambda_si3
# SiII
lambda_si2 = np.array([1260.4221, 1193.2897, 1190.4158, 989.8731,
	1526.7066, 1304.3702, 1020.6989])
f_si2 = ([1.007000, 0.499100, 0.250200, 0.133000, 
	0.11600, 0.09400, 0.028280]) 
g_si2 = f_si2 * lambda_si2
# MgII
lambda_mg2 = np.array([2796.352, 2803.531])
f_mg2 = np.array([0.6123, 0.3054])
g_mg2 = f_mg2 * lambda_mg2
# OI
lambda_o1 = np.array([1302.1685, 988.7734])
f_o1 = np.array([0.048870, 0.043180])
g_o1 = f_o1 * lambda_o1
# Fe2
lambda_fe2 = np.array([1144.9379, 1608.45085, 1063.1764, 1096.8769,
       1260.533, 1121.9748, 1081.8748, 1143.2260, 1125.4477]) 
f_fe2 = np.array([0.083, 0.0577, 0.0547, 0.032700, 
       0.024000, 0.0290, 0.012600, 0.0192, 0.0156])
g_fe2 = f_fe2 * lambda_fe2


# Constants 
c =  299792.458 # speed of light in km/s 

