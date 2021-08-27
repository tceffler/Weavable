#!/usr/bin/env python

import numpy as np
import os, sys

if len(sys.argv) != 2:
	print "ERROR: USAGE - %s <fwq_times.dat>" % sys.argv[0]
	sys.exit(1)

datfile = sys.argv[1]
try:
	fwqd = np.loadtxt(datfile, dtype=np.float64, usecols=(0,), unpack=True)                                                                                 
except:
	print "ERROR: Failed to load datfile %s" % datfile
	sys.exit(1)

fwqd_min = np.min(fwqd)
fwqd_scaled = (fwqd - fwqd_min) / fwqd_min

fwqs_max = np.max(fwqd_scaled)
fwqs_avg = np.average(fwqd_scaled)
fwqs_std = np.std(fwqd_scaled, dtype=np.float64)
#print "Data file [%s] scaled noise: max=%s avg=%s stddev=%s" % (datfile, fwqs_max, fwqs_avg, fwqs_std)

metric = 0.005
if metric > fwqs_std:
	print "Data file [%s] check successful: noise stddev < %s" % (datfile, metric)
	sys.exit(0)
else:
	print "Data file [%s] scaled noise: max=%s avg=%s stddev=%s" % (datfile, fwqs_max, fwqs_avg, fwqs_std)
	print "Data file [%s] check failed: noise stddev > %s" % (datfile, metric)
	sys.exit(1)
