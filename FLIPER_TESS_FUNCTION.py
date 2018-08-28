#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:17:00 2018

@author: Lisa Bugnet
@contact: lisa.bugnet@cea.fr
This code is the property of L. Bugnet (please see and cite Bugnet et al.,2018).
More info on the method are available on GitHub: https://github.com/lbugnet/FLIPER

The user should use the FLIPER class to calculate FliPer values
from 0.2,0.7,7,20 and 50 muHz.

A calling example is reported at the beginning of the code
"""
#---------------------------------------------------------
##### CALLING SEQUENCE:
#---------------------------------------------------------

'''
---------------------------------------------------------
DATA PREPARATION    (OPTIONNAL: COMPUTE PSD FROM LIGHT CURVES BY USING THE CONVERT CLASS)

star_path_LC='/PATH_TO_STARS/Star1.clean'
star_number=int(PATH_DATA[ii].replace(DATA+'Star', '').replace('.clean',''))
flux=CONVERT().get_ts(star_path_LC)[1]
time=CONVERT().get_ts(star_path_LC)[0]
star_tab_psd=CONVERT().compute_ps(time, flux, star_path_LC, star_number)

FLIPER CALCULATION

Fliper_values=FLIPER().Fp(star_tab_psd)
print(Fliper_values.fp07[0])
print(Fliper_values.fp7[0])
print(Fliper_values.fp20[0])
print(Fliper_values.fp50[0])
---------------------------------------------------------
'''


from __future__ import division
from astropy.io import fits
import numpy as np
import os, os.path
from montecarlo import montecarlo
from math import *
import gatspy.periodic as gp
from gatspy.periodic import LombScargleFast

class FLIPER:
#"""Class defining the FliPer"""

    def __init__(self):
        self.nom = "FliPer"
        self.id=[]
        self.fp07=[]
        self.fp7=[]
        self.fp20=[]
        self.fp02=[]
        self.fp50=[]

    def Fp(self, star_tab_psd):
        """
        Compute FliPer values from 0.7, 7, 20, & 50 muHz
        star_tab_psd[0] contains frequencies in muHz
        star_tab_psd[1] contains power density in ppm2/muHz
        star_tab_psd can be computed using the CONVERT class if needed
        """
        end         =   277# muHz
        noise       =   self.HIGH_NOISE(star_tab_psd)
        Fp07_val    =   self.REGION(star_tab_psd, 0.7, end) - noise
        Fp7_val     =   self.REGION(star_tab_psd, 7, end)   - noise
        Fp20_val    =   self.REGION(star_tab_psd, 20, end)  - noise
        Fp50_val    =   self.REGION(star_tab_psd, 50, end)  - noise

        self.fp07.append(Fp07_val)
        self.fp7.append(Fp7_val)
        self.fp20.append(Fp20_val)
        self.fp50.append(Fp50_val)
        return self

    def HIGH_NOISE(self, star_tab_psd):
        """
        Function that computes photon noise from last 100 bins of the spectra
        """
        data_arr_freq   =   star_tab_psd[0]
        data_arr_pow    =   star_tab_psd[1]
        siglower        =   np.mean(data_arr_pow[-100:])
        return siglower

    def REGION(self,star_tab_psd,inic,end):
        """
        Function that calculates the average power in a given frequency range on PSD
        """
        x       =   np.float64(star_tab_psd)[0]
        y       =   np.float64(star_tab_psd)[1]
        ys      =   y[np.where((x >= inic) & (x <= end))]
        average =   np.mean(ys)
        return average





class CONVERT:
    def __init__(self):
        self.psd_path   =   []
        self.psd        =   [[],[]]
        self.LC         =   [[],[]]
        self.kic        =   []

    def LC_PATH_TO_LC(self, star_path_LC):
        """
        Extract light curve from path
        """
        tab     =   [x.split() for x in open(star_path_LC).readlines()]
        tab     =   tab[5:]
        freq_tab    =   [tab[:][ii][0] for ii in range(0, len(tab)) if tab[:][ii][1]!='nan']
        power_tab   =   [tab[:][ii][1] for ii in range(0, len(tab)) if tab[:][ii][1]!='nan']
        freq_tab    =   np.asarray([float(i) for i in freq_tab])
        power_tab   =   np.asarray([float(i) for i in power_tab])
        star_tab_LC =   np.column_stack((freq_tab,power_tab))
        self.LC    +=   list(map(list, zip(*star_tab_LC)))
        return star_tab_LC

    def get_ts(self, star_path_LC):
        """
        Import data
        """
        time, flux, flag    =   np.loadtxt(star_path_LC, unpack=True)
        tottime     =   np.max(time)-np.min(time)
        flux        =   ((flux/np.nanmedian(flux))-1)*1e6       #convert flux from e- to ppm !!!!
        sel         =   np.where(np.isnan(flux)==True)
        flux[sel]   =   0.0
        time       -=   time[0]
        time       *=   86400.0         #put in sec
        cadence     =   np.median(np.diff(time))
        return time, flux, cadence, tottime

    def normalise(self, time, flux, f, p, bw):
        """
        Normalise according to Parseval's theorem
        """
        rhs     =   1.0 / len(flux) * np.sum(flux**2.0)
        lhs     =   p.sum()
        ratio   =   rhs / lhs
        return p * ratio / bw / 1e6


    def compute_ps(self, time, flux, star_path_LC_20, ind):
        """
        Compute power spectrum using gatspy fast lomb scargle
        """
        time, flux, dt, tottime     =   self.get_ts(star_path_LC_20)
        dt      =   float(dt)
        tottime =   float(tottime)*86400.0#put in sec
        # Nyquist frequency
        nyq     =   1.0 / (2.0*dt)
        # Frequency bin width
        df      =   1.0 / tottime
        # Number of frequencies to compute
        Nf      =   nyq / df
        # Compute psd
        f, p    =   gp.lomb_scargle_fast.lomb_scargle_fast(time, flux,
                                                      f0=df,
                                                      df=df, Nf=Nf,
                                                      use_fft=True)
        # Calibrate power
        p       =   self.normalise(time, flux, f, p, df)
        return f*1e6, p #frequencies in muHz
