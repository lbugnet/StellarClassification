import numpy as np
import os
import glob

class FeatCalc(object):
    def __init__(self, metadata='Data_Batch_TDA3_all.txt', 
                 lclistcall='noisy/lightcurves/Star*.noisy', 
                 freqdir='noisy/frequencies',
                 somfile=None,
                 cardinality=64, dimx=1, dimy=400,
                 nfrequencies=6):
        ''' Initialiser for class to bring together and calculate features 
        	for classification
        
        Parameters
        ----------------
        metadata: str
            Filepath leading to the metadata file, e.g. Data_Batch1_noisy.txt
            
        lclistcall: str
            A call to ls to recover all the lightcurve datafiles
            
        freqdir: str
        	Directory containing the frequency files as output by Vicki et al.
        	      	
        somfile: str
        	Filepath to saved SOM Kohonen Layer. Can be left blank, but self.calc()
        	will not run

        cardinality: 	int, default 64
        	Number of bins in each SOM pixel
        	
        dimx:			int, default 1
        	size of SOM x dimension
        
        dimy:			int, default 400
        	size of SOM y dimension
        
        n_frequencies: int
        	Number of frequencies to use, starting with the strongest
            
        Returns
        -----------------
        NA
        
        Examples
        -----------------
        A typical use of the class/function would follow:
        	A = FeatCalc()
        	A.calc()
        	print(A.features)
        	np.savetxt(outputfile,A.features)
        	
        '''
        #get lightcurve list, input files
        self.metadata = np.genfromtxt(metadata,delimiter=',',dtype=None)
        self.inlist = glob.glob(lclistcall)
        self.freqdir = freqdir
        self.nfreq = nfrequencies
        self.features = np.zeros([len(self.inlist),self.nfreq+14])
        self.dimx = dimx
        self.dimy = dimy
        self.cardinality = cardinality
        if somfile:
            self.som = self.loadSOM(somfile, dimx, dimy, cardinality)
        else:
            self.som = None
        self.forbiddenfreqs = [13.49/4.] #(4 harmonics will be removed)
        self.ids = self.metadata['f0'].astype('unicode')
        self.types = self.metadata['f10'].astype('unicode')
        self.teff = self.metadata['f6']
        self.logg = self.metadata['f8']
     
    def calc(self):
        ''' Function to calculate features and populate the features array
        
        Parameters
        ----------------
		NA
		            
        Returns
        -----------------
        No return, but leaves self.features array populated
        '''
        if not self.som:
            print('No SOM defined, exiting')
            return 0
        for i,infile in enumerate(self.inlist):
            dotindex = os.path.split(infile)[1].find('.')
            self.activeid = os.path.split(infile)[1][:dotindex]     
            print(self.activeid)
            
            try:
                #load
                self.lc = self.loadLC(infile)
        
                #load relevant frequencies file
                self.freqdat = self.loadFreqs()

                #extract/calc frequencies
                freqs = self.frequencies()
                EBper = self.EBperiod(freqs[0])
            
                #calc features
                self.features[i,0] = int(self.activeid[4:])
                self.features[i,1:self.nfreq+1] = freqs
                self.features[i,self.nfreq+1:self.nfreq+3] = self.freq_ampratios()
                self.features[i,self.nfreq+3:self.nfreq+5] = self.freq_phasediffs()
                self.features[i,1] = EBper
                self.features[i,self.nfreq+5:self.nfreq+7] = self.SOMloc(EBper)
                self.features[i,self.nfreq+7:self.nfreq+9] = self.phase_features(EBper)
                self.features[i,self.nfreq+9:self.nfreq+11] = self.p2p_features()
                self.features[i,self.nfreq+11] = self.get_Teff()
                #try:
                self.features[i,self.nfreq+12:] = self.guy_features(infile)
                #except: #a problem for clean files that don't vary
                    #self.features[i,self.nfreq+12:] = np.random.normal(0,0.5,2)
            except IOError:
                print('Skipped, lc or frequency file not found')
        self.features = self.features[np.argsort(self.features[:,0]),:] #sort by id

    def SOM_alldataprep(self, outfile=None):
        ''' Function to create an array of normalised lightcurves to train a SOM
        
        Parameters
        ----------------
		outfile:		str, optional
			Filepath to save array to. If not populated, just returns array
		            
        Returns
        -----------------
        SOMarray:		np array, [n_lightcurves, cardinality]
        	Array of phase-folded, binned lightcurves
        '''    
        SOMarray = np.ones(self.cardinality)
        for i,infile in enumerate(self.inlist):
            dotindex = os.path.split(infile)[1].find('.')
            self.activeid = os.path.split(infile)[1][:dotindex]
            print(self.activeid)
            
            try:
                #load lc
                self.lc = self.loadLC(infile, linflatten=True)
        
                #load relevant frequencies file
                self.freqdat = self.loadFreqs()
            
                #extract/calc frequencies
                freqs = self.frequencies()
                EBper = self.EBperiod(freqs[0])
            
                if EBper > 0: #ignores others
                    binlc,range = self.prepFilePhasefold(EBper,self.cardinality)
                    SOMarray = np.vstack((SOMarray,binlc))
            except IOError:
                print('Skipped, lc or frequency file not found')
        if outfile:
            np.savetxt(outfile,SOMarray[1:,:])
        return SOMarray[1:,:] #drop first line as this is just ones
    
    def SOM_train(self, SOMarrayfile, outfile=None, nsteps=300, learningrate=0.1):
        ''' Function to train a SOM
        
        Parameters
        ----------------
		SOMarrayfile:	str
			Filepath to txt file containing SOMarray

		outfile:		str, optional
			Filepath to save array to. If not populated, just returns array
		            
        nsteps:			int, optional
        	number of training steps for SOM
        
        learningrate:	float, optional
        	parameter for SOM, controls speed at which it changes. Between 0 and 1.
        	
        Returns
        -----------------
        som object:		object
        	Trained som
        '''
        import selfsom
        
        SOMarray = np.genfromtxt(SOMarrayfile)
        cardinality = SOMarray.shape[1]
        
        def Init(sample):
            return np.random.uniform(0,1,size=(self.dimx,self.dimy,cardinality))
        
        som = selfsom.SimpleSOMMapper((self.dimx,self.dimy),nsteps,initialization_func=Init,
        							  learning_rate=0.1)
        som.train(SOMarray)
        if outfile:
            self.kohonenSave(som.K,outfile)
        self.som = som
        return som
        
    def loadLC(self,infile, linflatten=False):
        """
        Loads a TESS lightcurve (currently just the TASC WG0 simulated ones), 
        normalised and with NaNs removed.
        
        Inputs
        -----------------
        infile: 	str
        	Filepath to one lightcurve file
        	
        linflatten: bool
        	Remove a linear fit to the flux?
        	
        Returns
        -----------------
        lc: 		dict
         	lightcurve as dict, with keys time, flux, error. 
        	error is populated with zeros.
        """
        dat = np.genfromtxt(infile)
        time = dat[:,0]
        flux = dat[:,1]
        err = np.zeros(len(time))
        nancut = np.isnan(time) | np.isnan(flux)
        norm = np.median(flux[~nancut])
        lc = {}
        lc['time'] = time[~nancut]
        lc['flux'] = flux[~nancut]/norm
        lc['error'] = err[~nancut]/norm
        del dat
        
        if linflatten:
            lc['flux'] = lc['flux'] - np.polyval(np.polyfit(lc['time'],lc['flux'],1),lc['time']) + 1
        return lc
            
    def loadFreqs(self):
        """
        Loads frequencies given the class frequency dir and the current active ID.
        	
        Returns
        -----------------
        freqdat: ndarray
        	Frequencies for object, strongest first.
        """    
        freqfile = os.path.join(self.freqdir,self.activeid)+'.slscleanlog'
        freqdat = np.genfromtxt(freqfile)
        if freqdat.shape[0] == 0:
            freqdat = np.zeros([2,9]) - 10        
        elif len(freqdat.shape) == 1: #only one significant peak
            temp = np.zeros([2,len(freqdat)])
            temp[0,:] = freqdat
            temp[1,:] -= 10
            freqdat = temp
        return freqdat
    
    def loadSOM(self, somfile, dimx=1, dimy=400, cardinality=64):
        """
        Loads a previously trained SOM.
        
        Inputs
        -----------------
        somfile: 		str
        	Filepath to saved SOM (saved using self.kohonenSave)
        	
        cardinality: 	int, default 64
        	Number of bins in each SOM pixel
        	
        dimx:			int, default 1
        	size of SOM x dimension
        
        dimy:			int, default 400
        	size of SOM y dimension
        	
        Returns
        -----------------
        som:	 object
         	Trained som object
        """
        def Init(sample):
            '''
            Initialisation function for SOM.
            '''
            return np.random.uniform(0,1,size=(dimx,dimy,cardinality))

        import selfsom
        som = selfsom.SimpleSOMMapper((dimx,dimy),1,initialization_func=Init,learning_rate=0.1)
        loadk = self.kohonenLoad(somfile)
        som.train(loadk)  #purposeless but tricks the SOM into thinking it's been trained. Don't ask.
        som._K = loadk
        return som

    def kohonenSave(self,layer,outfile):  #basically a 3d >> 2d saver
        """
        Takes a 3d array and saves it to txt file in a recoverable way.
        
        Inputs
        -----------------
        layer: 		ndarray, 3 dimensional, size [i,j,k]
        	Array to save.
        	
        outfile: 	str
        	Filepath to save to.
        """
        with open(outfile,'w') as f:
            f.write(str(layer.shape[0])+','+str(layer.shape[1])+','+str(layer.shape[2])+'\n')
            for i in range(layer.shape[0]):
                for j in range(layer.shape[1]):
                    for k in range(layer.shape[2]):
                        f.write(str(layer[i,j,k]))
                        if k < layer.shape[2]-1:
                            f.write(',')
                    f.write('\n')
        
    def kohonenLoad(self, infile):
        """
        Loads a 3d array saved with self.kohonenSave(). Auto-detects dimensions.
        
        Inputs
        -----------------
        infile: str
        	Filepath to load
        	
        Returns
        -----------------
        out: ndarray, size [i,j,k]
         	Loaded array.
        """
        with open(infile,'r') as f:
            lines = f.readlines()
        newshape = lines[0].strip('\n').split(',')
        out = np.zeros([int(newshape[0]),int(newshape[1]),int(newshape[2])])
        for i in range(int(newshape[0])):
            for j in range(int(newshape[1])):
                line = lines[1+(i*int(newshape[1]))+j].strip('\n').split(',')
                for k in range(int(newshape[2])):
                    out[i,j,k] = float(line[k])
        return out
            
    def SOMloc(self, per): #return SOM location and phase binned amplitude
        """
        Returns the location on the current som using the current loaded lc.
        
        Inputs
        -----------------
        per: 			float
        	Period to phasefold the lightcurve at.

        Returns
        -----------------
        map: 	int
         	Location on SOM (assumes 1d SOM).
        range: float
        	Amplitude of the binned phase-folded lightcurve
        """
        if per < 0:
            return -10
        SOMarray,range = self.prepFilePhasefold(per,self.cardinality)
        SOMarray = np.vstack((SOMarray,np.ones(len(SOMarray)))) #tricks som code into thinking we have more than one
        map = self.som(SOMarray)
        map = map[0,1]
        return map, range

    def get_Teff(self):
        """
        Returns effective temperature from TDA simulated data info file.
        """
        idx = np.where(self.ids==self.activeid.strip('.'))[0][0]
        return self.teff[idx]
        
    def frequencies(self): #return nfrequencies strongest frequencies, ignoring those near set forbidden frequencies
        """
        Cuts frequency data down to desired number of frequencies, and removes harmonics
        of forbidden frequencies
        	
        Returns
        -----------------
        freqs: ndarray [self.nfreqs]
         	array of frequencies
        """
        freqs = []
        self.usedfreqs = []
        j = 0
        while len(freqs)<self.nfreq:
            freq = 1./(self.freqdat[j,1]*1e-6)/86400.  #convert to days
            #print freq
            #check to cut bad frequencies
            cut = False
            if (freq < 0) or (freq > np.max(self.lc['time'])-np.min(self.lc['time'])):  #means there weren't even one or two frequencies, or frequency too long
                cut = True
            for freqtocut in self.forbiddenfreqs:
                for k in range(4):  #cuts 4 harmonics of frequency, within +-3% of given frequency
                    if (1./freq > (1./((k+1)*freqtocut))*(1-0.01)) & (1./freq < (1./((k+1)*freqtocut))*(1+0.01)):
                        cut = True
            if not cut:
                freqs.append(freq) #convert microhertz to days
                #print freqs[-1]
                self.usedfreqs.append(j)
            j += 1
            if j >= len(self.freqdat[:,0]):
                break
        #fill in any unfilled frequencies with negative numbers
        gap = self.nfreq - len(freqs)
        if gap > 0:
            for k in range(gap):
                freqs.append(-10)
        self.usedfreqs = np.array(self.usedfreqs)
        return np.array(freqs)
        
    def freq_ampratios(self): #return 2 ampratios - 2:1 and 3:1
        """
        Amplitude ratios of frequencies
          	
        Returns
        -----------------
        amp21, amp31: float, float
        	ratio of 2nd to 1st and 3rd to 1st frequency amplitudes
         	
        """
        if len(self.usedfreqs) >= 2:
            amp21 = self.freqdat[self.usedfreqs[1],3]/self.freqdat[self.usedfreqs[0],3]
        else:
            amp21 = 0
        if len(self.usedfreqs) >= 3:
            amp31 = self.freqdat[self.usedfreqs[2],3]/self.freqdat[self.usedfreqs[0],3]
        else:
            amp31 = 0
        return amp21,amp31
        
    def freq_phasediffs(self):  #return 2 phase diffs - 2:1 and 3:1
        """
        Phase differences of frequencies
          	
        Returns
        -----------------
        phi21, phi31: float, float
        	phase difference of 2nd to 1st and 3rd to 1st frequencies
         	
        """
        if len(self.usedfreqs) >= 2:
            phi21 = self.freqdat[self.usedfreqs[1],5] - 2*self.freqdat[self.usedfreqs[0],5]
        else:
            phi21 = -10
        if len(self.usedfreqs) >= 3:
            phi31 = self.freqdat[self.usedfreqs[2],5] - 3*self.freqdat[self.usedfreqs[0],5]
        else:
            phi31 = -10
        return phi21,phi31    
    
    def phase_features(self, per):  #return phase p2p 98 perc, phase p2p mean
        """
        Returns p2p features connected to phase fold

        Inputs
        -----------------
        per: 			float
        	Period to phasefold self.lc at.
          	
        Returns
        -----------------
        p2p 98th percentile: 	float
        	98th percentile of point-to-point differences of phasefold
        p2p mean:				float
        	Mean of point-to-point differences of phasefold
         	
        """
        phase = self.phasefold(self.lc['time'],per)
        p2p = np.abs(np.diff(self.lc['flux'][np.argsort(phase)]))
        return np.percentile(p2p,98),np.mean(p2p)
    
    def p2p_features(self):  #return p2p 98 perc, p2p mean
        """
        Returns p2p features on self.lc

        Returns
        -----------------
        p2p 98th percentile: 	float
        	98th percentile of point-to-point differences of lightcurve
        p2p mean:				float
        	Mean of point-to-point differences of lightcurve
         	
        """
        p2p = np.abs(np.diff(self.lc['flux']))
        return np.percentile(p2p,98),np.mean(p2p)
    
    def EBperiod(self, per, cut_outliers=0):
        """
        Tests for phase variation at double the current prime period,
        to correct EB periods.

        Inputs
        -----------------
        per: 			float
        	Period to phasefold self.lc at.
        cut_outliers:	float
        	outliers ignored if difference from median in bin divided by the MAD is 
        	greater than cut_outliers.
        	        
        Returns
        -----------------
        corrected period: float
        	Either initial period or double      
        """
        if per < 0:
            return per
        flux_flat = self.lc['flux'] - np.polyval(np.polyfit(self.lc['time'],self.lc['flux'],1),self.lc['time']) + 1
        
        phaselc2P = np.zeros([len(self.lc['time']),2])
        phaselc2P = self.phasefold(self.lc['time'],per*2)
        idx = np.argsort(phaselc2P)
        phaselc2P = phaselc2P[idx]
        flux = flux_flat[idx]
        binnedlc2P = self.binPhaseLC(phaselc2P,flux,64,cut_outliers=5) #ADD OUTLIER CUTS?

        minima = np.argmin(binnedlc2P[:,1])
        posssecondary = np.mod(np.abs(binnedlc2P[:,0]-np.mod(binnedlc2P[minima,0]+0.5,1.)),1.)
        posssecondary = np.where((posssecondary<0.05) | (posssecondary > 0.95))[0]   #within 0.05 either side of phase 0.5 from minima

        pointsort = np.sort(self.lc['flux'])
        top10points = np.median(pointsort[-30:])
        bottom10points = np.median(pointsort[:30])
        
        periodlim = (np.max(self.lc['time'])-np.min(self.lc['time']))/2. #no effective limit, could be changed
        #print 'EBP'
        #print np.min(binnedlc2P[posssecondary,1]) - binnedlc2P[minima,1]
        #print 0.03*(top10points-bottom10points)
        if np.min(binnedlc2P[posssecondary,1]) - binnedlc2P[minima,1] > 0.0025 and np.min(binnedlc2P[posssecondary,1]) - binnedlc2P[minima,1] > 0.03*(top10points-bottom10points) and per*2<=periodlim:  
            return 2*per
        else:
            return per

    def get_metric(self, low_f=1.0, high_f=288.0, white_npts=100):
        """
        Features from Guy Davies  
        """
        white = np.median(self.ds.power[-white_npts:])
        mean = np.mean(self.ds.power[(self.ds.freq > low_f) \
                                      & (self.ds.freq < high_f)])
        med = np.median(self.ds.power[(self.ds.freq > low_f) \
                                      & (self.ds.freq < high_f)])
        std = np.std(self.ds.power[(self.ds.freq > low_f) \
                                      & (self.ds.freq < high_f)])
        return white, mean, med, std


    def guy_features(self,infile):
        """
        Features from Guy Davies  
        """
        from TDAdata import Dataset
        self.ds = Dataset(self.activeid, infile)
        vars_dict = {'Flicker': [['w100', 'mean1', 'med1', 'std1'],[0.5, 288.0, 100]]}
        self.ds.read_timeseries(sigma_clip=3.0)
        self.ds.power_spectrum()
        tmp = vars_dict['Flicker'][1]
        w, m, mm, s = self.get_metric(low_f=tmp[0], high_f=tmp[1], white_npts=tmp[2])
        return w, m-w
        
    def binPhaseLC(self,phase, flux, nbins, cut_outliers=0):
        """
        Bins a lightcurve, typically phase-folded.

        Inputs
        -----------------
        phase: 			ndarray, N
        	Phase data (could use a time array instead)
        flux:			ndarray, N
        	Flux data
        nbins:			int
        	Number of bins to use
        cut_outliers:	float
        	If not zero, cuts outliers where (difference to median)/MAD > cut_outliers 
        	        
        Returns
        -----------------
        binnedlc:		ndarray, (nbins, 2)    
        	Array of (bin phases, binned fluxes)
        """
        bin_edges = np.arange(nbins)/float(nbins)
        bin_indices = np.digitize(phase,bin_edges) - 1
        binnedlc = np.zeros([nbins,2])
        binnedlc[:,0] = 1./nbins * 0.5 +bin_edges  #fixes phase of all bins - means ignoring locations of points in bin, but necessary for SOM mapping
        for bin in range(nbins):
            inbin = np.where(bin_indices==bin)[0]
            if cut_outliers:
                mad = np.median(np.abs(flux[inbin]-np.median(flux[inbin])))
                outliers = np.abs((flux[inbin] - np.median(flux[inbin])))/mad <= cut_outliers
                inbin = inbin[outliers]
            if np.sum(bin_indices==bin) > 0:
                binnedlc[bin,1] = np.mean(flux[inbin])  #doesn't make use of sorted phase array, could probably be faster?
            else:
                binnedlc[bin,1] = np.mean(flux)  #bit awkward this, but only alternative is to interpolate?
        return binnedlc

    def phasefold(self,time,per,t0=0):
        return np.mod(time-t0,per)/per

    def prepFilePhasefold(self, period, cardinality):
        """
        Prepares a lightcurve for using with the SOM.

        Inputs
        -----------------
        period: 			float
        	Period to phasefold self.lc at
        cardinality:		int
        	Number of bins used in SOM
        	        
        Returns
        -----------------
        binnedlc:		ndarray, (cardinality, 2)    
        	Array of (bin phases, binned fluxes)
        range:			float
        	Max - Min if binned lightcurve
        """
        phase = self.phasefold(self.lc['time'],period)
        idx = np.argsort(phase)
        flux = self.lc['flux'][idx]
        phase = phase[idx]
        binnedlc = self.binPhaseLC(phase,flux,cardinality)
        #normalise to between 0 and 1
        minflux = np.min(binnedlc[:,1])
        maxflux = np.max(binnedlc[:,1])
        if maxflux != minflux:
            binnedlc[:,1] = (binnedlc[:,1]-minflux) / (maxflux-minflux)
        else:
            binnedlc[:,1] = np.ones(cardinality)
        #offset so minimum is at phase 0
        binnedlc[:,0] = np.mod(binnedlc[:,0]-binnedlc[np.argmin(binnedlc[:,1]),0],1)
        binnedlc = binnedlc[np.argsort(binnedlc[:,0]),:]
        return binnedlc[:,1],maxflux-minflux

    def plot_diagnostic(self,target=None):
        """
        Plots lightcurves and phasefolds for inputs, for testing.

        Inputs
        -----------------
        target: 	list
        	list of ids to show. If not provided, shows all one-by-one
        """
        go = False
            
        for i,infile in enumerate(self.inlist):
            dotindex = os.path.split(infile)[1].find('.')
            self.activeid = os.path.split(infile)[1][:dotindex]
            if target:
                if self.activeid in target:
                    go = True
            else:
                go = True
                
            if go:
                print(self.activeid)
            
                #load lc
                self.lc = self.loadLC(infile, linflatten=True)
        
                #load relevant frequencies file
                self.freqdat = self.loadFreqs()
            
                #extract/calc frequencies
                freqs = self.frequencies()
                EBper = self.EBperiod(freqs[0])
            
                import pylab as p
                p.ion()
                p.figure(1)
                p.clf()
                phase = np.mod(self.lc['time'],EBper)/EBper
                p.plot(phase, self.lc['flux'],'r.')
                p.pause(1)
                p.figure(2)
                p.clf()
                p.plot(self.lc['time'],self.lc['flux'],'b.')
                print(freqs[0])
                print(EBper)
                idx = np.where(self.ids==self.activeid)[0]
                print(self.types[idx])
                p.pause(1)
                raw_input()
                go = False