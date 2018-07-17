#!/bin/env python

from __future__ import division, print_function
import numpy, pylab, math
import icecube
from icecube import dataio
from icecube import dataclasses
from icecube import CascadeVariables
import os

#inputs
filelist="SigFiles.txt"
NFiles=10

#Arb params
WFLength=1000
FFTLength=501
Offset=50
todo=10
NoiseFloor=0.05
MaxDistance=250

#Set the GCD to use
gcd=dataio.I3File("/data/sim/IceCube/2011/filtered/level2/neutrino-generator/10099/00000-00999/GeoCalibDetectorStatus_IC86.55697_corrected_V2.i3.gz")

#Extract geometry object
gcd.rewind()
fr=gcd.pop_frame()
geo=fr.Get("I3Geometry")

#Load files

sigFilesI3=numpy.loadtxt(filelist,dtype='string')[0:NFiles]


#Load the profiles and calculate kernels
import cPickle
KernelsFile=open("DOMKernelsLong.dat",'r')
WFDictionary=cPickle.load(KernelsFile)
KernelsFile.close()
FilterDictionary={}
KernelDictionary={}
for key in WFDictionary.keys():
    SignalKernel=numpy.fft.rfft(WFDictionary[key])
    KernelDictionary[key]=SignalKernel
    FilterDictionary[key]=numpy.abs(SignalKernel)*numpy.abs(SignalKernel)/(numpy.abs(SignalKernel)*numpy.abs(SignalKernel)+NoiseFloor*NoiseFloor)
DistanceLims=KernelDictionary.keys()


#Extract the waveforms and distances
Count=0    
Distances=[]
Waveforms=[]
Energies=[]
Times=[]
Charges=[]
MaxWFTimes=[]
BinWidths=[]

EnergyLimLow=5e3
EnergyLimHigh=1e7

fout=dataio.I3File("output.i3",'w')

for sgFile in sigFilesI3:
    Count=Count+1
    print("opening "+sgFile+",  "+str(Count)+" of " + str(len(sigFilesI3)))
    sigFile=dataio.I3File(sgFile)
    sigFile.rewind()
    while(sigFile.more()):
        #pop a frame
        fr=sigFile.pop_physics()

        #grab the weight dict and event weight
        WeightDict=fr.Get("I3MCWeightDict")
        Weight=(365*24*3600)*1.0e-8*pow(WeightDict["PrimaryNeutrinoEnergy"], -2)*WeightDict["OneWeight"]/(WeightDict["NEvents"]*len(sigFilesI3))

        #Find the cascade center
        if not fr.Has("CascadeLlhVertexFit") :
            continue

        if( WeightDict["PrimaryNeutrinoEnergy"]<EnergyLimLow) or (WeightDict["PrimaryNeutrinoEnergy"]>EnergyLimHigh):    
            continue
        csc=fr.Get('CascadeLlhVertexFit')
        cscpos=csc.pos

        wf=fr.Get('CalibratedWaveformsHLCATWD')
        DistancesThisEvt=[]
        WaveformsThisEvt=[]
        TimesThisEvt=[]
        ChargesThisEvt=[]
        BinWidthsThisEvt=[]
        for om, wf_series in wf:
            for w in wf_series:
                if(w.status==0):
                    DOMPos=geo.omgeo[om].position
                    DistancesThisEvt.append((cscpos-DOMPos).r)
                    WaveformsThisEvt.append(w.waveform)
                    TimesThisEvt.append(w.time)
                    ChargesThisEvt.append(sum(w.waveform))
                    BinWidthsThisEvt.append(w.bin_width)
                    break
        StartTime=numpy.min(TimesThisEvt)
        TimeShiftsThisEvt =(TimesThisEvt-StartTime)/BinWidthsThisEvt

        TotalFilteredFFT=numpy.zeros(FFTLength,dtype='complex')
        TotalSumWF=numpy.zeros(WFLength)
        for wf in range(0,len(WaveformsThisEvt)):
            Shift=int(TimeShiftsThisEvt[wf])+Offset
            Kernel=[]

            if(DistancesThisEvt[wf]>MaxDistance):
                continue

            for dis in range(0,len(DistanceLims)-1):
                if(DistanceLims[dis]<DistancesThisEvt[wf] and DistanceLims[dis+1]>DistancesThisEvt[wf]) and len(WaveformsThisEvt[wf])==128:
                    Kernel=KernelDictionary[DistanceLims[dis]]
                    Filter=FilterDictionary[DistanceLims[dis]]
                    break
            if(len(Kernel)==0):
                continue
            if(Shift>WFLength):
                continue

            ToDeconvWF=numpy.zeros(WFLength)

            for w in range(0,min(128,(WFLength-Shift))):
                ToDeconvWF[w+Shift]+=numpy.array(WaveformsThisEvt[wf][w])
            ToDeconvFFT=numpy.fft.rfft(ToDeconvWF)
            FilteredFFT=ToDeconvFFT*Filter/Kernel
            TotalFilteredFFT+=FilteredFFT
            TotalSumWF+=ToDeconvWF
        TotalDeconvWF=numpy.fft.irfft(TotalFilteredFFT)
    
        fr["SumWF"]=dataclasses.I3VectorDouble(TotalSumWF)
        fr["DecWF"]=dataclasses.I3VectorDouble(TotalDeconvWF)
        fr["WeightDict"]=WeightDict
#        fr["I3MCWeightDict"]=WeightDict

        fout.push(fr)
    sigFile.close()
fout.close()
