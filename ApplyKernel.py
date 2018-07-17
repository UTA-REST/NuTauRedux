#!/bin/env python

from __future__ import division, print_function
import numpy, pylab, math
import icecube
from icecube import dataio
from icecube import dataclasses
from icecube import CascadeVariables
import os


gcd=dataio.I3File("/data/sim/IceCube/2011/filtered/level2/neutrino-generator/10099/00000-00999/GeoCalibDetectorStatus_IC86.55697_corrected_V2.i3.gz")

gcd.rewind()
fr=gcd.pop_frame()
geo=fr.Get("I3Geometry")


bgFilesI3=[]
bgDir="/data/user/dxu/batch/myjobs/EHE/level4a/nc_nutau/10099/outfile_reOP/update_07122016/"
bgFiles=os.listdir(bgDir)
for i in bgFiles:
    if(i.find(".i3.bz2")>0):
        bgFilesI3.append(i)

Count=0    
NoCascadesBG=0
Distances=[]
Waveforms=[]
Energies=[]
Times=[]
Charges=[]
MaxWFTimes=[]
BinWidths=[]

EnergyLimLow=5e3
EnergyLimHigh=1e6

for bgFile in bgFilesI3:
    Count=Count+1
    print("opening "+bgFile+",  "+str(Count)+" of " + str(len(bgFilesI3)))
    bgFile=dataio.I3File(bgDir+bgFile)
    bgFile.rewind()
    while(bgFile.more()):
        #pop a frame
        fr=bgFile.pop_physics()

        #grab the weight dict and event weight
        WeightDict=fr.Get("I3MCWeightDict")
        Weight=(365*24*3600)*1.0e-8*pow(WeightDict["PrimaryNeutrinoEnergy"], -2)*WeightDict["OneWeight"]/(WeightDict["NEvents"]*len(bgFilesI3))

        #Find the cascade center
        if not fr.Has("CascadeLlhVertexFit") :
            NoCascadesBG=NoCascadesBG+1
            continue

        if( WeightDict["PrimaryNeutrinoEnergy"]<EnergyLimLow) or (WeightDict["PrimaryNeutrinoEnergy"]>EnergyLimHigh):    
            continue
        csc=fr.Get('CascadeLlhVertexFit')
        cscpos=csc.pos

        #Get the waveforms
#        rd=fr.Get('In')
        wf=fr.Get('CalibratedWaveformsHLCATWD')
        DistancesThisEvt=[]
        WaveformsThisEvt=[]
        EnergiesThisEvt=[]
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
            Energies.append(csc.energy)
            Distances.append(DistancesThisEvt)
            Waveforms.append(WaveformsThisEvt)
            Charges.append(ChargesThisEvt)
            Times.append(TimesThisEvt)
            BinWidths.append(BinWidthsThisEvt)
            MaxWFTimes.append(TimesThisEvt[numpy.argmax(ChargesThisEvt)])

            

TimeShifts=[]
for tindex in range(0,len(Times)):
    StartTime=numpy.min(Times[tindex])
    TimeShifts.append((Times[tindex]-StartTime)/BinWidths[tindex])
    therange=numpy.arange(0,128)

W_Normd=[]
W_Sum=[]

ylimlo=-40
ylimhi=120


DistanceLims=numpy.arange(0,250,10)
AveWFs=[]
for dis in range(0,len(DistanceLims)-1):
    NWF=0
    TotCharge=0
    WFLength=500
    AveWF=numpy.zeros(WFLength)

    for i in range(0,len(Waveforms)):
        for j in range(0,len(Waveforms[i])):
            if(DistanceLims[dis]<Distances[i][j] and DistanceLims[dis+1]>Distances[i][j]) and len(Waveforms[i][j])==128:
                Shift=int(TimeShifts[i][j])
                if(Shift<WFLength):
                    Charge=0
                    for w in range(0,min(128,(WFLength-Shift))):
                        AveWF[w+Shift]+=numpy.array(Waveforms[i][j][w])
                        Charge+=Waveforms[i][j][w]
                    TotCharge+=Charge
                    NWF+=1
    AveWFs.append(AveWF/TotCharge)

    print(NWF,TotCharge)
    

WFDictionary={}
AveWFs[0]=AveWFs[1]
for i in range(0, len(AveWFs)):
    WFDictionary[DistanceLims[i]]=AveWFs[i]
    
import cPickle
KernelsFile=open("DOMKernelsLong.dat",'w')
cPickle.dump(WFDictionary,KernelsFile)
KernelsFile.close()
