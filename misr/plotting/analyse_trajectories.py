import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
from os import path
import glob
import warnings
import math as m
import pandas as pd
import time
from freq_and_phase_extract import freq_phase_ampl


def analyseFiles():
    freqDep = []
    filez = []
    # Find all measurements in current folder and subfolders
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(".dat")]:
            fname=path.join(dirpath, filename)
            # print(fname)
            filez.append(fname)


    for filepath in filez:
        #print(filepath)
        trackData = np.loadtxt(filepath)

        # For now, we get the approximate frequency, I offset and I amplitude from file name directly.
        approxFreq = int((re.search(r"freq\d+", filepath)[0])[4:])/1000
        approxIoffs = int((re.search(r"offs\d+", filepath)[0])[4:])/1000
        approxIampl = int((re.search(r"ampl\d+", filepath)[0])[4:])/1000
        print("Filename data:   Freq:", approxFreq, "  I ampl:", approxIampl, "  I offs:", approxIoffs)

        system_response = freq_phase_ampl(trackData, approxFreq, freq_err=1, plot_track_results=False)


        # print("Measured resp:   Freq:{0:5.3f} ({1:5.3f})  Phase:{2:5.3f} ({3:5.3f})   Ampl:{4:4.3f} ({5:5.3f})".
        #       format(system_response['freq'], system_response['freq_err'],
        #              system_response['phase'], system_response['phase_err'],
        #              system_response['ampl'], system_response['ampl_err']), end="\n\n")

        # # Samo hiter output za Mentorja 26. 11. 2020
        #print(filepath[33:], approxIampl, system_response['freq'], system_response['ampl'])
        print(filepath, approxIampl, system_response['freq'], system_response['ampl'])
        #print("created: %s" % time.ctime(path.getmtime(filepath)))
        #print(path.getmtime(filepath))
        freqDep.append([filepath, path.getmtime(filepath)-zeroTime, approxFreq,system_response['ampl']])


    df = pd.DataFrame(freqDep, columns=['Filename','Time','Frequency','Response']) 
    
    print(df.sort_values('Frequency'))

    #df.to_csv(r'results.dat', header=None, index=None, sep=' ', mode='a')
    df.to_csv(r'results.txt', index=None, sep=' ', mode='w')
    return df


def showTimeEvolution(df):

    df=df.sort_values('Time')
    fig, ax = plt.subplots()
    ax.set_yscale('log')
 
  
    df=df.loc[df['Frequency'] <2 ]
    df['Time']=df['Time'] / 60.0 / 60.0
    df=df.loc[df['Time']  > 60 ]
    df=df.loc[ df['Filename'].str.contains("A")]

    for key, grp in df.groupby(['Frequency']):
        print(key)
        ax = grp.plot(ax=ax, style='.-', x='Time', y='Response', label=key)

    #plt.legend(loc='lower left',title='Frequency [Hz]')
    plt.legend(loc='lower right',title='Frequency [Hz]')
    ax.set_xlabel("Time [hrs]")
    ax.set_ylabel("Response [px]")
    plt.savefig('chart.png')
    plt.show()

def compareTimeEvolution(df):
    df = df.sort_values('Time')
    fig, ax = plt.subplots()
    #ax.set_yscale('log')
 
  
    df=df.loc[df['Frequency'] == 0.2 ]
    df['Time']=df['Time'] / 60.0 / 60.0
   # df=df.loc[df['Time']  < 20 ]
    
    df['Experiment'] = # keywordname
    df['Experiment'].loc[ df['Filename'].str.contains("B")]= # another one keyword

    for key, grp in df.groupby(['Experiment']):
        print(key)
        ax = grp.plot(ax=ax, style='.-', x='Time', y='Response', label=key)

    #plt.legend(loc='lower left',title='Frequency [Hz]')
    plt.legend(loc='lower left',title='Experiment')
    ax.set_xlabel("Time [hrs]")
    ax.set_ylabel("Response [px]")
    plt.title('Time evolution of a response at 0.2 Hz')
    plt.savefig('chart.png')
    plt.show()



def showFreqResponse(df):
    df = df.sort_values('Frequency')
    #fig, ax = plt.subplots()
    plt.plot(df["Frequency"], df["Response"],".")
    plt.title("st pik = {}".format(len(df["Frequency"])))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Frequency")
    plt.ylabel("Response")
    #ax.set_yscale('log')

    plt.savefig('chart.png')
    plt.show()


if __name__ == "__main__":

    if os.path.isfile(resultFile):
        df = pd.read_csv(resultFile, sep=' ')
        print(df.head())
    else:
        df = analyseFiles()
    

    #showTimeEvolution(df)
    #compareTimeEvolution(df)
    showFreqResponse(df)