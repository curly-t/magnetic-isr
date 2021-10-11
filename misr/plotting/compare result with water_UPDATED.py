from matplotlib import pyplot as plt



#rabis samo results.txt + wateresults.txt (results od vode) ki ga da tadruga scripta (analyze_trajectories.py)



#
import numpy as np
import pandas as pd
import re
df = pd.read_csv("results.txt", sep=' ')
df = df.sort_values('Frequency')


vzorciDict= # keyword dict linking directory names to the names of proteins
pike= # keyword dict linking direcotry names to marker types


def razlicneAmp(df, vzorec="NAME OF VZOREC"):
    current=df["Filename"][0].split("_")[4][4:]
    offset=df["Filename"][1].split("_")[3][6:]
    xi=[]
    yi=[]
    for point in range(len(df["Frequency"])):
        if current != df["Filename"][point].split("_")[4][4:]:
            plt.plot(xi,yi,".", label="amplituda = {}".format(current)) #labelje amplituda
            current=df["Filename"][point].split("_")[4][4:]
            xi=[]
            yi=[]
        else:
            xi.append(df["Frequency"][point])
            yi.append(df["Response"][point])
    plt.title("Vzorec {}, offset = {}".format(vzorec, offset))
    plt.plot(xi,yi,".", label="amplituda = {}".format(current))

def isteAmp(df):
    #stara
    current=df["Filename"][0].split("_")[2]
    offset=df["Filename"][1].split("_")[3][6:]
    xi=[]
    yi=[]
    for point in range(len(df["Frequency"])):
        if current != df["Filename"][point].split("_")[2]:
            plt.plot(xi,yi,".", label="Sample = {}".format(vzorciDict[current])) #label je imevzorca
            current=df["Filename"][point].split("_")[2]
            xi=[]
            yi=[]
        else:
            xi.append(df["Frequency"][point])
            yi.append(df["Response"][point])
    plt.title("Amplitude = {}, Offset = {}".format(df["Filename"][1].split("_")[4][4:], offset))
    plt.plot(xi,yi,".", label="Sample = {}".format(current))
    return offset, df["Filename"][1].split("_")[4][4:]

def isteAmp2(df):
    #za vec razlicnih vzorcev
    current=re.search(r"_1003_\w+_", df["Filename"][1])[0][6:-1]
    offset=re.search(r"offs\d+", df["Filename"][1])[0][4:]
    amplitude=re.search(r"ampl\d+", df["Filename"][1])[0][4:]
    grafData={}
    xi=[]
    yi=[]
    for point in range(len(df["Frequency"])):
        if current == re.search(r"_1003_\w+_", df["Filename"][point])[0][6:-1]:
            xi.append(df["Frequency"][point])
            yi.append(df["Response"][point])
        else:
            grafData[current]=(xi,yi)
            current=re.search(r"_1003_\w+_", df["Filename"][point])[0][6:-1]
            xi=[]
            yi=[]
    grafData[current]=(xi,yi)
    plt.title("Amplitude = {}, Offset = {}".format(amplitude, offset))
    for vzorc in grafData:
        x,y=grafData[vzorc]
        plt.plot(x,y,pike[vzorc], label="Sample = {}".format(vzorciDict[vzorc])) #labelje imevzorca
    print(grafData.keys())
    
    return offset, amplitude

def isteAmp3(df):
    '''za vec poskusov istga vzorca'''
    current=df["Filename"][0].split("_")[2]
    offset=re.search(r"offs\d+", df["Filename"][1])[0][4:]
    amplitude=re.search(r"ampl\d+", df["Filename"][1])[0][4:]
    poskusid=re.search(r"_\d+\\", df["Filename"][0])[0]
    grafData={}
    
    xi=[]
    yi=[]
    for point in range(len(df["Frequency"])):
        if poskusid ==  re.search(r"_\d+\\", df["Filename"][point])[0]:
            xi.append(df["Frequency"][point])
            yi.append(df["Response"][point])
        else:
            grafData[poskusid[1:-1]]=(xi,yi)
            poskusid=re.search(r"_\d+\\", df["Filename"][point])[0]
            xi=[]
            yi=[]
    grafData[poskusid[1:-1]]=(xi,yi)
    plt.title("Sample = {}, Amplitude = {}, Offset = {}".format(vzorciDict[current], amplitude, offset))
    
    for poskus in grafData:
        x,y=grafData[poskus]
        data = pd.DataFrame(data={'x': x, 'y': y})
        data=data.sort_values('x')
        plt.plot(data['x'], data['y'], '.--',label="exp number = {}".format(poskus), linewidth=0.5) #labelje imevzorca #pike[current],
    print(grafData.keys())
    return offset, amplitude


def plot_water():
    '''add points for water from wateresults.txt'''
    df = pd.read_csv("wateresults.txt", sep=' ')
    df = df.sort_values('Frequency')

    plt.plot(df["Frequency"],df["Response"], ".", c="cyan", label="Water")
    return

#----------------------------

#razlicneAmp(df)
offset, amplitude = isteAmp2(df)
offset, amplitude = isteAmp3(df)

#plot_water()


#----------------------------

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Frequency")
plt.ylabel("Response")
plt.legend()
plt.savefig(offset + "_" + amplitude + "_" + "chart.png")
plt.show()