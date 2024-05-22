import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

from tabulate import tabulate
#############################################################
def analize_voltages(headers, its, ittime, verrors,gues,prtvol):
    itsmean = np.mean(its, axis=1)
    ittimemean = np.mean(ittime, axis=1)
    verrorsmean = np.concatenate(([0], np.mean(verrors, axis=1)))
    meanitstimenum = ittimemean/itsmean
    meangues = np.mean(gues, axis=1)
    dailytime = np.sum(ittime,1)
    meanitaccelaration = itsmean/itsmean[0]
    meantimeaccelaration = ittimemean/ittimemean[0]
    
    headers = ['']+headers
    
    mtx = np.block([['Mean it number:',itsmean],
                    ['Daily time [ms]:',dailytime],
                    ['Mean total time [ms]:',ittimemean],
                    ['Mean it time [ms/its]:',meanitstimenum],
                    ['Mean it aceleration [its/its]:',meanitaccelaration],
                    ['Mean time aceleration [ms/ms]:',meantimeaccelaration],
                    ['Mean abs gues:',meangues],
                    ['Mean error [%]:',verrorsmean]])
    
    if prtvol:
        print(tabulate(mtx, headers=headers, tablefmt='grid'))
    
    return itsmean, ittimemean, verrorsmean, meangues
#############################################################
def print_example_quantities(t, ese, bars0,ve,file_name):
    plt.clf()
    N = ese.shape[0]
    fig, axes = plt.subplots(3, 2,figsize=(20 * 2 / 2.54, 10 * 2 / 2.54))
    for i in range(N):
        axes[0, 0].plot(t / 60, np.real(np.squeeze(ese[i,:])))
        axes[0, 1].plot(t / 60, np.abs(np.real(np.squeeze(ese[i,:])))/np.abs(np.squeeze(ese[i,:])))
        
        axes[1, 0].plot(t / 60, np.abs(ve[i,:]))
        axes[1, 1].plot(t / 60, np.angle(ve[i,:], deg=True))
        
    axes[2, 0].plot(t / 60, np.real(np.squeeze(bars0))/1000)
    axes[2, 1].plot(t / 60, np.abs(np.real(np.squeeze(bars0)))/np.abs(np.squeeze(bars0)))
    
    # custom_ticks = np.arange(0, 25, 1)
    # custom_labels = ['00:00','','','','','','','','','','','','12:00','','','','','','','','','','','','24:00']
    
    custom_ticks = np.arange(0, t[-1]/60+1, 1)
    hours = ['00:00','','','','','','','','','','','','12:00','','','','','','','','','','','']
    custom_labels = []
    for days in range(int(t[-1]/60/24)):
        custom_labels = custom_labels+hours
    custom_labels = custom_labels + ['00:00']
    
    for ax in axes.flat:
        ax.set_xlim(0, 24)
        ax.grid(True)
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels(custom_labels)
    
    axes[0, 0].set_title('Consumed Power [kW]')
    axes[0, 1].set_title('Consumed Power Factor')
    axes[1, 0].set_title('Voltage moduli [p.u.]')
    axes[1, 1].set_title('Voltage angles [°]')
    axes[2, 0].set_title('Power injected at root-node [MW]')
    axes[2, 1].set_title('Power Factor at root-node')
    
    plt.tight_layout()
    plt.show()
    fig.savefig(file_name+'.eps', format='eps')
    plt.close()
    return
#############################################################
def print_battery_quantities(t, Pbat, Qbat, SOC, file_name):
    plt.clf()
    N = Pbat.shape[0]
    SSS = Pbat+1j*Qbat
    fig, axes = plt.subplots(2, 2,figsize=(20 * 2 / 2.54, 10 * 2 / 2.54))
    for i in range(N):
        axes[0,0].plot(t / 60, np.squeeze(Pbat[i,:]))
        axes[0,1].plot(t / 60, np.abs(np.squeeze(Pbat[i,:]))/np.abs(np.squeeze(SSS[i,:])))
        axes[1,0].plot(t / 60, np.squeeze(SOC[i,:]))
        
    custom_ticks = np.arange(0, t[-1]/60+1, 1)
    hours = ['00:00','','','','','','','','','','','','12:00','','','','','','','','','','','']
    custom_labels = []
    for days in range(int(t[-1]/60/24)):
        custom_labels = custom_labels+hours
    custom_labels = custom_labels + ['00:00']
    
    
    for ax in axes.flat:
        ax.set_xlim(0, t[-1]/60)
        ax.grid(True)
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels(custom_labels)
    
    axes[0,0].set_title('Battery consumption [kW]')
    axes[0,1].set_title('Battery Power Factor')
    axes[1,0].set_title('SOC [p.u.]')
    
    plt.tight_layout()
    plt.show()
    fig.savefig(file_name+'.eps', format='eps')
    plt.close()
    return
#############################################################
def print_power(t, Ppv, Pload, ese, Qpv, Qload, bars0):
    plt.clf()
    N = Ppv.shape[0]
    fig, axes = plt.subplots(4, 3,figsize=(20 * 2 / 2.54, 10 * 2 / 2.54))
    for i in range(N):
        axes[0, 0].plot(t / 60, Ppv[i, :])
        axes[1, 0].plot(t / 60, Pload[i, :])
        axes[2, 0].plot(t / 60, np.real(ese[i, :]))
      
        axes[0, 1].plot(t / 60, Qpv[i, :])
        axes[1, 1].plot(t / 60, Qload[i, :])
        axes[2, 1].plot(t / 60, np.imag(ese[i, :]))
      
        ppp = Ppv[i, :]
        ppp[ppp == 0] = 0.001
        qqq = Qpv[i, :]
      
        axes[0, 2].plot(t / 60, np.divide(np.abs(ppp), np.sqrt(ppp**2 + qqq**2)))
        axes[1, 2].plot(t / 60, np.divide(Pload[i, :], np.sqrt(Pload[i, :]**2 + Qload[i, :]**2)))
        axes[2, 2].plot(t / 60,np.divide(np.abs(np.real(ese[i, :])), np.abs(ese[i, :])))
    
    axes[3, 0].plot(t / 60, np.real(np.squeeze(bars0)))
    axes[3, 1].plot(t / 60, np.imag(np.squeeze(bars0)))
    axes[3, 2].plot(t / 60, np.divide(np.real(np.squeeze(bars0)), np.abs(np.squeeze(bars0))))
    
    for ax in axes.flat:
        ax.set_xlim(0, 24)
        ax.grid(True)
    plt.tight_layout()
    plt.show()
    #fig.savefig('balance.png', format='png')
    #fig.savefig('balance.eps', format='eps')
    plt.close()
#############################################################
def print_voltages(t, ve, its, ittime, verrors,gues):
    N = ve.shape[0]
    Nalgs = its.shape[0]
    plt.clf()
    fig, axes = plt.subplots(3, 2,figsize=(20 * 2 / 2.54, 10 * 2 / 2.54))
    for i in range(N):
        axes[0, 0].plot(t / 60, np.abs(ve[i, :]))
        axes[0, 1].plot(t / 60, np.angle(ve[i, :]))
    
    for i in range(Nalgs):
        axes[1, 0].plot(t / 60, np.squeeze(its[i, :]))
        axes[1, 1].plot(t / 60, np.squeeze(ittime[i, :]))
        axes[2, 1].plot(t / 60, np.squeeze(gues[i, :]))
    for i in range(Nalgs - 1):
        axes[2, 0].plot(t / 60, np.squeeze(verrors[i, :]))
    for ax in axes.flat:
        ax.set_xlim(0, 24)
        ax.grid(True)
    plt.tight_layout()
    plt.show()
    #fig.savefig('voltages.png', format='png')
    #fig.savefig('voltages.eps', format='eps')
    plt.close()
    del fig, axes
    return
#############################################################
def reactive_power(P, fp):
    noise = np.array([random.gauss(1, 0.02) for _ in range(len(P))])
    Q = P * np.sqrt(np.reciprocal(fp**2) - 1) * noise
    return Q
#############################################################
def read_load_profile():
    filename = 'load_com_profile.csv'
    df = pd.read_csv(filename)
    com_model = df.values
    ################################
    filename = 'load_res_profile.csv'
    df = pd.read_csv(filename)
    res_model = df.values
    ################################
    filename = 'load_ind_profile.csv'
    df = pd.read_csv(filename)
    ind_model = df.values
    
    return com_model, res_model, ind_model
#############################################################
def read_pv_profile():
    filename = 'pv_profile.csv'
    df = pd.read_csv(filename)
    model = df.values
    
    return model
#############################################################
def load_interpole(load_rated, tt, load_mix, model_com, model_res,model_ind):
    hr = math.floor(tt / 60)
    if hr > 23:
        hr = hr - 24
        tt = tt - 24 * 60
    inst = tt / (24 * 60)
    times = np.array([[inst**3], [inst**2], [inst], [1]])
    ####################
    #com
    coefs = model_com[:, hr]
    com_profile = coefs @ times
    
    noise = np.array([random.gauss(1, 0.02) for _ in range(len(load_rated))])
    load_com = load_rated * load_mix[:, 0] * noise * com_profile
    load_com = np.clip(load_com, 0, None)  #eliminate negatives
    ####################
    #res
    coefs = model_res[:, hr]
    res_profile = coefs @ times
    
    noise = np.array([random.gauss(1, 0.02) for _ in range(len(load_rated))])
    load_res = load_rated * load_mix[:, 1] * noise * res_profile
    load_res = np.clip(load_res, 0, None)  #eliminate negatives
    ####################
    #ind
    coefs = model_ind[:, hr]
    ind_profile = coefs @ times
    
    noise = np.array([random.gauss(1, 0.02) for _ in range(len(load_rated))])
    load_ind = load_rated * load_mix[:, 2] * noise * ind_profile
    load_ind = np.clip(load_ind, 0, None)  #eliminate negatives
    
    return load_com, load_res, load_ind
#############################################################
def pv_interpole(pv_installed, tt, model):
    hr = math.floor(tt / 60)
    if hr > 23:
        hr = hr - 24
        tt = tt - 24 * 60
    
    coefs = model[:, hr]
    inst = tt / (24 * 60)
    times = np.array([[inst**3], [inst**2], [inst], [1]])
    pv_profile = coefs @ times
    
    noise = np.array([random.gauss(1, 0.02) for _ in range(len(pv_installed))])
    pv_generated = pv_installed * pv_profile * noise
    pv_generated = np.clip(pv_generated, 0, None)  #eliminate negatives
    
    return pv_generated
#############################################################
def bat_interpole(T,Pref, fp_ref,charge_cap, charge,charge_min,charge_max):
    
    N = len(charge)
    P = np.zeros((N,1))
    Q = np.zeros((N,1))
    
    for ii in range(N):
        c1 = charge[ii]<charge_max*charge_cap[ii]
        c2 = charge[ii]>charge_min*charge_cap[ii]
        c3 = Pref[ii]>0
        c  = (c1 and c2) or (not(c1) and not(c3)) or (not(c2) and c3)
        if c:
            P[ii] = Pref[ii]
        # P[ii] = Pref[ii]
        qaux = P[ii] * np.sqrt(np.reciprocal(fp_ref[ii]**2) - 1)
        Q[ii] = -abs(qaux)
        
        charge[ii] = charge[ii] + (T/60)*P[ii]

    return np.squeeze(P),np.squeeze(Q),charge