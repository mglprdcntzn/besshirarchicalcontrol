import numpy as np
import matplotlib.pyplot as plt
import circuit_fun as ct
import loads_lib as lds
import time_fun as tm
import NR_fun as fx

from scipy.linalg import expm

########################################################
#Circuit generator
N = 30
Dmin = 150  #min distance btwn nodes
Dmax = 300  #max distance btwn nodes
V = 13.2  #in kV
nodes, lines = ct.create_circuit(N, Dmin, Dmax)

ST = 150 #[kVA] rate trafos
S  = ST*N 
########################################################
#loads at nodes
Shouse = 6#kVA
houses = lds.load_houses(nodes, Shouse, ST)
load   = houses*Shouse
#generation at nodes
pv = ct.DG_circuit(nodes, 0.25, ST/8, 0.01*ST)  #in kW
#impendances of the circuit
Y, Y0, Y00 = ct.impendances_circuit(lines, load)
Ybase = S / (V**2) / 1000  #divide by 1000 to obtain Ybase in Ohms
########################################################
#normalized circuit
barY = Y / Ybase
barY0 = Y0 / Ybase
barY00 = Y00 / Ybase

barLoad = load / S
barpv = pv / S
########################################################
ct.summary_circuit(nodes, lines,load,pv,'example_circuit')
########################################################
#define time
t0 = -0.5*24*60  #begining of time in min
tf = 1*24 * 60  #end of time
T = 0.1  #time step
nn = int((tf - t0) / T) + 1  #num of instants
t = np.linspace(t0, tf, nn)  #time vector in mins
########################################################
#SC algorithm params
itmax = 25
prec = 0.00001
########################################################
#load DG and load profiles
PVmodel = tm.read_pv_profile()
fpPV = 0.99
########################################################
#prefill vectors and matrices
ve = np.zeros((N, nn), dtype=complex)
ese = np.zeros((N, nn), dtype=complex)
bars0 = np.zeros((1, nn), dtype=complex)
Ppv = np.zeros((N, nn))
Pload = np.zeros((N, nn))
Qpv = np.zeros((N, nn))
Qload = np.zeros((N, nn))
fpPvVec = np.zeros((N, nn))
fpLoadVec = np.zeros((N, nn))
its = np.zeros((1, nn))
ittime = np.zeros((1, nn))
gues = np.zeros((1, nn))
########################################################
#initial conditions for iterations
R0 = np.eye(N)
Phi0 = np.zeros((N, N))
V0 = R0 @ expm(1j * Phi0)
########################################################
#perfiles de carga iniciales
perfiles_carga, perfiles_residuales = lds.load_houses_profiles(houses)
perfiles_residuales_old = perfiles_residuales
########################################################
# OPEN LOOP
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print("Open Loop")
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
#loop through time
for kk in range(nn):
    ##################################
    #instante
    tt = t[kk]
    if tt==0:
        kinit = kk
    #renovar perfiles de carga cada 24hrs
    if (tt)%(24*60)==0:
        perfiles_residuales_old = perfiles_residuales
        perfiles_carga, perfiles_residuales = lds.load_houses_profiles(houses)
    ##################################
    #PV generation
    pv_gen = tm.pv_interpole(barpv, tt%(24*60), PVmodel)
    
    Ppv[:, kk] = pv_gen
    Qpv[:, kk] = tm.reactive_power(pv_gen, fpPV)
    ##################################
    #Load
    Paux, Qaux                 = lds.load_interpole(tt%(24*60), perfiles_carga, perfiles_residuales_old)
    Pload[:, kk], Qload[:, kk] = Paux/S, Qaux/S
    ##################################
    #Power balance
    Pbalance = Pload[:, kk] - Ppv[:, kk]
    Qbalance = Qload[:, kk] - Qpv[:, kk]
    
    barS = Pbalance + 1j * Qbalance
    ese[:, kk] = barS
    barS = np.diag(barS)
    ##################################
    ve[:, kk], its[0, kk], ittime[0, kk], gues[0, kk] = fx.SC(barY, barY0, -barS, V0, itmax, prec)
    ##################################
    #power at root node
    VVV = np.diag(ve[:, kk])
    bars0[:, kk] = np.ones((1, N)) @ np.conj(barY0) @ (np.conj(VVV) - np.eye(N)) @ np.ones((N, 1))
    ##################################
    #initial conditions
    V0 = VVV
########################################################
#drawing constants
custom_ticks = np.arange(0, t[-1]/60+1, 1)
hours = ['00:00','','','','','','','','','','','','12:00','','','','','','','','','','','']
custom_labels = []
for days in range(int(t[-1]/60/24)):
    custom_labels = custom_labels+hours
custom_labels = custom_labels + ['24:00']
########################################################
#draw quantities
ve_ol     = np.abs(ve)
P000_ol   = np.real(np.squeeze(S*bars0))/1000
Pii       = np.real(np.squeeze(S*ese))

fig, axes = plt.subplots(1,3,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(ve_ol))
axes[1].plot(t / 60, np.transpose(Pii))
axes[2].plot(t / 60, np.transpose(P000_ol))
axes[0].set_title('Voltage amplitudes')
axes[1].set_title('Net power consumption at nodes')
axes[2].set_title('Power injected at root node')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')
axes[2].text(-0.1, 1.07, 'c)', transform=axes[2].transAxes, fontsize=14, va='top', ha='left')
axes[0].set_ylabel('$V$ (p.u.)')
axes[1].set_ylabel('$P_i$ (kW)')
axes[2].set_ylabel('$P_0$ (MW)')

P0olmax = np.ceil(max(P000_ol)*100)/100 + 0.01
P0olmin = np.floor(min(P000_ol)*100)/100 - 0.01
axes[2].set_ylim(P0olmin, P0olmax)

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Time (hrs)')
fig.tight_layout()

file_name = 'OpenLoop'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')

#######################################################
#rescue from open loop
Vabs_OL = np.abs(ve)
P0_OL   = np.real(np.squeeze(S*bars0))/1000
pf0_OL  = np.abs(np.real(np.squeeze(bars0)))/np.abs(np.squeeze(bars0))

#######################################################
# OPEN LOOP
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print("Battery following reference")
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
#batts specs
bat_cap= 0.5*70/V #70kWh=avg elec car; 50Wh = laptop battery
Pbat_rate  = 10
Pbat_max   = 8
Pbat_scape = 8

batmin = 0.1 #min % charge
batmax = 0.9 #max % charge

fprate = 0.98
fpmax  = 0.99
fpmin  = 0.80
########################################################
#select nodes with BESS
bp     = 0.4
n_BESS = round(bp*N)

nodes_with_BESS = np.random.choice(list(range(0,N)) , n_BESS,replace=False)

nodes_wout_BESS = list(range(0,N))
for iii in nodes_with_BESS:
    nodes_wout_BESS.remove(iii)

installed_batteries = np.zeros((N,))
installed_batteries[nodes_with_BESS] = 1

Pminmax = Pbat_max*installed_batteries#
Prate   = Pbat_rate*installed_batteries#
Psafty  = Pbat_scape*installed_batteries#

SOCinit = np.squeeze(np.random.normal(0.5,0.2,N))#initial condition random
#######################################################
bc  = 1.00
bat = bat_cap*bc*installed_batteries #installed batteries in Ahr; 
########################################################
#batteries perfect profile
Ppv_total = Ppv.sum(0)
Epv_total = Ppv_total.sum()*(tf-t0)/60
Ebat_total = V*bat.sum()

Ppv_mean  = 0.5*Ppv_total.sum()/nn

Pbat_SP   = S*(Ppv_total - Ppv_mean)/n_BESS

########################################################
#prefill vectors for primary ctrl simulation
Pbat_fil    = 1*installed_batteries

fpbat_SP     = np.ones((N, 1))

Pbat  = np.zeros((N, nn))
Qbat  = np.zeros((N, nn))
Ebat  = np.zeros((N, nn))
SOC   = np.zeros((N, nn))
SOC[:,0] = 1*SOCinit
SOC[np.isinf(SOC[:,0]),0] = 0
Eant  = SOC[:,0]*bat*V
    
Vav = 0.99*V*np.ones((N, 1))
########################################################
#loop through time with bat SP Ctrl 
for kk in range(nn):
    ##################################
    #instante
    tt = t[kk]
    ##################################
    #Batt
    if Pbat_SP[kk]>Pbat_max:
        Pbat_SP[kk] = Pbat_max
    elif Pbat_SP[kk]<-Pbat_max:
        Pbat_SP[kk] = -Pbat_max
    if Pbat_SP[kk]>=0:
        fpbat_SP = fpmax*np.ones((N, 1))
    else:
        fpbat_SP = fpmin*np.ones((N, 1))
    
    p, q, e = tm.bat_interpole(T,np.squeeze(Pbat_SP[kk]*np.ones((N,1))), np.squeeze(fpbat_SP),V*bat, Eant ,batmin,batmax)
    
    Pbat[:, kk] = p/S
    Qbat[:, kk] = q/S
    Ebat[:, kk] = e
    Eant        = e
    SOC[:, kk]  = e/bat/V
    SOC[np.isinf(SOC[:,kk]),kk] = 0
    SOC[np.isnan(SOC[:,kk]),kk] = 0
    ##################################
    #Power balance (in pu)
    Pbalance = Pload[:, kk] + Pbat[:,kk] - Ppv[:, kk]
    Qbalance = Qload[:, kk] + Qbat[:,kk] - Qpv[:, kk] 
    
    barS = Pbalance + 1j * Qbalance
    ese[:, kk] = barS
    barS = np.diag(barS)
    ##################################
    ve[:, kk], its[0, kk], ittime[0, kk], gues[0, kk] = fx.SC(barY, barY0, -barS, V0, itmax, prec)
    ##################################
    #power at root node
    VVV = np.diag(ve[:, kk])
    bars0[:, kk] = np.ones((1, N)) @ np.conj(barY0) @ (np.conj(VVV) - np.eye(N)) @ np.ones((N, 1))
    ##################################
    #initial conditions
    V0 = VVV
########################################################
#draw quantities
ve_batsp     = np.abs(ve)
P000_batsp   = np.real(np.squeeze(S*bars0))/1000
Pii_batsp    = np.real(np.squeeze(S*ese))
pf000_batsp  = np.abs(np.real(np.squeeze(bars0)))/np.abs(np.squeeze(bars0))


fig, axes = plt.subplots(1,3,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(ve_batsp))
axes[1].plot(t / 60, np.transpose(Pii_batsp))
axes[2].plot(t / 60, np.transpose(P000_batsp))
axes[0].set_title('Voltage amplitudes')
axes[1].set_title('Net power consumption at nodes')
axes[2].set_title('Power injected at root node')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')
axes[2].text(-0.1, 1.07, 'c)', transform=axes[2].transAxes, fontsize=14, va='top', ha='left')
axes[0].set_ylabel('$V$ (p.u.)')
axes[1].set_ylabel('$P_i$ (kW)')
axes[2].set_ylabel('$P_0$ (MW)')

P0batspmax = np.ceil(max(P000_batsp)*100)/100 + 0.01
P0batspmin = np.floor(min(P000_batsp)*100)/100 - 0.01
axes[2].set_ylim(P0batspmin, P0batspmax)

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Time (hrs)')
    ax.set_xlabel('Time (hrs)')
fig.tight_layout()

file_name = 'BatScheduled_1'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')


fig, axes = plt.subplots(1,3,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))

axes[0].plot(t / 60, np.mean(Vabs_OL, axis=0),label='Open loop')
axes[0].plot(t / 60, np.mean(ve_batsp, axis=0),label='Power scheduled')
axes[0].legend(loc='best')

axes[1].plot(t / 60, np.transpose(P0_OL),label='Open loop')
axes[1].plot(t / 60, np.transpose(P000_batsp),label='Power scheduled')
axes[1].legend(loc='best')

axes[2].plot(t / 60, np.transpose(Pbat*S))

axes[0].set_title('Mean voltage aplitudes')
axes[1].set_title('Power injected at root node')
axes[2].set_title('Bat Pwr Scheduled')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')
axes[2].text(-0.1, 1.07, 'c)', transform=axes[2].transAxes, fontsize=14, va='top', ha='left')
axes[0].set_ylabel('$V$ (p.u.)')
axes[1].set_ylabel('$P_0$ (MW)')
axes[2].set_ylabel('$P_{bat}$ (kW)')


for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Time (hrs)')
fig.tight_layout()

file_name = 'BatScheduled_2'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')

#######################################################
# CLOSED LOOP
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print("Primary Level Closed Loop")
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
#penetration of batteries in circuit
bat_capacities = [0.30, 0.60, 1.0]
########################################################
#variables to draw
mean_V_1c = np.zeros((len(bat_capacities)+1, nn))
max_V_1c  = np.zeros((len(bat_capacities)+1, nn))
min_V_1c  = np.zeros((len(bat_capacities)+1, nn))

P000_1c   = np.zeros((len(bat_capacities)+1, nn))
pf000_1c  = np.zeros((len(bat_capacities)+1, nn))

legends_1c = []

#rescue from open loop
mean_V_1c[0,:]=np.mean(Vabs_OL, axis=0)
max_V_1c[0,:] =np.max(Vabs_OL, axis=0)
min_V_1c[0,:] =np.min(Vabs_OL, axis=0)

P000_1c[0,:] = P0_OL
pf000_1c[0,:] = pf0_OL

legends_1c.append('Open loop')
########################################################
#Fuzzy constants
ep_a   = 0.01
ep_b   = 0.01
ep_c   = 0.0001
########################################################
#Primary Control periods
T1 = 1 #time for 1.Ctrl in Bat
n1 = np.floor(T1/T)
ndev = 2

ctrlsteps1 = np.random.randint(max(1,n1-ndev),n1+ndev,N)
delay1     = np.random.randint(1,n1,N)

T1real = ctrlsteps1*T
########################################################
#Control constants
Kp  = np.squeeze(np.random.normal(6,0.1,N))*1e2#Eant/bat/V

########################################################
#Average filters
tau_v = 3*60*np.squeeze(np.random.normal(1,0.1,N))#
Aavv   = np.exp(-T1/tau_v)
Bavv   = (1-Aavv)

########################################################
#Control filters
tau_fp  = T1*2*np.squeeze(np.random.normal(1,0.1,N))## ca. 10min
Afp     = np.exp(-T1/tau_fp)
Bfp     = (1-Afp)

tau_ff = T1*2*np.squeeze(np.random.normal(1,0.1,N))## ca. 10min
Aff    = np.exp(-T1/tau_ff)
Bff    = (1-Aff)

########################################################
#loop through capacities
bp_count=0
for bc in bat_capacities:
    ########################################################
    bat = bat_cap*bc*installed_batteries #installed batteries in Ahr; 
    ########################################################
    #prefill vectors for primary ctrl simulation
    Pbat_SP     = 0*installed_batteries#np.ones((N, 1))
    Pbat_fil    = 1*Pbat_SP
    Pbat_SPant  = 1*Pbat_SP
    Pbat_ctrl   = 1*Pbat_fil
    
    fpbat_SP     = np.ones((N, 1))
    fpbat_fil    = 1*fpbat_SP
    fpbat_filant = 1*fpbat_fil
    fpbat_ctrl   = 1*fpbat_fil
    
    Pbat  = np.zeros((N, nn))
    Qbat  = np.zeros((N, nn))
    Ebat  = np.zeros((N, nn))
    SOC   = np.zeros((N, nn))
    SOC[:,0] = 1*SOCinit
    SOC[np.isinf(SOC[:,0]),0] = 0
    Eant  = SOC[:,0]*bat*V
        
    Vav = 0.99*V*np.ones((N, 1))
    ########################################################
    #loop through time with Primary Ctrl 
    for kk in range(nn):
        ##################################
        #instante
        tt = t[kk]
        ##################################
        #Batt    
        p, q, e = tm.bat_interpole(T,np.squeeze(Pbat_SP), np.squeeze(fpbat_SP),V*bat, Eant ,batmin,batmax)
        
        Pbat[:, kk] = p/S
        Qbat[:, kk] = q/S
        Ebat[:, kk] = e
        Eant        = e
        SOC[:, kk]  = e/bat/V
        SOC[np.isinf(SOC[:,kk]),kk] = 0
        SOC[np.isnan(SOC[:,kk]),kk] = 0
        ##################################
        #Power balance (in pu)
        Pbalance = Pload[:, kk] + Pbat[:,kk] - Ppv[:, kk]
        Qbalance = Qload[:, kk] + Qbat[:,kk] - Qpv[:, kk] 
        
        barS = Pbalance + 1j * Qbalance
        ese[:, kk] = barS
        barS = np.diag(barS)
        ##################################
        ve[:, kk], its[0, kk], ittime[0, kk], gues[0, kk] = fx.SC(barY, barY0, -barS, V0, itmax, prec)
        ##################################
        #power at root node
        VVV = np.diag(ve[:, kk])
        bars0[:, kk] = np.ones((1, N)) @ np.conj(barY0) @ (np.conj(VVV) - np.eye(N)) @ np.ones((N, 1))
        ##################################
        #initial conditions
        V0 = VVV
        
        if kk==0:#initial conditions of average of voltages
            Vav = abs(ve[:, kk])
        ##################################
        #control algorithms
        for ii in nodes_with_BESS:         
            ###############################
            #control primario
            rr1 = (kk-delay1[ii])%ctrlsteps1[ii]
            if rr1 == 0:
                ##############################################
                #MediciOn y filtro de voltaje
                Vkk         = abs(ve[ii, kk])
                DeltaV      = Vav[ii] - Vkk
                Vav[ii]     = Aavv[ii]*Vav[ii] + Bavv[ii]*Vkk
                ##############################################
                if t[kk]>t0/2:
                    #MediciOn Pot
                    Pbatpu = Pbat[ii, kk]*S/Prate[ii]
                    ##############################################
                    #fuzzyficaciOn de baterIa
                    alfa        = 1/(1+np.exp(-(SOC[ii,kk]-batmax)/ep_a))
                    beta        = 1/(1+np.exp( (SOC[ii,kk]-batmin)/ep_b))
                    gama        = 1/(1+np.exp(  Pbatpu/ep_c)) 
                    ##############################################
                    #Variables de control
                    Pbat_ctrl[ii]    =  Pbatpu - Kp[ii]*T1real[ii]*DeltaV
                    ##############################################
                    #desfuzzificaciOn de variables de ctrl
                    Pbat_fil[ii]     = (beta-alfa)*Psafty[ii]/Prate[ii] + (1-alfa-beta)*Pbat_ctrl[ii]
                    
                    fpbat_fil[ii]    = gama*fpmin + (1-gama)*fpmax
                    ##############################################
                    #Filtros de setpoints
                    Paux             = Afp[ii]*Pbat_SPant[ii]  + Bfp[ii]*Pbat_fil[ii]
                    Pbat_SP[ii]      = min(max(Paux*Prate[ii],-Pminmax[ii]),Pminmax[ii])
                    Pbat_SPant[ii]   = Pbat_SP[ii]/Prate[ii]
                    
                    fpaux            = Aff[ii]*fpbat_filant[ii] + Bff[ii]*fpbat_fil[ii]
                    fpbat_SP[ii]     = min(max(fpaux,fpmin),fpmax)
                    fpbat_filant[ii] = fpbat_SP[ii]
    ########################################################
    #save quantities for drawing
    bp_count = bp_count+1
    
    Vabs = np.abs(ve)
    mean_V_1c[bp_count,:]= np.mean(Vabs, axis=0)
    max_V_1c[bp_count,:] = np.max(Vabs, axis=0)
    min_V_1c[bp_count,:] = np.min(Vabs, axis=0)

    P000_1c[bp_count,:]  = np.real(np.squeeze(S*bars0))/1000
    pf000_1c[bp_count,:] = np.abs(np.real(np.squeeze(bars0)))/np.abs(np.squeeze(bars0))
    
    legends_1c.append('Prim. Ctrl.: '+str(bc*100)+'%')
    
    SOC_1c  = SOC*1
    Pbat_1c = Pbat*S
    
########################################################
#plot quantities
fig, axes = plt.subplots(1,3,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(mean_V_1c))
axes[1].plot(t / 60, np.transpose(P000_1c))
axes[2].plot(t / 60, np.transpose(pf000_1c))
axes[0].set_title('Mean voltage amplitudes')
axes[1].set_title('Power injected at root node')
axes[2].set_title('Power factor at root node')
axes[1].set_ylim(P0olmin, P0olmax)

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')
axes[2].text(-0.1, 1.07, 'c)', transform=axes[2].transAxes, fontsize=14, va='top', ha='left')
axes[0].set_ylabel('$V$ (p.u.)')
axes[1].set_ylabel('$P_0$ (MW)')
axes[2].set_ylabel('$f_{p}$')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Time (hrs)')
fig.legend(legends_1c,loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=4)
fig.tight_layout()

file_name = 'PrimCtrl'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')

# # fig, axes = plt.subplots(1, 1)
# fig, axes = plt.subplots(1,1,figsize=(20 * 2 / 2.54, 10 * 2 / 2.54))
# axes.plot(t / 60, np.transpose(mean_V_1c))
# axes.set_xlim(0, t[-1]/60)
# axes.grid(True)
# axes.set_xticks(custom_ticks)
# axes.set_xticklabels(custom_labels)
# plt.tight_layout()
# plt.legend(legends_1c)

# # fig, axes = plt.subplots(2, 1)
# fig, axes = plt.subplots(2,1,figsize=(20 * 2 / 2.54, 10 * 2 / 2.54))
# axes[0].plot(t / 60, np.transpose(P000_1c))
# axes[1].plot(t / 60, np.transpose(pf000_1c))
# for ax in axes.flat:
#     ax.set_xlim(0, t[-1]/60)
#     ax.grid(True)
#     ax.set_xticks(custom_ticks)
#     ax.set_xticklabels(custom_labels)
# plt.tight_layout()


































#######################################################
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print("Secondary Level Closed Loop")
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
#the following code runs over the last penetration value
########################################################
#Secondary Control periods
T2 = 5 #time for 2.Ctrl in Bat
n2 = np.floor(T2/T)
ndev = 2

ctrlsteps2 = np.random.randint(max(1,n2-ndev),n2+ndev,N)
delay2     = np.random.randint(1,n2,N)

T2real = ctrlsteps2*T
########################################################
#neighbors
neighbours = []
for ii in range(N):
    neighboursofii = []
    for jj in range(N):
        if ii!=jj:
            d = np.sqrt((nodes[ii+1][0] - nodes[jj+1][0])**2 + (nodes[ii+1][1] - nodes[jj+1][1])**2)
            if d<1000:
                neighboursofii.append(jj)
    neighbours.append(neighboursofii)
########################################################
#Control constants
ele  = np.squeeze(np.random.normal(5,0.1,N))*1e-2#Eant/bat/V
########################################################
#prefill vectors for secondary ctrl simulation fixed consensus
Pbat_SP     = 0*installed_batteries#np.ones((N, 1))
Pbat_fil    = 1*Pbat_SP
Pbat_SPant  = 1*Pbat_SP
Pbat_ctrl   = 1*Pbat_fil

fpbat_SP     = np.ones((N, 1))
fpbat_fil    = 1*fpbat_SP
fpbat_filant = 1*fpbat_fil
fpbat_ctrl   = 1*fpbat_fil

Pbat  = np.zeros((N, nn))
Qbat  = np.zeros((N, nn))
Ebat  = np.zeros((N, nn))
SOC   = np.zeros((N, nn))
SOC[:,0] = 1*SOCinit
SOC[np.isinf(SOC[:,0]),0] = 0
Eant  = SOC[:,0]*bat*V
    
Vav = 0.99*V*np.ones((N, 1))

Pbatcom = np.ones((N, 1))
P2nd    = np.zeros((N, 1))
########################################################
#loop through time with Secondary ctrl fixed consensus
for kk in range(nn):
    ##################################
    #instante
    tt = t[kk]
    ##################################
    #Batt    
    p, q, e = tm.bat_interpole(T,np.squeeze(Pbat_SP), np.squeeze(fpbat_SP),V*bat, Eant ,batmin,batmax)
    
    Pbat[:, kk] = p/S
    Qbat[:, kk] = q/S
    Ebat[:, kk] = e
    Eant        = e
    SOC[:, kk]  = e/bat/V
    SOC[np.isinf(SOC[:,kk]),kk] = 0
    SOC[np.isnan(SOC[:,kk]),kk] = 0
    ##################################
    #Power balance (in pu)
    Pbalance = Pload[:, kk] + Pbat[:,kk] - Ppv[:, kk]
    Qbalance = Qload[:, kk] + Qbat[:,kk] - Qpv[:, kk] 
    
    barS = Pbalance + 1j * Qbalance
    ese[:, kk] = barS
    barS = np.diag(barS)
    ##################################
    ve[:, kk], its[0, kk], ittime[0, kk], gues[0, kk] = fx.SC(barY, barY0, -barS, V0, itmax, prec)
    ##################################
    #power at root node
    VVV = np.diag(ve[:, kk])
    bars0[:, kk] = np.ones((1, N)) @ np.conj(barY0) @ (np.conj(VVV) - np.eye(N)) @ np.ones((N, 1))
    ##################################
    #initial conditions
    V0 = VVV
    
    if kk==0:#initial conditions of average of voltages
        Vav = abs(ve[:, kk])
    ##################################
    #control algorithms
    for ii in nodes_with_BESS:         
        ###############################
        #control primario
        rr1 = (kk-delay1[ii])%ctrlsteps1[ii]
        if rr1 == 0:
            ##############################################
            #MediciOn y filtro de voltaje
            Vkk         = abs(ve[ii, kk])
            DeltaV      = Vav[ii] - Vkk
            Vav[ii]     = Aavv[ii]*Vav[ii] + Bavv[ii]*Vkk
            ##############################################
            if t[kk]>t0/2:
                #MediciOn Pot
                Pbatpu = Pbat[ii, kk]*S/Prate[ii]
                ##############################################
                #fuzzyficaciOn de baterIa
                alfa        = 1/(1+np.exp(-(SOC[ii,kk]-batmax)/ep_a))
                beta        = 1/(1+np.exp( (SOC[ii,kk]-batmin)/ep_b))
                gama        = 1/(1+np.exp(  Pbatpu/ep_c)) 
                ##############################################
                #Variables de control
                Pbat_ctrl[ii]    =  Pbatpu - Kp[ii]*T1real[ii]*DeltaV + P2nd[ii] #with secondary action!
                ##############################################
                #desfuzzificaciOn de variables de ctrl
                Pbat_fil[ii]     = (beta-alfa)*Psafty[ii]/Prate[ii] + (1-alfa-beta)*Pbat_ctrl[ii]
                
                fpbat_fil[ii]    = gama*fpmin + (1-gama)*fpmax
                ##############################################
                #Filtros de setpoints
                Paux             = Afp[ii]*Pbat_SPant[ii]  + Bfp[ii]*Pbat_fil[ii]
                Pbat_SP[ii]      = min(max(Paux*Prate[ii],-Pminmax[ii]),Pminmax[ii])
                Pbat_SPant[ii]   = Pbat_SP[ii]/Prate[ii]
                
                fpaux            = Aff[ii]*fpbat_filant[ii] + Bff[ii]*fpbat_fil[ii]
                fpbat_SP[ii]     = min(max(fpaux,fpmin),fpmax)
                fpbat_filant[ii] = fpbat_SP[ii]
        ###############################
        #control secundario
        rr2 = (kk-delay2[ii])%ctrlsteps2[ii]
        if rr2 == 0 and t[kk]>t0/2:
            ##############################################
            Pbatcom[ii] = Pbat[ii, kk]*S/Prate[ii]#SOC[ii, kk] #
            Paux        = 0
            
            for jj in neighbours[ii]:
                if jj in nodes_with_BESS:
                    Paux = Paux + Pbatcom[jj] - Pbatcom[ii]
            
            Nneigii  = len(neighbours[ii])
            P2nd[ii] = ele[ii]*Paux/Nneigii
########################################################
#rescue variables for plotting aftwards
Vabs       = np.abs(ve)
mean_V_2c1 = np.mean(Vabs, axis=0)

P000_2c1   = np.real(np.squeeze(S*bars0))/1000
pf000_2c1  = np.abs(np.real(np.squeeze(bars0)))/np.abs(np.squeeze(bars0))

SOC_2c1    = 1*SOC
Pbat_2c1   = Pbat*S
########################################################
#prefill vectors for secondary ctrl simulation random consensus
Pbat_SP     = 0*installed_batteries#np.ones((N, 1))
Pbat_fil    = 1*Pbat_SP
Pbat_SPant  = 1*Pbat_SP
Pbat_ctrl   = 1*Pbat_fil

fpbat_SP     = np.ones((N, 1))
fpbat_fil    = 1*fpbat_SP
fpbat_filant = 1*fpbat_fil
fpbat_ctrl   = 1*fpbat_fil

Pbat  = np.zeros((N, nn))
Qbat  = np.zeros((N, nn))
Ebat  = np.zeros((N, nn))
SOC   = np.zeros((N, nn))
SOC[:,0] = 1*SOCinit
SOC[np.isinf(SOC[:,0]),0] = 0
Eant  = SOC[:,0]*bat*V
    
Vav = 0.99*V*np.ones((N, 1))

Pbatcom = np.ones((N, 1))
P2nd    = np.zeros((N, 1))
########################################################
#loop through time with Secondary ctrl random consensus
for kk in range(nn):
    ##################################
    #instante
    tt = t[kk]
    ##################################
    #Batt    
    p, q, e = tm.bat_interpole(T,np.squeeze(Pbat_SP), np.squeeze(fpbat_SP),V*bat, Eant ,batmin,batmax)
    
    Pbat[:, kk] = p/S
    Qbat[:, kk] = q/S
    Ebat[:, kk] = e
    Eant        = e
    SOC[:, kk]  = e/bat/V
    SOC[np.isinf(SOC[:,kk]),kk] = 0
    SOC[np.isnan(SOC[:,kk]),kk] = 0
    ##################################
    #Power balance (in pu)
    Pbalance = Pload[:, kk] + Pbat[:,kk] - Ppv[:, kk]
    Qbalance = Qload[:, kk] + Qbat[:,kk] - Qpv[:, kk] 
    
    barS = Pbalance + 1j * Qbalance
    ese[:, kk] = barS
    barS = np.diag(barS)
    ##################################
    ve[:, kk], its[0, kk], ittime[0, kk], gues[0, kk] = fx.SC(barY, barY0, -barS, V0, itmax, prec)
    ##################################
    #power at root node
    VVV = np.diag(ve[:, kk])
    bars0[:, kk] = np.ones((1, N)) @ np.conj(barY0) @ (np.conj(VVV) - np.eye(N)) @ np.ones((N, 1))
    ##################################
    #initial conditions
    V0 = VVV
    
    if kk==0:#initial conditions of average of voltages
        Vav = abs(ve[:, kk])
    ##################################
    #control algorithms
    for ii in nodes_with_BESS:         
        ###############################
        #control primario
        rr1 = (kk-delay1[ii])%ctrlsteps1[ii]
        if rr1 == 0:
            ##############################################
            #MediciOn y filtro de voltaje
            Vkk         = abs(ve[ii, kk])
            DeltaV      = Vav[ii] - Vkk
            Vav[ii]     = Aavv[ii]*Vav[ii] + Bavv[ii]*Vkk
            ##############################################
            if t[kk]>t0/2:
                #MediciOn Pot
                Pbatpu = Pbat[ii, kk]*S/Prate[ii]
                ##############################################
                #fuzzyficaciOn de baterIa
                alfa        = 1/(1+np.exp(-(SOC[ii,kk]-batmax)/ep_a))
                beta        = 1/(1+np.exp( (SOC[ii,kk]-batmin)/ep_b))
                gama        = 1/(1+np.exp(  Pbatpu/ep_c)) 
                ##############################################
                #Variables de control
                Pbat_ctrl[ii]    =  Pbatpu - Kp[ii]*T1real[ii]*DeltaV + P2nd[ii] #with secondary action!
                ##############################################
                #desfuzzificaciOn de variables de ctrl
                Pbat_fil[ii]     = (beta-alfa)*Psafty[ii]/Prate[ii] + (1-alfa-beta)*Pbat_ctrl[ii]
                
                fpbat_fil[ii]    = gama*fpmin + (1-gama)*fpmax
                ##############################################
                #Filtros de setpoints
                Paux             = Afp[ii]*Pbat_SPant[ii]  + Bfp[ii]*Pbat_fil[ii]
                Pbat_SP[ii]      = min(max(Paux*Prate[ii],-Pminmax[ii]),Pminmax[ii])
                Pbat_SPant[ii]   = Pbat_SP[ii]/Prate[ii]
                
                fpaux            = Aff[ii]*fpbat_filant[ii] + Bff[ii]*fpbat_fil[ii]
                fpbat_SP[ii]     = min(max(fpaux,fpmin),fpmax)
                fpbat_filant[ii] = fpbat_SP[ii]
        ###############################
        #control secundario
        rr2 = (kk-delay2[ii])%ctrlsteps2[ii]
        if rr2 == 0 and t[kk]>t0/2:
            ##############################################
            Pbatcom[ii] = Pbat[ii, kk]*S/Prate[ii]#SOC[ii, kk] #
            permiso = True
            while permiso:
                jj = np.random.choice(neighbours[ii])
                if jj in nodes_with_BESS:
                    P2nd[ii] = T2real[ii]*ele[ii]*(Pbatcom[jj] - Pbatcom[ii])
                    permiso  = False
########################################################
#rescue variables for plotting aftwards
Vabs       = np.abs(ve)
mean_V_2c2 = np.mean(Vabs, axis=0)

P000_2c2   = np.real(np.squeeze(S*bars0))/1000
pf000_2c2  = np.abs(np.real(np.squeeze(bars0)))/np.abs(np.squeeze(bars0))

SOC_2c2    = 1*SOC
Pbat_2c2   = Pbat*S
########################################################
#plots
fig, axes = plt.subplots(1,3,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(mean_V_1c[0,:]))
axes[0].plot(t / 60, np.transpose(mean_V_1c[-1,:]))
axes[0].plot(t / 60, np.transpose(mean_V_2c1))
axes[0].plot(t / 60, np.transpose(mean_V_2c2))
axes[0].set_title('Mean voltage amplitudes')

axes[1].plot(t / 60, np.transpose(P000_1c[0,:]))
axes[1].plot(t / 60, np.transpose(P000_1c[-1,:]))
axes[1].plot(t / 60, np.transpose(P000_2c1))
axes[1].plot(t / 60, np.transpose(P000_2c2))
axes[1].set_title('Power injected at root node')
axes[1].set_ylim(P0olmin, P0olmax)


axes[2].plot(t / 60, np.transpose(pf000_1c[0,:]))
axes[2].plot(t / 60, np.transpose(pf000_1c[-1,:]))
axes[2].plot(t / 60, np.transpose(pf000_2c1))
axes[2].plot(t / 60, np.transpose(pf000_2c2))
axes[2].set_title('Power factor at root node')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')
axes[2].text(-0.1, 1.07, 'c)', transform=axes[2].transAxes, fontsize=14, va='top', ha='left')
axes[0].set_ylabel('$V$ (p.u.)')
axes[1].set_ylabel('$P_0$ (MW)')
axes[2].set_ylabel('$f_{p}$')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Time (hrs)')
fig.legend(['Open loop','Primary ctrl','Fixed consensus ctrl','Random consensus ctrl'],loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=4)
fig.tight_layout()

file_name = 'SecondCtrl'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')

# fig, axes = plt.subplots(1,1,figsize=(20 * 2 / 2.54, 10 * 2 / 2.54))
# axes.plot(t / 60, np.transpose(mean_V_1c[0,:]))
# axes.plot(t / 60, np.transpose(mean_V_1c[-1,:]))
# axes.plot(t / 60, np.transpose(mean_V_2c1))
# axes.plot(t / 60, np.transpose(mean_V_2c2))
# axes.set_title('Mean Voltage Amp [pu]')
# axes.set_xlim(0, t[-1]/60)
# axes.grid(True)
# axes.set_xticks(custom_ticks)
# axes.set_xticklabels(custom_labels)
# plt.tight_layout()
# plt.legend(['Open loop','Primary ctrl','Fixed consensus ctrl','Random consensus ctrl'])

# fig, axes = plt.subplots(1,1,figsize=(20 * 2 / 2.54, 10 * 2 / 2.54))
# axes.plot(t / 60, np.transpose(pf000_1c[0,:]))
# axes.plot(t / 60, np.transpose(pf000_1c[-1,:]))
# axes.plot(t / 60, np.transpose(pf000_2c1))
# axes.plot(t / 60, np.transpose(pf000_2c2))
# axes.set_title('Root Power Factor')
# axes.set_xlim(0, t[-1]/60)
# axes.grid(True)
# axes.set_xticks(custom_ticks)
# axes.set_xticklabels(custom_labels)
# plt.tight_layout()
# plt.legend(['Open loop','Primary ctrl','Fixed consensus ctrl','Random consensus ctrl'])

# fig, axes = plt.subplots(1,1,figsize=(20 * 2 / 2.54, 10 * 2 / 2.54))
# axes.plot(t / 60, np.transpose(P000_1c[0,:]))
# axes.plot(t / 60, np.transpose(P000_1c[-1,:]))
# axes.plot(t / 60, np.transpose(P000_2c1))
# axes.plot(t / 60, np.transpose(P000_2c2))
# axes.set_title('Root Power [kW]')
# axes.set_xlim(0, t[-1]/60)
# axes.grid(True)
# axes.set_xticks(custom_ticks)
# axes.set_xticklabels(custom_labels)
# plt.tight_layout()
# plt.legend(['Open loop','Primary ctrl','Fixed consensus ctrl','Random consensus ctrl'])


fig, axes = plt.subplots(1,3,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(SOC_1c))
axes[1].plot(t / 60, np.transpose(SOC_2c1))
axes[2].plot(t / 60, np.transpose(SOC_2c2))
axes[0].set_title('SoC Primary Ctrl')
axes[1].set_title('SoC Fixed Sec. Ctrl')
axes[2].set_title('SoC Switch Sec. Ctrl')

axes[0].text(-0.1, 1.09, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.09, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')
axes[2].text(-0.1, 1.09, 'c)', transform=axes[2].transAxes, fontsize=14, va='top', ha='left')
# axes[0].set_ylabel('SoC')
# axes[1].set_ylabel('SoC')
# axes[2].set_ylabel('SoC')

for ax in axes.flat:
    ax.set_ylim(0, 1)
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Time (hrs)')
fig.tight_layout()

file_name = 'SOC'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')


fig, axes = plt.subplots(1,3,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(Pbat_1c))
axes[1].plot(t / 60, np.transpose(Pbat_2c1))
axes[2].plot(t / 60, np.transpose(Pbat_2c2))
axes[0].set_title('Bat Pwr Primary Ctrl')
axes[1].set_title('Bat Pwr Fixed Sec. Ctrl')
axes[2].set_title('Bat Pwr Switch Sec. Ctrl')

axes[0].text(-0.1, 1.07, 'a)', transform=axes[0].transAxes, fontsize=14, va='top', ha='left')
axes[1].text(-0.1, 1.07, 'b)', transform=axes[1].transAxes, fontsize=14, va='top', ha='left')
axes[2].text(-0.1, 1.07, 'c)', transform=axes[2].transAxes, fontsize=14, va='top', ha='left')
axes[0].set_ylabel('$P_{bat}$ (kW)')
axes[1].set_ylabel('$P_{bat}$ (kW)')
axes[2].set_ylabel('$P_{bat}$ (kW)')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_xlabel('Time (hrs)')
fig.tight_layout()

file_name = 'Pbat'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')

###############################
std_mean_V_OL     = np.std(mean_V_1c[0,kinit:-1], ddof=1)
std_mean_V_pwrsch = np.std(np.mean(ve_batsp[:,kinit:-1], axis=0), ddof=1)
std_mean_V_1clast = np.std(mean_V_1c[-1,kinit:-1], ddof=1)
std_mean_V_2c1    = np.std(mean_V_2c1[kinit:-1], ddof=1)
std_mean_V_2c2    = np.std(mean_V_2c2[kinit:-1], ddof=1)

values1= [round(100*std_mean_V_pwrsch/std_mean_V_OL),
          round(100*std_mean_V_1clast/std_mean_V_OL),
          round(100*std_mean_V_2c1/std_mean_V_OL),
          round(100*std_mean_V_2c2/std_mean_V_OL),
          ]

print('First row of table:')
latex_table = "& $"+"\%$ & $".join(map(str, values1)) + "\%$\\\\\n"
print(latex_table)


std_mean_P0_OL     = np.std(P000_1c[0,kinit:-1], ddof=1)
std_mean_P0_pwrsch = np.std(P000_batsp[kinit:-1], ddof=1)
std_mean_P0_1clast = np.std(P000_1c[-1,kinit:-1], ddof=1)
std_mean_P0_2c1    = np.std(P000_2c1[kinit:-1], ddof=1)
std_mean_P0_2c2    = np.std(P000_2c2[kinit:-1], ddof=1)

values2= [round(100*std_mean_P0_pwrsch/std_mean_P0_OL),
          round(100*std_mean_P0_1clast/std_mean_P0_OL),
          round(100*std_mean_P0_2c1/std_mean_P0_OL),
          round(100*std_mean_P0_2c2/std_mean_P0_OL),
          ]

print('Second row of table:')
latex_table = "& $"+"\%$ & $".join(map(str, values2)) + "\%$\\\\\n"
print(latex_table)

mean_pf_ol  = np.mean(pf000_1c[0,kinit:-1])
mean_pf_ps  = np.mean(pf000_batsp[kinit:-1])
mean_pf_1c  = np.mean(pf000_1c[-1,kinit:-1])
mean_pf_2c1 = np.mean(pf000_2c1[kinit:-1])
mean_pf_2c2 = np.mean(pf000_2c2[kinit:-1])
    
values4= [round(mean_pf_ol,4),
          round(mean_pf_ps,4),
          round(mean_pf_1c,4),
          round(mean_pf_2c1,4),
          round(mean_pf_2c2,4),
          ]

print('Third row of table:')
latex_table = "& $"+"$ & $".join(map(str, values4)) + "$\\\\\n"
print(latex_table)

mean_std_Pbat1c  = np.mean(np.std(Pbat_1c[nodes_with_BESS,kinit:-1], axis=0, ddof=1))
mean_std_Pbat2c1 = np.mean(np.std(Pbat_2c1[nodes_with_BESS,kinit:-1], axis=0, ddof=1))
mean_std_Pbat2c2 = np.mean(np.std(Pbat_2c2[nodes_with_BESS,kinit:-1], axis=0, ddof=1))

values3= [round(mean_std_Pbat1c,3),
          round(mean_std_Pbat2c1,3),
          round(mean_std_Pbat2c2,3)
          ]

print('Fourth row of table:')
latex_table = "& $"+"$ & $".join(map(str, values3)) + "$\\\\\n"
print(latex_table)