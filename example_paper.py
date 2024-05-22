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
t0 = -1.5*24*60  #begining of time in min
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
axes[0].set_title('Voltage amplitudes [p.u.]')
axes[1].set_title('Net power consumption at nodes [kW]')
axes[2].set_title('Power injected at root node [MW]')

P0olmax = np.ceil(max(P000_ol)*100)/100 + 0.01
P0olmin = np.floor(min(P000_ol)*100)/100 - 0.01
axes[2].set_ylim(P0olmin, P0olmax)

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
fig.tight_layout()

file_name = 'OpenLoop'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')

#######################################################
# CLOSED LOOP
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print("Primary Level Closed Loop")
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
########################################################
#penetration of batteries in circuit
# bat_penetration = [0.15, 0.25, 0.50, 0.75, 0.85, 1.00]#p.u.
bat_penetration = [0.5, 0.75, 1.00]
# bat_penetration = [1.0]
########################################################
#variables to draw
mean_V_1c = np.zeros((len(bat_penetration)+1, nn))
max_V_1c  = np.zeros((len(bat_penetration)+1, nn))
min_V_1c  = np.zeros((len(bat_penetration)+1, nn))

P000_1c   = np.zeros((len(bat_penetration)+1, nn))
pf000_1c  = np.zeros((len(bat_penetration)+1, nn))

legends_1c = []

#rescue from open loop
Vabs = np.abs(ve)
mean_V_1c[0,:]=np.mean(Vabs, axis=0)
max_V_1c[0,:] =np.max(Vabs, axis=0)
min_V_1c[0,:] =np.min(Vabs, axis=0)

P000_1c[0,:] = np.real(np.squeeze(S*bars0))/1000
pf000_1c[0,:] = np.abs(np.real(np.squeeze(bars0)))/np.abs(np.squeeze(bars0))

legends_1c.append('Open loop')

########################################################
#batts specs
bat_cap= 0.5*70/V #70kWh=avg elec car; 50Wh = laptop battery
batmin = 0.1 #min % charge
batmax = 0.9 #max % charge
Pbat_rate  = 10
Pbat_max   = 10
Pbat_scape = 10

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

fprate = 0.98
fpmax  = 0.99
fpmin  = 0.80
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
#loop through penetrations
bp_count=0
nodes_wout_BESS = list(range(0,N))
nodes_with_BESS = []
for bp in bat_penetration:
    ########################################################
    #select nodes with BESS
    n_nodes_withBESS = len(nodes_with_BESS)
    n_BESS = round(bp*N)
    for _ in range(n_BESS-n_nodes_withBESS):
        if nodes_wout_BESS:  # Check if array1 is not empty
            element = np.random.choice(nodes_wout_BESS)
            nodes_wout_BESS.remove(element)
            nodes_with_BESS.append(element)
    ########################################################
    installed_batteries = np.zeros((N,))
    installed_batteries[nodes_with_BESS] = 1
    bat = bat_cap*installed_batteries #installed batteries in Ahr; 

    Pminmax = Pbat_max*installed_batteries#
    Prate   = Pbat_rate*installed_batteries#
    Psafty  = Pbat_scape*installed_batteries#
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
    SOC[:,0] = np.squeeze(np.random.normal(0.5,0.1,N))#initial condition random
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
    #draw quantities
    bp_count = bp_count+1
    
    Vabs = np.abs(ve)
    mean_V_1c[bp_count,:]= np.mean(Vabs, axis=0)
    max_V_1c[bp_count,:] = np.max(Vabs, axis=0)
    min_V_1c[bp_count,:] = np.min(Vabs, axis=0)

    P000_1c[bp_count,:]  = np.real(np.squeeze(S*bars0))/1000
    pf000_1c[bp_count,:] = np.abs(np.real(np.squeeze(bars0)))/np.abs(np.squeeze(bars0))
    
    legends_1c.append('Prim. Ctrl.: '+str(bp*100)+'%')
    
    SOC_1c  = SOC*1
    Pbat_1c = Pbat*S
    
########################################################
#plot quantities
fig, axes = plt.subplots(1,3,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(mean_V_1c))
axes[1].plot(t / 60, np.transpose(P000_1c))
axes[2].plot(t / 60, np.transpose(pf000_1c))
axes[0].set_title('Mean voltage amplitudes [p.u.]')
axes[1].set_title('Power injected at root node [MW]')
axes[2].set_title('Power factor at root node')
axes[1].set_ylim(P0olmin, P0olmax)


for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
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
# SOC[:,0] = np.squeeze(np.random.normal(0.5,0.1,N))#initial condition random
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
# SOC[:,0] = np.squeeze(np.random.normal(0.5,0.1,N))#initial condition random
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
axes[0].set_title('Mean voltage amplitudes [p.u.]')

axes[1].plot(t / 60, np.transpose(P000_1c[0,:]))
axes[1].plot(t / 60, np.transpose(P000_1c[-1,:]))
axes[1].plot(t / 60, np.transpose(P000_2c1))
axes[1].plot(t / 60, np.transpose(P000_2c2))
axes[1].set_title('Power injected at root node [MW]')
axes[1].set_ylim(P0olmin, P0olmax)


axes[2].plot(t / 60, np.transpose(pf000_1c[0,:]))
axes[2].plot(t / 60, np.transpose(pf000_1c[-1,:]))
axes[2].plot(t / 60, np.transpose(pf000_2c1))
axes[2].plot(t / 60, np.transpose(pf000_2c2))
axes[2].set_title('Power factor at root node')

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
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

for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
fig.tight_layout()

file_name = 'SOC'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')


fig, axes = plt.subplots(1,3,figsize=(15 * 2 / 2.54, 5 * 2 / 2.54))
axes[0].plot(t / 60, np.transpose(Pbat_1c))
axes[1].plot(t / 60, np.transpose(Pbat_2c1))
axes[2].plot(t / 60, np.transpose(Pbat_2c2))
axes[0].set_title('Bat Pwr Primary Ctrl [kW]')
axes[1].set_title('Bat Pwr Fixed Sec. Ctrl [kW]')
axes[2].set_title('Bat Pwr Switch Sec. Ctrl [kW]')
for ax in axes.flat:
    ax.set_xlim(0, t[-1]/60)
    ax.grid(True)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels)
fig.tight_layout()

file_name = 'Pbat'
fig.savefig(file_name+'.eps', format='eps', bbox_inches='tight')
