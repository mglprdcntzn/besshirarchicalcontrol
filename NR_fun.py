import numpy as np
import math
import time
from scipy.linalg import expm
import networkx as nx
from scipy.sparse import diags
#############################################################
def matrices_circuit(barY, barY0):
    N = barY.shape[0]
    unos = np.ones((N, 1))
    barY00 = np.squeeze(np.transpose(unos)@barY0@unos)
    
    L = np.block([[barY00, np.transpose(unos)@barY0], [barY0@unos, barY]])
    G = nx.from_numpy_matrix(L)
    
    for ii in range(N+1):
        G.remove_edge(ii,ii)
    
    D = np.transpose(nx.incidence_matrix(G, oriented=True).toarray())
    
    absD = abs(D)
        
    edges = nx.get_edge_attributes(G, 'weight')
    hatW = []
    for edge in edges:
        hatW.append(-edges[edge])
    hatW = diags(hatW,0).toarray()
    hatG = np.imag(hatW)
    
    A = np.transpose(D)
    _, _, V = np.linalg.svd(A)    
    null_space_indices = np.where(np.isclose(np.linalg.svd(A)[1], 0))
    
    
    if len(null_space_indices[0]) == 0:
        Dbot = []
    else:
        Dbot = V.T[:, null_space_indices].squeeze()   
    
    return D,absD,Dbot,hatW,hatG
########################################################
def FPPF(barY, barY0, barS, V0, itmax, prec):
    D,absD,Dbot,hatW,hatG = matrices_circuit(barY, barY0)
    
    hatGinv = np.linalg.inv(hatG)
    Dpse    = D@np.linalg.pinv(D.T@D)
    Dpse2   = np.linalg.pinv(D.T@D)@D.T
    
    N = V0.shape[0]
    unos = np.ones((N, 1))
    absE = D.shape[0]
    unosE = np.ones((absE, 1))
    
    barq = np.imag(barS)@unos
    hatp = np.real( np.block([[-unos.T@barS@unos], [barS@unos]])  )
    
    barB = np.imag(barY)
    barBinv = np.linalg.inv(barB)
    barB0 = np.imag(barY0)
    
    absDLT = np.block([0*unos, np.eye(N)]) @absD.T
    
    ccc    = len(Dbot)>0
    if ccc:
        if len(Dbot)==Dbot.shape[0]:
            Dbot = np.reshape(Dbot, (len(Dbot), 1))
        yslack = np.ones((Dbot.shape[1],1))
    
    E = []
    for ee in range(D.shape[0]):
        ii = -1
        jj = -1
        node = 0
        while ii==-1 or jj==-1:
            if D[ee][node]>0:
                ii = node
            elif D[ee][node]<0:
                jj = node
            node = node + 1
        E.append((ii,jj))

    ##
    start_time = time.time()
    #loop
    R = np.abs(V0)
    rrr = R@unos
    Theta = np.zeros((absE,1))
    V = V0
    ve = V0@unos
        
    hhh = np.zeros((absE,1))
        
    permiso = True
    itcont = 0
    while permiso:
        g = gue(V, barY, barY0, barS)
        
        for ee in range(D.shape[0]):
            iee = E[ee][0]
            jee = E[ee][1]
            if iee == 0:
                Vik = 1
            else:
                Vik = rrr[iee-1]
                
            if jee == 0:
                Vjk = 1
            else:
                Vjk = rrr[jee-1]
                
            hhh[ee] = Vik*Vjk
        
        if ccc:
            we = -np.diag(np.squeeze(1/hhh))@hatGinv@(Dpse@hatp + Dbot@yslack)
            JJJk = Dbot.T@np.diag(np.squeeze( (unosE - we**2)**(-1/2) ))@np.diag(np.squeeze(1/hhh))@hatGinv@Dbot
            Theta = np.array([math.asin(x) for x in we]).T
            Theta = np.reshape(Theta, (len(Theta), 1))
            yslack = yslack + np.linalg.inv(JJJk)@Dbot.T@Theta
        else:    
            we = -np.diag(np.squeeze(1/hhh))@hatGinv@Dpse@hatp
            Theta = np.array([math.asin(x) for x in we]).T
            Theta = np.reshape(Theta, (len(Theta), 1))
        
        meanphase = -np.block([1, 0*unos.T])@Dpse2@Theta
        phi       = np.block([0*unos, np.eye(N)])@Dpse2@Theta + unos*meanphase        
        
        uu = unosE - (unosE - we**2)**(1/2)
        
        RRRinv = np.diag(np.squeeze(1/rrr))
        HHH = np.diag(np.squeeze(hhh))
        
        rrr = barBinv@RRRinv@( barq - absDLT@hatG@HHH@uu ) - barBinv@barB0@unos 
        
        # Vant = V
        V = np.diag(np.squeeze(rrr))@np.diag(np.exp(1j*np.squeeze(phi)))
        
        itcont = itcont + 1
        # cond3  = np.linalg.norm(Vant@unos - V@unos) < (0.00000001*np.linalg.norm(Vant@unos))
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:# or cond3:
            permiso = 0
            break
    
    end_time = time.time()
    ve  = V@unos
    return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def SC(barY, barY0, barS, V0, itmax, prec):
    N = V0.shape[0]
    unos = np.ones((N, 1))
    invY = np.linalg.inv(barY)
    Y0unos = barY0@unos
    conbarS = np.conj(barS)
    ##
    start_time = time.time()
    #loop
    V = V0
    ve = V@unos
    permiso = True
    itcont = 0
    while permiso:
        g = gue(V, barY, barY0, barS)
      
        ve  = V@unos
        conVinv = np.conj(1/ve)
      
        ve = invY@(conbarS@conVinv -  Y0unos)
      
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
            permiso = 0
            break
        V  = np.diag(np.squeeze(ve))
    end_time = time.time()
    return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def FPX(barY, barY0, barS, V0, itmax, prec):
    N = V0.shape[0]
    unos = np.ones((N, 1))
    conY = np.conj(barY)
    conY0= np.conj(barY0)
    invY = np.linalg.inv(barY)
    Sconj = np.conj(barS)
    ##
    start_time = time.time()
    #loop
    V = V0
    R = abs(V0)
    ePhi = np.linalg.inv(R)@V0
    XXX  = np.linalg.inv(barS - R@conY@R)
    permiso = True
    itcont = 0
    while permiso:
        g = gue(V, barY, barY0, barS)
    
        XXX  = np.linalg.inv(barS - R@conY@R)
        xxx  = np.squeeze(XXX@R@conY0@unos)
        ePhi = np.diag(np.conj(xxx))
        
        rrr  = np.diag(xxx)@invY@(np.linalg.inv(R)@ePhi@Sconj - barY0)@unos
        R    = np.diag(np.squeeze(rrr))
                  
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
            permiso = 0
            break
        
        V  = R@ePhi
    end_time = time.time()
    ve = V@unos
    return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)

########################################################
def NRI(barY, barY0, barS, V0, itmax, prec):
    N = V0.shape[0]
    Id = np.eye(N)
    unos = np.ones((N, 1))
    conY = np.conj(barY)
    invconY = np.linalg.inv(conY)
    ##
    start_time = time.time()
    #loop
    V = V0
    ve = V@unos
    permiso = True
    itcont = 0
    while permiso:
        g = gue(V, barY, barY0, barS)
        
        ve  = V@unos
        Vinv = np.diag(np.squeeze(1/ve))
        h = Vinv@g
          
        AAA = barS@np.square(Vinv)
        FFF = invconY@AAA
        HHH = np.linalg.inv(Id -FFF@np.conj(FFF))@invconY
        Hh  = HHH@h
              
        ve = ve + np.conj(FFF)@Hh - np.conj(Hh)
          
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
            permiso = 0
            break
        
        V  = np.diag(np.squeeze(ve))
    end_time = time.time()
    return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRI0(barY, barY0, barS, V0, itmax, prec):
    N = V0.shape[0]
    Id = np.eye(N)
    unos = np.ones((N, 1))
    conY = np.conj(barY)
    invconY = np.linalg.inv(conY)
    
    Vinv = np.linalg.inv(V0)
    AAA = barS@np.square(Vinv)
    FFF = invconY@AAA
    XXX = np.linalg.inv(Id -FFF@np.conj(FFF))
    HHH = XXX@invconY
    ##
    start_time = time.time()
    #loop
    V = V0
    ve = V@unos
    permiso = True
    itcont = 0
    while permiso:
        g = gue(V, barY, barY0, barS)
        
        ve  = V@unos
        Vinv = np.diag(np.squeeze(1/ve))
        h = Vinv@g
          
        Hh  = HHH@h
              
        ve = ve + np.conj(FFF)@Hh - np.conj(Hh)
          
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
            permiso = 0
            break
        
        V  = np.diag(np.squeeze(ve))
    end_time = time.time()
    return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRIsum(barY, barY0, barS, V0, itmax, prec,M):
    N = V0.shape[0]
    Id = np.eye(N)
    unos = np.ones((N, 1))
    conY = np.conj(barY)
    invconY = np.linalg.inv(conY)
    ##
    start_time = time.time()
    #loop
    V = V0
    ve = V@unos
    permiso = True
    itcont = 0
    while permiso:
        g = gue(V, barY, barY0, barS)
        
        ve  = V@unos
        Vinv = np.diag(np.squeeze(1/ve))
        h = Vinv@g
          
        AAA = barS@np.square(Vinv)
        FFF = invconY@AAA
        PPP = FFF@np.conj(FFF)
        
        XXX = Id
        for m in range(M):
            XXX = XXX + np.linalg.matrix_power(PPP, m+1)
        
        #XXX = Id + PPP
        
        HHH = XXX@invconY
        Hh  = HHH@h
              
        ve = ve + np.conj(FFF)@Hh - np.conj(Hh)
          
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
            permiso = 0
            break
        
        V  = np.diag(np.squeeze(ve))
    end_time = time.time()
    return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRIM(barY, barY0, barS, V0, itmax, prec):
    N = V0.shape[0]
    Id = np.eye(N)
    unos = np.ones((N, 1))
    conY = np.conj(barY)
    invconY = np.linalg.inv(conY)
    
    Vinv = np.linalg.inv(V0)
    AAA = barS@np.square(Vinv)
    FFF = invconY@AAA
    XXX = np.linalg.inv(Id -FFF@np.conj(FFF))
    ##
    start_time = time.time()
    #loop
    V = V0
    ve = V@unos
    permiso = True
    itcont = 0
    while permiso:
        g = gue(V, barY, barY0, barS)
        
        ve  = V@unos
        Vinv = np.diag(np.squeeze(1/ve))
        h = Vinv@g
                
        FFF = invconY@barS@np.square(Vinv)
        Hh  = XXX@invconY@h
              
        ve = ve + np.conj(FFF)@Hh - np.conj(Hh)
                
        XXX = Id + FFF@np.conj(XXX)@np.conj(FFF)
        
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
            permiso = 0
            break
        
        V  = np.diag(np.squeeze(ve))
    end_time = time.time()
    return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def IWA(barY, barY0, barS, V0, itmax, prec):
    N = V0.shape[0]
    Id = np.eye(N)
    unos = np.ones((N, 1))
    conY0 = np.conj(barY0)
    conY = np.conj(barY)
    invconY = np.linalg.inv(conY)
      
    Vinv = np.linalg.inv(V0)
    A0 = conY0 + np.diag(np.squeeze(conY @ np.conj(V0) @ unos))
    F0 = invconY@Vinv@A0
    H0 = np.linalg.inv(Id -F0@np.conj(F0))@invconY@Vinv
    g0 = gue(V0, barY, barY0, barS)
    F0con = np.conj(F0)
      
    ##
    start_time = time.time()
    #loop
    Deltave = 0*unos
    V= V0
    ve0 = V0@unos
        
    permiso = True
    itcont = 0
    while permiso:
        g = gue(V, barY, barY0, barS)
          
        DeltaV = np.diag(np.squeeze(Deltave))
        Deltag = g0 + DeltaV@conY@np.conj(DeltaV)@unos
        HDve   = H0@Deltag
          
        Deltave = F0con@HDve - np.conj(HDve)
          
        ve = ve0 + Deltave
          
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
            permiso = 0
            break
        
        V  = np.diag(np.squeeze(ve))
    end_time = time.time()
    return np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRX(barY, barY0, barS, V0, itmax, prec):
    N = V0.shape[0]
    Id = np.eye(N)
    unos = np.ones((N, 1))
    conY0 = np.conj(barY0)
    conY = np.conj(barY)
    invconY = np.linalg.inv(conY)
    ##
    start_time = time.time()
    #loop
    V = V0
    ve = V@unos
    permiso = True
    itcont = 0
    while permiso:
        g = gue(V, barY, barY0, barS)
        
        vvv  = V@unos
        Vinv = np.diag(np.squeeze(1/vvv))
         
        AAA = conY0 + np.diag(np.squeeze(conY @ np.conj(V) @ unos))
        FFF = invconY@Vinv@AAA
        HHH = np.linalg.inv(Id -FFF@np.conj(FFF))@invconY@Vinv
        Hg  = HHH@g
              
        ve = V@unos
        ve = ve + np.conj(FFF)@Hg - np.conj(Hg)
         
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
            permiso = 0
            break
        
        V  = np.diag(np.squeeze(ve))
    end_time = time.time()
    return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRX0(barY, barY0, barS, V0, itmax, prec):
    N = V0.shape[0]
    Id = np.eye(N)
    unos = np.ones((N, 1))
    conY0 = np.conj(barY0)
    conY = np.conj(barY)
    invconY = np.linalg.inv(conY)
    
    Vinv = np.linalg.inv(V0)
    AAA = conY0 + np.diag(np.squeeze(conY @ np.conj(V0) @ unos))
    FFF = invconY@Vinv@AAA
    XXX = np.linalg.inv(Id -FFF@np.conj(FFF))
    ##
    start_time = time.time()
    #loop
    V = V0
    ve = V@unos
    permiso = True
    itcont = 0
    while permiso:
        g = gue(V, barY, barY0, barS)
        
        vvv  = V@unos
        Vinv = np.diag(np.squeeze(1/vvv))
         
        HHH = XXX@invconY@Vinv
        Hg  = HHH@g
              
        ve = V@unos
        ve = ve + np.conj(FFF)@Hg - np.conj(Hg)
         
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
            permiso = 0
            break
        
        V  = np.diag(np.squeeze(ve))
    end_time = time.time()
    return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def GS(barY, barY0, barS, V0, itmax, prec):
    N = V0.shape[0]
    unos = np.ones((N, 1))
    ##
    start_time = time.time()
    #Matrix Linv
    L = np.tril(barY)
    U = barY - L
    invL = np.linalg.inv(L)
    #other matrices 
    barY0vec = barY0@unos;
    Svec     = barS@unos;
    cSvec    = np.conj(Svec);
      
    #loop
    V = V0
    ve = V@unos
    permiso = True
    itcont = 0
    while permiso:
        g = gue(V, barY, barY0, barS)
        
        ve = V@unos
        ve = invL@(np.diag(np.squeeze(np.reciprocal(np.conj(ve))))@cSvec  - barY0vec - U@ve)
        
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
            permiso = 0
            break
      
        V  = np.diag(np.squeeze(ve))
    end_time = time.time()
    return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)

########################################################
def NRR(barY, barY0, barS, V0, itmax, prec):
    N = V0.shape[0]
    Id = np.eye(N)
    unos = np.ones((N, 1))
    conY0 = np.conj(barY0)
    conY = np.conj(barY)
    invconY = np.linalg.inv(conY)
    ##
    start_time = time.time()
    #loop
    V = V0
    U = np.real(V)
    W = np.imag(V)
    uw = np.block([[U @ unos], [W @ unos]])
    permiso = True
    itcont = 0
    while permiso:
        g = gue(V, barY, barY0, barS)
        
        vvv  = V@unos
        Vinv = np.diag(np.squeeze(1/vvv))
         
        AAA = conY0 + np.diag(np.squeeze(conY @ np.conj(V) @ unos))
        FFF = invconY@Vinv@AAA
        HHH = np.linalg.inv(Id -FFF@np.conj(FFF))@invconY@Vinv
        GGG = -np.conj(FFF)@HHH
        
        Mtx1 = np.block([[Id, Id], [-1j * Id, 1j * Id]])
        Mtx2 = np.block([[GGG, np.conj(HHH)], [HHH, np.conj(GGG)]])
        Mtx3 = np.block([[Id, 1j*Id], [Id, -1j * Id]])
        
        Mtx = np.real(0.5 * Mtx1 @ Mtx2 @ Mtx3)
                      
        Delta = Mtx @ np.block([[np.real(g)], [np.imag(g)]])
        
        uw = uw - Delta
        
        U = np.diag(np.squeeze(uw[:N]))
        W = np.diag(np.squeeze(uw[N:]))
        V = U + 1j*W
         
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
            permiso = 0
            break
    end_time = time.time()
    ve = V@unos
    return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRC(barY, barY0, barS, R0, Phi0, itmax, prec):
    N = R0.shape[0]
    Id = np.eye(N)
    unos = np.ones((N, 1))
    conY0 = np.conj(barY0)
    conY = np.conj(barY)
      
    permiso = True
    itcont = 0
      
    R = R0
    Phi = Phi0 
    rp = np.block([[R @ unos], [Phi @ unos]])
      
    start_time = time.time()
    while permiso:
        EEE = expm(1j * Phi)
        V = R @ EEE
        g = gue(V, barY, barY0, barS)
      
        AAA = conY0 + np.diag(np.squeeze(conY @ np.conj(V) @ unos))
        BBB = V @ conY
        
        Mtx1 = np.block([[Id, Id], [-1j * Id, 1j * Id]])
        Mtx2 = np.block([[AAA, BBB], [np.conj(BBB), np.conj(AAA)]])
        Mtx3 = np.block([[EEE, 1j * V], [np.conj(EEE), -1j * np.conj(V)]])
        
        Mtx = np.real(0.5 * Mtx1 @ Mtx2 @ Mtx3)
        Mtx = np.linalg.inv(Mtx)
        
        Delta = Mtx @ np.block([[np.real(g)], [np.imag(g)]])
        
        rp = rp - Delta
       
        R = np.diag(np.squeeze(rp[:N]))
        Phi = np.diag(np.squeeze(rp[N:]))
        
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
            permiso = 0
            break
        
    end_time = time.time()
    ve = R @ expm(1j * Phi) @ unos
      
    return np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRCinv(barY, barY0, barS, R0, Phi0, itmax, prec):
    N = R0.shape[0]
    Id = np.eye(N)
    unos = np.ones((N, 1))
    conY0 = np.conj(barY0)
    conY = np.conj(barY)
    invconY = np.linalg.inv(conY)
      
    permiso = True
    itcont = 0
     
    R = R0
    Phi = Phi0 
    rp = np.block([[R @ unos], [Phi @ unos]])
    
    start_time = time.time()
    while permiso:
        EEE = expm(1j * Phi)
        V = R @ EEE
        g = gue(V, barY, barY0, barS)
      
        vvv  = V@unos
        Vinv = np.diag(np.squeeze(1/vvv))
      
        AAA = conY0 + np.diag(np.squeeze(conY @ np.conj(V) @ unos))
        FFF = invconY@Vinv@AAA
        HHH = np.linalg.inv(Id -FFF@np.conj(FFF))@invconY@Vinv
        GGG = -np.conj(FFF)@HHH
        
        Mtx1 = np.block([[np.conj(EEE), EEE], [-1j * Vinv, 1j * np.conj(Vinv)]])
        Mtx2 = np.block([[GGG, np.conj(HHH)], [HHH, np.conj(GGG)]])
        Mtx3 = np.block([[Id, 1j*Id], [Id, -1j * Id]])
        
        Mtx = np.real(0.5 * Mtx1 @ Mtx2 @ Mtx3)
        
        Delta = Mtx @ np.block([[np.real(g)], [np.imag(g)]])
        
        rp = rp - Delta
        
        R = np.diag(np.squeeze(rp[:N]))
        Phi = np.diag(np.squeeze(rp[N:]))
        
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
            permiso = 0
            break
      
    end_time = time.time()
    ve = R @ expm(1j * Phi) @ unos
    
    return np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRD(barY, barY0, barS, R0, Phi0, itmax, prec):
    N = R0.shape[0]
    unos = np.ones((N, 1))
    invY = np.imag(np.linalg.inv(barY))
      
    permiso = True
    itcont = 0
      
    R  = R0
    Phi = Phi0
    r  = R@unos
    ph = Phi@unos
      
    start_time = time.time()
    while permiso:
        V = R @ expm(1j * Phi)
        g = gue(V, barY, barY0, barS)
        
        r  = r - invY@np.imag(g)
        ph  = ph - invY@np.real(g)
          
        R = np.diag(np.squeeze(r))
        Phi = np.diag(np.squeeze(ph))
          
        itcont = itcont + 1
        if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
            permiso = 0
            break
      
    end_time = time.time()
    ve = R @ expm(1j * Phi) @ unos
      
    return np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def gue(VVV, barY, barY0, barS):
    unos = np.ones((VVV.shape[0], 1))
    g = VVV @ np.conj(barY0) @ unos + VVV @ np.conj(barY) @ np.conj(VVV @ unos) - barS @ unos
    return g