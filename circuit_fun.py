import numpy as np
import random
import math
import matplotlib.pyplot as plt
#############################################################
def load_circuit(nodes, LoadMean, dev):
    N = nodes.shape[0]
    
    load = np.array([random.gauss(LoadMean, dev) for _ in range(N - 1)])
    load = np.clip(load, 0, None)  #eliminate negatives
    classes = 3  #residential,industrial,commercial
    loadmix = np.random.dirichlet(np.ones(classes), N - 1)
    
    return load, loadmix
#############################################################
def DG_circuit(nodes, PVprob, PVmean, PVdev):
    N = nodes.shape[0]
    
    pv = np.array([
      random.gauss(PVmean, PVdev) *
      random.choices([0, 1], weights=[1 - PVprob, PVprob])[0]
      for _ in range(N - 1)
    ])
    pv = np.clip(pv, 0, None)  #eliminate negatives
    
    return pv

#############################################################
def impendances_circuit(lines, load):
    ###############
    S = 10*np.ceil(load/10).sum()
    N  = load.shape[0] + 1 #N nodes + 1 root
    NB = lines.shape[0] #num of branches
    ###############
    Rperkm = 0.2870120
    Xperkm = 0.5508298
    Rpermt = Rperkm / 1000
    Xpermt = Xperkm / 1000
    ###############
    children = {}
    for nn in range(N):
        children[nn] = [nn]
    for ll in range(NB):
        newchild = lines[ll,1].astype(int)-1
        parent = lines[ll, 0].astype(int)-1
        #add as child of immediate parent
        if newchild not in children.get(parent,[]):
            children[parent].append(newchild)
        #add as child of grand-parents
        for grandparent in range(ll):
            if parent in children.get(grandparent,[]):
                if newchild not in children.get(grandparent,[]):
                    children[grandparent].append(newchild)
    ratedpower = np.zeros(N)
    powers = 10*np.ceil(np.concatenate(([0], load))/10)
    for child in children:
        ratedpower[child] = powers[children.get(child,[])].sum()
    ###############
    epsilon = 1/100
    upper = 1 + epsilon
    lower = 1- epsilon
    ###############
    W = np.zeros((NB, NB), dtype=complex)  #admitances of each line
    D = np.zeros((NB, N), dtype=int)  #incidence matrix
    
    for ll in range(NB):
        origin  = lines[ll, 0].astype(int) - 1
        destiny = lines[ll, 1].astype(int) - 1
        Rll = Rpermt
        Xll = Xpermt
        if ratedpower[destiny]>5000:
            factor = (5000/ratedpower[destiny])**2
            Rll = Rll*factor
            Xll = Xll*factor
        #impedance
        Z = lines[ll, 2] * ( Rll* random.uniform(lower, upper) + 1j * Xll * random.uniform(lower, upper) )
        #admittance
        W[ll, ll] = 1 / Z
        #incidences
        D[ll, origin] = -1
        D[ll, destiny] = 1
    
    #admitances matrix
    hatY = np.transpose(D) @ W @ D
    
    #partitionate adm matrx
    Y = hatY[1:N, 1:N]
    Y0 = np.diag(hatY[0, 1:N])
    Y00 = hatY[0, 0]
    
    return Y, Y0, Y00
#############################################################
def summary_circuit(nodes, lines,load,dg,plt_name):
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    N = nodes.shape[0]
    Nl = lines.shape[0]
    Sload = 10*np.ceil(load/10).sum()/1000
    Sdg = np.round(dg.sum())/1000
    length = np.round(lines[:,2].sum())/1000
    
    print('Example circuit nodes: %s' % N)
    print('Example circuit lines: %s' % Nl)
    print('Example circuit rated power (load): %s [MVA]' % Sload)
    print('Example circuit installed DG: %s [MVA]' % Sdg)
    print('Example circuit total length: %s [km]' % length)
    
    
    cts_data = (N,Sload,Sdg,length)
    print_circuit(nodes, lines, plt_name,cts_data)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    return
#############################################################
def print_circuit(nodes, lines, plt_name,cts_data):
    #prepare figure
    fig, ax = plt.subplots(figsize=(10 * 2 / 2.54, 10 * 2 / 2.54))
    #ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    #nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], color='green',zorder=2)
    #lines
    rows, cols = lines.shape
    for ll in range(0, rows):
        xx = np.array([
          nodes[lines[ll, 0].astype(int) - 1, 0],
          nodes[lines[ll, 1].astype(int) - 1, 0]
        ])
        yy = np.array([
          nodes[lines[ll, 0].astype(int) - 1, 1],
          nodes[lines[ll, 1].astype(int) - 1, 1]
        ])
        ax.plot(xx, yy, linestyle='-', color='blue', alpha=0.5,zorder=1)
    #labels on nodes
    for i in range(len(nodes[:, 0])):
        ax.text(nodes[i, 0]+40, nodes[i, 1]+40, i, fontsize=10, ha='center', va='center',color='green',zorder=2)    
    #find corners coordinates
    plt.tight_layout()
    left, right = ax.get_xlim()
    lower,upper = ax.get_ylim()
    width = right-left
    height = upper-lower
    #draw scale reference at upper left corner
    scale0x = left + 20
    scale0y = upper - 20
    ax.plot([scale0x,scale0x+400], [scale0y,scale0y], linestyle='-', color='black')
    ax.plot([scale0x,scale0x], [scale0y+10,scale0y-10], linestyle='-', color='black')
    ax.plot([scale0x+100,scale0x+100], [scale0y+10,scale0y-10], linestyle='-', color='black')
    ax.plot([scale0x+200,scale0x+200], [scale0y+10,scale0y-10], linestyle='-', color='black')
    ax.plot([scale0x+300,scale0x+300], [scale0y+10,scale0y-10], linestyle='-', color='black')
    ax.plot([scale0x+400,scale0x+400], [scale0y+10,scale0y-10], linestyle='-', color='black')
    
    ax.text(scale0x+200, scale0y-15, '200[m]', fontsize=8, ha='center', va='top',color='black') 
    ax.text(scale0x+400, scale0y-15, '400[m]', fontsize=8, ha='center', va='top',color='black') 
    #find corner with more/less points
    corner_upleft = 0
    corner_upright = 0
    corner_dwnleft = 0
    corner_dwnright = 0
    for i in range(len(nodes[:, 0])):
        left_side = False
        lower_side = False
        right_side = False
        upper_side = False
        if nodes[i,0] < (left + width/3):
            left_side = True
        if nodes[i,0] > (right - width/3):
            right_side = True
        if nodes[i,1] < (lower + height/3):
            lower_side = True
        if nodes[i,1] > (upper - height/3):
            upper_side = True
        
        if left_side and lower_side:
            corner_dwnleft = corner_dwnleft + 1
        if left_side and upper_side:
            corner_upleft = corner_upleft + 1
        if right_side and lower_side:
            corner_dwnright = corner_dwnright + 1
        if right_side and upper_side:
            corner_upright = corner_upright + 1
    
    minfreecorner = min((corner_upright,corner_dwnleft,corner_dwnright,corner_upleft))
    if minfreecorner == corner_upleft:
        datax = scale0x
        datay = scale0y - 100
        vall  = 'top'
        hall  = 'left'
    if minfreecorner == corner_upright:
        datax = right-20
        datay = scale0y
        vall  = 'top'
        hall  = 'right'
    if minfreecorner == corner_dwnleft:
        datax = scale0x
        datay = lower+20
        vall  = 'bottom'
        hall  = 'left'
    if minfreecorner == corner_dwnright:
        datax = right-20
        datay = lower+20
        vall  = 'bottom'
        hall  = 'right'
    
    #draw ct data
    N,Sload,Sdg,length = cts_data
    
    ax.text(datax, datay, 
            'Non-root nodes: '+str(N-1)+'\n'+
            'Total Rated Power: '+str(Sload)+'[MVA]\n'+
            'Total installed PV: '+str(Sdg)+'[MVA]\n'+
            'Total length: '+str(length)+'[km]',
            fontsize=12, ha=hall, va=vall,color='black') 
    
    # Saving the plot to an image file
    fig.savefig(plt_name+'.eps', format='eps')
    plt.show()
    fig.clf()
    return
#############################################################
def create_circuit(N, Dmin, Dmax):
    #input parameters
    Dm = 0.5 * (Dmin + Dmax)
    
    x0 = 0
    y0 = 0
    theta = 0
    dTheta = 2
    
    #prefill vectors
    nodes = np.zeros((N + 1, 3))  #x,y,angle
    vecinos = np.zeros((N + 1, 1))  #number of neighbours
    lines = np.zeros((N, 3))  #ii origin, jj destiny
    
    nodes[0, :] = [x0, y0, theta]
    ll = 1  #lines counters
    PossibleExtraLines = np.zeros((2 * N, 3))
    ExtraLine = 0  #counter of extralines
    #length of branches (in nodes)
    BranchMin = 4
    BranchMax = min(200, round(N / math.log(N)))
    BranchLength = BranchMin + round(random.random() * (BranchMax - BranchMin))
    Branch = 0  #lines  in branch
    iiorigin = 1  #previous node
    #run over the nodes
    for ii in range(2, N + 2):
        #get origin
        Origin = nodes[iiorigin - 1, 0:2]
        #new position
        dd = Dm * random.gauss(1, 0.5)
        dd = min(Dmax, max(Dmin, dd))
      
        dth = dTheta * random.random() * random.choice([-1, 1])
        theta = theta + dth
      
        x = Origin[0] + dd * math.cos(math.radians(theta))
        y = Origin[1] + dd * math.sin(math.radians(theta))
        #save position
        nodes[ii - 1, :] = [x, y, theta]
        lines[ll - 1, :] = [iiorigin, ii, dd]
        ll = ll + 1
        vecinos[ii - 1] = vecinos[ii - 1] + 1
        vecinos[iiorigin - 1] = vecinos[iiorigin - 1] + 1
        #check that it is not the last of branch
        Branch = Branch + 1
        last = Branch >= BranchLength
        #check that there are no other nodes too close
        cerca = 0
        for jj in range(1, ii - Branch - 1):
          if jj != iiorigin:
              xjj = nodes[jj - 1, 0]
              yjj = nodes[jj - 1, 1]
              xf = x + Dmax * math.cos(math.radians(theta))
              yf = y + Dmax * math.sin(math.radians(theta))
          
              d1 = math.sqrt((x - xjj)**2 + (y - yjj)**2)
              d2 = math.sqrt((xf - xjj)**2 + (yf - yjj)**2)
              dij = max(d1, d2)
          
              if dij < Dmax:
                  cerca = 1
                  if Branch > 1:
                      if ExtraLine < 2 * N:
                          ExtraLine = ExtraLine + 1
                          PossibleExtraLines[ExtraLine - 1, :] = [ii, jj, d1]
                      break
        if last or cerca:
            BranchLength = BranchMin + round(random.random() *
                                             (BranchMax - BranchMin))
            iiorigin = 1 + round((ii - 1) * random.random())
            while vecinos[iiorigin - 1] > 3 or vecinos[iiorigin - 1] < 2:
                iiorigin = 1 + round((ii - 1) * random.random())
            dth = -90
            if vecinos[iiorigin - 1] < 3:
                dth = 90
            theta = nodes[iiorigin - 1, 2] + dth
            Branch = 0
        else:
            iiorigin = ii
            theta = nodes[iiorigin - 1, 2]
    
    #aglomerate lines
    criteria1 = PossibleExtraLines[:, 2] > Dmin
    criteria2 = PossibleExtraLines[:, 2] < 2 * Dmax
    PossibleExtraLines = PossibleExtraLines[criteria1 & criteria2]
    lines = np.concatenate((lines, PossibleExtraLines), axis=0)
    #clean unnecessary info from nodes
    nodes = nodes[:, 0:2]
    
    #output the circuit
    return nodes, lines