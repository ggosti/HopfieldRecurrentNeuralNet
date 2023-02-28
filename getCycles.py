#import learningRNN as lrnn
import numpy as np
import matplotlib.pyplot as plt
import time


##################################

def stateIndex2stateVec(m,N,typ = 1):
    """
    Returns the binary vector sigma_0 that corresponds to the index m, where m is a int between 0 and 2**N
    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
    typ=1 --> {-1,1}    typ=0 --> {0,1} 
    """
    sigma_0 = [ (1+typ)* (int(float(m)/2**i) % 2) - typ for i in range(0,N)]    # typ=1 --> [-1,1]    typ=0 --> [0,1]
    sigma_0.reverse()
    sigma_0 = np.array(sigma_0)
    return sigma_0


# sigma_1 --> decimale = k
def stateVec2stateIndex(sigma,N,typ = 1):
    """
    Returns the index m that corresponds to the binary vector sigma_0, where m is a int between 0 and 2**N
    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
    typ=1 --> {-1,1}    typ=0 --> {0,1} 
    """
    k=int(0)
    for i in range(0,N):
        k=k+(sigma[i]+typ)/(1+typ)*2**(N-i-1)   # typ=1 --> [-1,1]    typ=0 --> [0,1]
    return int(k)

def stateIndex2stateVecSeq(ms,N,typ = 1):
    """
    Returns a list of binary vectors sigmas that correspond to the list of indexs ms, 
    where m in ms is a int between 0 and 2**N
    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
    typ=1 --> {-1,1}    typ=0 --> {0,1} 
    """
    # type: (state index sequence, number of neurons, typ) -> state vector sequence
    sigmas = [ stateIndex2stateVec(m,N,typ) for m in ms]
    sigmas = np.array(sigmas)
    return sigmas

def stateVec2stateIndexSeq(sigmas,N,typ = 1):
    """
    Returns a list of bindexes ms that correspond to the list of binary vectors sigmas,
    where m in ms is a int between 0 and 2**N
    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
    typ=1 --> {-1,1}    typ=0 --> {0,1} 
    """
    ms = [ stateVec2stateIndex(s,N,typ) for s in sigmas]
    ms = np.array(ms)
    return ms

def transPy(sigma_path0,net1,N,typ = 1, thr = 0):
    """
    transiton function. net1 is the network that generates the ttransitions
    
    If sigma_path0 is a binary vector it generates the corresponding transtions.
    
    If sigma_path0 is a list of binary vectors it generates a list with the corresponding transtions.
    
    typ determins if the neuron activation state is defined in {-1,1} or {0,1} 
    typ=1 --> {-1,1}    typ=0 --> {0,1} 
    """
    if not net1.dtype is np.dtype(np.float32):#net1 == np.float32:
        net1 = np.float32(net1)
    if not sigma_path0.dtype is np.dtype(np.float32):
        sigma_path0 = np.float32(sigma_path0)
    sigma_path1 = net1.dot(sigma_path0.T)
    #print(sigma_path1)
    sigma_path1 [sigma_path1  == 0] = 0.000001
    #print(sigma_path1)
    sigma_path1 = (1-typ+np.sign(sigma_path1 +thr))/(2-typ)
    #print(sigma_path1)
    return sigma_path1.T   

################################
# Generate matrices util
###############################3

def symmetrizeOld(matrix):
    assert (matrix.shape[0] == matrix.shape[1])
    matrix2 = matrix.copy()
    for x in range(matrix.shape[0]):
        for y in range(x):
            matrix2[x, y] = matrix[y, x]
    return matrix2

def symmetrize(matrix):
    assert (matrix.shape[0] == matrix.shape[1])
    #print np.triu(matrix.T,0)
    #print np.tril(matrix.T,-1)
    matrix2 = np.triu(matrix) + np.tril(matrix.T, -1)
    return matrix2


def antisymmetrizeOld(matrix):
    assert (matrix.shape[0] == matrix.shape[1])
    matrix2 = matrix.copy()
    for x in range(matrix.shape[0]):
        for y in range(x):
            matrix2[x, y] = -matrix[y, x]
    return matrix2

def antisymmetrize(matrix):
    assert (matrix.shape[0] == matrix.shape[1])
    #print np.triu(matrix.T,0)
    #print np.tril(matrix.T,-1)
    matrix2 = np.triu(matrix) - np.tril(matrix.T, -1)
    return matrix2

def sparseOld(matrix, rho):
    N = matrix.shape[0]
    randMat = np.random.rand(N,N)
    for x in range(N):
        for y in range(x):
            if randMat[x, y] < rho:
                matrix[x, y] = 0
                matrix[y, x] = 0
    print(np.sum(matrix!=0)/(n*(n-1)))
    #return matrix


def sparse(matrix, rho):
    N = matrix.shape[0]
    mask = (np.random.rand(N,N) < rho)
    mask = symmetrize(mask)
    matrix[mask] = 0

def get_connectivity_matrix(my_seed,N,rho,epsilon):
    '''
    This matrix is generated by the method described in
    Effect of dilution in asymmetric recurrent neural
    networks
    https://doi.org/10.1016/j.neunet.2018.04.003
    Parameters:
    ==========
    N: size of the networks
    my_seed: each matrix is defined by its seed through Numpy standard
                Marsenne-Twister
    Return:
    =======
    matrix:     1-(eps/2) * S_rho + (eps/2) * A_rho
                where _rho stands for the density of zero elements
    '''
    np.random.seed(my_seed)
    S = np.random.uniform(-1,1,(N,N))
    sparse(S, rho)
    S = symmetrize(S)  #(2*np.random.rand(N, N) - np.ones((N,N)))
    np.fill_diagonal(S,0)
    A = np.random.uniform(-1,1,(N,N))
    sparse(A, rho)
    A = antisymmetrize(A) #( 2*np.random.rand(N, N) - np.ones((N,N)) )
    np.fill_diagonal(A,0)
    return (1 - epsilon/2.)*S + epsilon/2.*A

def get_connectivity_matrix_fortranStyle(N,rho,epsilon):
    Ms = np.zeros((N,N))
    Ma = np.zeros((N, N))
    Matrice = np.zeros((N, N))
    for i in range(0,N):
        for j in range(0,N):
            A1 = (np.random.rand()-0.5)*2.
            A2 = (np.random.rand()-0.5)*2.
            B1 = np.random.rand()
            if B1 < rho:
                A1 = 0.0
            B2 = np.random.rand()
            if B2 < rho:
                A2 = 0.0
            Ms[i,j] = A1
            Ms[j,i] = A1
            Ma[i,j]=A2
            Ma[j,i]=-A2
    for i in range(0,N):
        for j in range(0,N):
            epsMez = epsilon/2.
            Matrice[i,j]=(1.- epsMez )*Ms[i,j] + epsMez * Ma[i,j]
    return Matrice

def getCycles(C,N,typ,thr):
    first_states_index = np.array((range(2**N)))
    #print('first_states_index')
    #print(first_states_index)
    first_states = stateIndex2stateVecSeq(first_states_index, N, typ)

    allCycles = []
    cycles = {}
    nodesInLCycles = {}
    curr_states = first_states
    for l in range(2**N): # find all cycle of increasing lenght l+1
        #print('---- Cycle length ',l,' -----')
        if len(curr_states) > 0:
            next_states = transPy(curr_states,C, N, typ, thr)
            next_states_indexes = stateVec2stateIndexSeq(next_states,N,typ)
            cyclesLGate = (first_states_index == next_states_indexes)
            if np.sum(cyclesLGate)>0:
                cycles[l]=[]
                #print('first_states_index')
                #print(first_states_index)
                #print("cyclesLGate",cyclesLGate)
                #print(first_states_index[cyclesLGate])
                #print(lCycles)
                nodesInLCycles[l] = list(first_states_index[cyclesLGate])
                #print(nodesInLCycles)
                curr_states = next_states[np.logical_not(cyclesLGate)]
                first_states_index = first_states_index[np.logical_not(cyclesLGate)]
                #lCycles = list(first_states_index[cyclesLGate])
            else:
                curr_states = next_states 
                #while(len(lCycles)>0):
                #    start = lCycles.pop()
                #    #print('start',start)
                #    start_vec = stateIndex2stateVec(start,N,typ)
                #    cycle = [start]
                #    for t in range(l):
                #        next_vec = transPy(start_vec,C, N, typ, thr)
                #        next = stateVec2stateIndex(next_vec,N,typ)
                #        #print('next',next)
                #        lCycles.remove(next)
                #        cycle.append(next)
                #        start_vec = next_vec
                #    #print('cycle',cycle)
                #    cycles[l].append(cycle)
                #    allCycles.append(cycle)
    return cycles,allCycles,nodesInLCycles


def getCyclesNX(C,N,typ,thr):
    import networkx as nx
    G = nx.DiGraph()
    #t1 = time.time()
    initial_states = stateIndex2stateVecSeq(range(np.power(2, N)), N, typ)
    final_states = transPy(initial_states, C, N, typ, thr)
    #t2 = time.time()
    #print 'generated trans. matrix',t2-t1

    #t1 = time.time()
    labels = {}
    for pre, post in zip(initial_states, final_states):  # add 100 sets of 100 1's
        prek = int(stateVec2stateIndex(pre, N, typ))
        postk = int(stateVec2stateIndex(post, N, typ))
        #print(prek, postk)
        #print(pre, post)
        G.add_edge(prek, postk)
        labels[prek] = pre

    loops = list(nx.cycles.simple_cycles(G))
    #print 'loops', loops
    #t2 = time.time()
    #print 'computed number of loops',t2-t1

    return loops, G

def test():
    N = 4
    m = np.random.rand(N, N)
    m1 = symmetrizeOld(m)
    m2 = symmetrize(m)
    print ((m1 == m2).all())
    a1 = antisymmetrizeOld(m)
    a2 = antisymmetrize(m)
    print ((a1 == a2).all())

def test2():
    N = 5
    typ = 0
    thr = 0
    #M1 = [[ 0.       ,  -0.18223438, -0.055102468, 0.2905751, -0.21429572],
    # [-0.581795 ,   0.        , -0.08177844 , 0.35558614, 0.006734699],
    # [0.11584915, -0.08792217 ,  0.         , -0.43793896, -0.022829354],
    # [0.5258726, 0.3362002   ,  0.41928437 ,  0.        ,  0.20275094],
    # [0.36236843, -0.88517904,  0.3373032,   -0.34560305,   0.       ]]
    M1 = [[0.0000000E+0000, - 0.5171256E+0000, -0.2989531E-0001, -0.7808704E+0000, 0.7945437E+0000],
          [-0.5171256E+0000, 0.0000000E+0000, -0.2580485E+0000, -0.6977754E+0000, -0.8953681E+0000],
          [-0.2989531E-0001, -0.2580485E+0000, 0.0000000E+0000, 0.8754015E+0000, 0.9808731E+0000],
          [-0.7808704E+0000, -0.6977754E+0000, 0.8754015E+0000, 0.0000000E+0000, -0.1136637E+0000],
          [0.7945437E+0000, -0.8953681E+0000, 0.9808731E+0000, -0.1136637E+0000, 0.0000000E+0000]]
    #M1 = [[0.000000000000000000000000E+0000, 0.350680351257324218750000E-0001, 0.638692855834960937500000E+0000, -0.755703926086425781250000E+0000, -0.619309902191162109375000E+0000],
    #      [-0.392206668853759765625000E+0000, 0.000000000000000000000000E+0000, -0.483913183212280273437500E+0000, 0.251545906066894531250000E-0001, 0.710200786590576171875000E+0000],
    #      [0.159279346466064453125000E+0000, -0.138426065444946289062500E+0000, 0.000000000000000000000000E+0000, 0.598297119140625000000000E-0001, 0.119009733200073242187500E+0000],
    #      [0.151961803436279296875000E+0000, -0.474640369415283203125000E+0000, 0.674424648284912109375000E+0000, 0.000000000000000000000000E+0000, 0.402131080627441406250000E-0001],
    #      [0.745916366577148437500000E-0002, -0.261993408203125000000000E+0000, - 0.740401744842529296875000E-0001, 0.243728160858154296875000E+0000, 0.000000000000000000000000E+0000]]
    M1 = np.array(M1).T
    print(M1)
    cycles,allCycles,nodesInLCycles = getCycles(M1, N, typ, thr)
    print('nodesInLCycles',nodesInLCycles)
    print('allCycles',allCycles,len(allCycles))

    loops, G = getCyclesNX(M1, N, typ, thr)
    print('loops', loops)
    num = len(loops)
    print(num)
    print('test',num == 5)

def test3():
    N = 5
    typ = 0
    thr = 0
    #M1 = [[ 0.       ,  -0.18223438, -0.055102468, 0.2905751, -0.21429572],
    # [-0.581795 ,   0.        , -0.08177844 , 0.35558614, 0.006734699],
    # [0.11584915, -0.08792217 ,  0.         , -0.43793896, -0.022829354],
    # [0.5258726, 0.3362002   ,  0.41928437 ,  0.        ,  0.20275094],
    # [0.36236843, -0.88517904,  0.3373032,   -0.34560305,   0.       ]]
    #M1 = [[0.0000000E+0000, - 0.5171256E+0000, -0.2989531E-0001, -0.7808704E+0000, 0.7945437E+0000],
    #      [-0.5171256E+0000, 0.0000000E+0000, -0.2580485E+0000, -0.6977754E+0000, -0.8953681E+0000],
    #      [-0.2989531E-0001, -0.2580485E+0000, 0.0000000E+0000, 0.8754015E+0000, 0.9808731E+0000],
    #      [-0.7808704E+0000, -0.6977754E+0000, 0.8754015E+0000, 0.0000000E+0000, -0.1136637E+0000],
    #      [0.7945437E+0000, -0.8953681E+0000, 0.9808731E+0000, -0.1136637E+0000, 0.0000000E+0000]]
    M1 = [[0.000000000000000000000000E+0000, 0.350680351257324218750000E-0001, 0.638692855834960937500000E+0000, -0.755703926086425781250000E+0000, -0.619309902191162109375000E+0000],
          [-0.392206668853759765625000E+0000, 0.000000000000000000000000E+0000, -0.483913183212280273437500E+0000, 0.251545906066894531250000E-0001, 0.710200786590576171875000E+0000],
          [0.159279346466064453125000E+0000, -0.138426065444946289062500E+0000, 0.000000000000000000000000E+0000, 0.598297119140625000000000E-0001, 0.119009733200073242187500E+0000],
          [0.151961803436279296875000E+0000, -0.474640369415283203125000E+0000, 0.674424648284912109375000E+0000, 0.000000000000000000000000E+0000, 0.402131080627441406250000E-0001],
          [0.745916366577148437500000E-0002, -0.261993408203125000000000E+0000, - 0.740401744842529296875000E-0001, 0.243728160858154296875000E+0000, 0.000000000000000000000000E+0000]]
    M1 = np.array(M1).T
    print(M1)
    cycles,allCycles,nodesInLCycles = getCycles(M1, N, typ, thr)
    print('nodesInLCycles',nodesInLCycles)
    print('allCycles',allCycles,len(allCycles))

    loops, G = getCyclesNX(M1, N, typ, thr)
    print('loops', loops)
    num = len(loops)
    print(num)
    print('test',num == 1)

if __name__ == "__main__":
    print("do tests")
    test()
    test2()
    test3()
    print("end tests")

    N = 10
    typ = 0 # typ = 0 neurons with binary activation states {0,1}, typ = 1  neurons with states {-1,1}.
            # typ=1 --> {-1,1}    typ=0 --> {0,1} 
    thr = 0
    runNum = 1000#1000
    seed = 8888#int(np.random.rand() * 10000)
    np.random.seed(seed)
    rhos = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.8,0.85,0.9,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99] #[0.0]#[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.8,0.85,0.9,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]

    avCycles = []
    for rho in rhos:
        numLoops = []
        for run in range(runNum):

            #print 'seed',seed
            #C = get_connectivity_matrix(seed,N=N,rho=rho,epsilon=1.0)
            C = get_connectivity_matrix_fortranStyle(N=N,rho=rho,epsilon=1.0)
            
            #cycles,allCycles,nodesInLCycles = getCycles(C, N, typ, thr)
            #print('nodesInLCycles',nodesInLCycles)

            loops,G = getCyclesNX(C, N, typ, thr)
            # print('loops', loops)
            num = len(loops)
            #print num
            numLoops.append(num)
            if False:

                pos = nx.layout.spring_layout(G)
                nodes = nx.draw_networkx_nodes(G, pos, node_color='blue')
                edges = nx.draw_networkx_edges(G, pos, arrowstyle='->',
                                               arrowsize=10, width=2)
                nx.draw_networkx_labels(G, pos, labels, font_size=16)

                ax = plt.gca()
                ax.set_axis_off()
                plt.show()

        print(numLoops)
        avL = np.mean(numLoops)
        print('rho',rho,'acLoops',avL)
        avCycles.append(avL)
    plt.figure()
    plt.plot(rhos,avCycles)
    plt.show()
