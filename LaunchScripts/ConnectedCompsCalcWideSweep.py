import numpy as np
import sys
#Usage:
#python thisprog.py threshold numofnetworks
#Will randomly initialize numofnetworks neural networks and train them until the error on a training set is less than threshold
#Will then try to interpolate between these networks while keeping error below that of threshold.
#Will tabulate the number of connected components found in this way





#Simple network: Given three integers a,b,c, [-100,100] chooses three random x-values, and evaluates
#the quadratic function a*x^2 + b*x + c at those values.
def func(x,a,b,c):
    return x*x*a + x*b + c

def generatecandidate3(a,b,c):


    candidate = [np.random.random() for x in xrange(1)]
    candidatesolutions = [func(x,a,b,c) for x in candidate]
    
    
    return candidate, candidatesolutions
	
	

import copy

alpha,hidden_dim,hidden_dim2 = (.001,12,4)

threshrange = np.linspace(.03,.1,101)

thresh = threshrange[int(sys.argv[1])%100]

dimsweep = int(int(sys.argv[1]) // 1000)
lengthsweep = [24,48,96]

alpharange = (.0005, .0001, .00005)
alpha = alpharange[dimsweep]
hidden_dim = lengthsweep[dimsweep]

synapses = []

#Testing starting in the same place
#synapse0 = 2*np.random.random((1,hidden_dim)) - 1
#synapse1 = 2*np.random.random((hidden_dim,hidden_dim2)) - 1
#synapse2 = 2*np.random.random((hidden_dim2,1)) - 1

for i in xrange(int(sys.argv[2])):
    
    synapse_0 = 2*np.random.random((1,hidden_dim)) - 1
    synapse_1 = 2*np.random.random((hidden_dim,1)) - 1
    #synapse_2 = 2*np.random.random((hidden_dim2,1)) - 1
    
    #synapse_0 = copy.deepcopy(synapse0)
    #synapse_1 = copy.deepcopy(synapse1)
    #synapse_2 = copy.deepcopy(synapse2)
    
    #remove the comment to get random initialization
    stopcond = True
    
    while stopcond:
        #print 'epoch:' + str(e)
        X = []
        y = []

        for i in xrange(10000):

            a,b = generatecandidate3(.5,.25,.1)
            X.append(a)
            y.append(b)
        X= np.array(X)
        y=np.array(y)

        j = 0

        while stopcond:


            #if j%5000 == 0: print j
            layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))

            #if(False):
            #    dropout_percent = .1
            #    layer_1 *= np.random.binomial([np.ones((len(X),hidden_dim))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))


            layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))
            #if(True):
            #    dropout_percent = .2
            #    layer_2 *= np.random.binomial([np.ones((len(layer_1),hidden_dim2))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))



            #layer_3 = 1/(1+np.exp(-(np.dot(layer_2,synapse_2))))
            #if(False):
            #    dropout_percent = .25
            #    layer_2 *= np.random.binomial([np.ones((len(layer_2),2))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))


            layer_2_delta = (layer_2- y)*(layer_2*(1-layer_2))
            #layer_2_delta = layer_3_delta.dot(synapse_2.T) * (layer_2 * (1-layer_2))
            layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))

            #synapse_2 -= (alpha * layer_2.T.dot(layer_3_delta))
            synapse_1 -= (alpha * layer_1.T.dot(layer_2_delta))
            synapse_0 -= (alpha * X.T.dot(layer_1_delta))

            # how much did we miss the target value?
            layer_2_error = layer_2 - y

            #if (j%50) == 0:
            #    print "Error after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error)))
            
            if np.mean(np.abs(layer_2_error)) < thresh:
                #print "Changing stopcond!"
                stopcond = False
            j+=1
    #remove the comment to get random initialization

    synapses.append([synapse_0,synapse_1])#,synapse_2])
	
	
	
	
	
#Idea: Take two networks as input.  Construct string connecting two nework with "beads" along the string.
#Stochastically (monte carlo?  simulated annealing?) wiggle the beads until the max on the beads is minimized


from random import gauss
import copy

def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]




#Definition for test set:

'''X = []
y = []

for i in xrange(100):
    j = i/100.
    a,b = [[j],[func(j,.5,.25,.1)]]
    X.append(a)
    y.append(b)

X= np.array(X)
y=np.array(y)'''



#returns a later thats t-between synapse1 and synapse2 (t ranges from 0 to 1)
def synapse_interpolate(synapse1, synapse2, t):
    return (synapse2-synapse1)*t + synapse1


X = []
y = []




def GenTest(X, y):
    
    X = []
    y = []
    for i in xrange(1000):
    
    
        a,b = generatecandidate3(.5,.25,.1)
        X.append(a)
        y.append(b)
        
    return np.array(X), np.array(y)


X, y = GenTest(X,y)




#Simple container to hold the weights defined on the beads

class WeightString:
    
    def __init__(self, w1, w2, numbeads, threshold, springk):
        self.w1 = w1
        self.w2 = w2
        self.beads = []
        self.velocity = []
        self.threshold = threshold
        self.springk = springk
        
        

        for n in xrange(numbeads):
            beaddata = []
            for k in xrange(len(self.w1)):
                beaddata.append(synapse_interpolate(self.w1[k],self.w2[k], (n + 1.)/(numbeads+1.)))
            self.beads.append(beaddata)
        
        self.velocity = copy.deepcopy(self.beads)
        for b in self.velocity:
            for v in b:
                v = 0.*v
        
        #self.beads.reverse()
        
        self.InitialEnergy = self.SpringEnergy()
        self.AllBeads = copy.deepcopy(self.beads)
        self.AllBeads.insert(0,self.w1)
        self.AllBeads.append(self.w2)
        
        self.ConvergedList = [False for f in xrange(len(self.AllBeads))]
        self.ConvergedList[0] = True
        self.ConvergedList[-1] = True
    
    
    def SpringNorm(self, order):
        
        total = 0.
        
        #Energy between mobile beads
        for i,b in enumerate(self.AllBeads):
            if i < len(self.AllBeads)-1:
                #print "Tallying energy between bead " + str(i) + " and bead " + str(i+1)
                subtotal = 0.
                for j in xrange(len(b)):
                    subtotal += np.linalg.norm(np.subtract(self.AllBeads[i][j],self.AllBeads[i+1][j]),ord=order)#/len(self.beads[0][j])
                total+=subtotal
        
        return total#/len(self.beads)
        
    
    def SpringEnergy(self):
        
        total = 0.
        
        #Energy between the pinned, immobile weight and the first bead
        subtotal = 0.
        for j in xrange(len(self.beads[0])):
            subtotal += np.linalg.norm(np.subtract(self.w1[j],self.beads[0][j]),ord=2)/len(self.beads[0][j])
        total+=subtotal
        
        #Energy between mobile beads
        for i,b in enumerate(self.beads):
            if i < len(self.beads)-1:
                #print "Tallying energy between bead " + str(i) + " and bead " + str(i+1)
                subtotal = 0.
                for j in xrange(len(b)):
                    subtotal += np.linalg.norm(np.subtract(self.beads[i][j],self.beads[i+1][j]),ord=2)/len(self.beads[0][j])
                total+=subtotal
                
        #Energy between pinned, immobile final weights, and the last bead
        subtotal = 0.
        for j in xrange(len(self.beads[-1])):
            subtotal += np.linalg.norm(np.subtract(self.w2[j],self.beads[-1][j]),ord=2)/len(self.beads[0][j])
        total+=subtotal
        
        return total/len(self.beads)
    
    
    
    
    def SGDBead(self, bead, X, y):
        layers = []
        l1 = 1/(1+np.exp(-(np.dot(X,self.AllBeads[bead][0]))))
        layers.append(l1)
        for i,b in enumerate(self.AllBeads[bead][1:]):
            l = 1/(1+np.exp(-(np.dot(layers[-1],b))))
            layers.append(l)

        layersdelta = []
        l3 = (layers[-1] - y)*(layers[-1]*(1-layers[-1])) #+ (1./regparam)*OldSpringEnergy*np.ones(np.shape(y))
        layersdelta.append(l3)
        for i,l in enumerate(layers[:-1]):
            ldelta = layersdelta[-1].dot(self.AllBeads[bead][-1-i].T) * (layers[:-1][-1-i]) * (1- (layers[:-1][-1-i]))
            layersdelta.append(ldelta)
        
        for i in xrange(len(layers)-1):
            if -i-1 != 0:
                self.AllBeads[bead][-i-1] -= .001*layers[-i-2].T.dot(layersdelta[i])
            else:
                self.AllBeads[bead][0] -= .001*X.T.dot(layersdelta[-1])
        
        finalerror = (layers[-1] - y)
        
        return np.mean(np.abs(finalerror))
    
    

  
    #monte carlo update step
    def UpdateBead(self, temperature, bead, X, y):
        
        
        regparam = 100.
        
        OldSpringEnergy = self.SpringEnergy()
        OldMax = [EvalNet(b,X)-y for b in self.beads]
        OldMaxError = max([np.mean(np.abs(om)) for om in OldMax])
        oe = OldSpringEnergy/100000. + OldMaxError
        
        #print "Old SE: " + str(OldSpringEnergy)
        #print "Old Max: " + str(OldMax)
        ####print "Oldmaxerror: " + str(OldMaxError)
        
        oldweight = copy.deepcopy(self.beads[bead])
            
        layers = []
        #print bead[0]
        l1 = 1/(1+np.exp(-(np.dot(X,self.beads[bead][0]))))
        layers.append(l1)
        for i,b in enumerate(self.beads[bead][1:]):
            l = 1/(1+np.exp(-(np.dot(layers[-1],b))))
            layers.append(l)


        #layer_3_delta = (layer_3- y)*(layer_3*(1-layer_3))
        #layer_2_delta = layer_3_delta.dot(synapse_2.T) * (layer_2 * (1-layer_2))
        #layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))

        #layersdelta = []
        layersdelta = []
        l3 = (layers[-1] - y)*(layers[-1]*(1-layers[-1])) #+ (1./regparam)*OldSpringEnergy*np.ones(np.shape(y))
        layersdelta.append(l3)
        for i,l in enumerate(layers[:-1]):
            ldelta = layersdelta[-1].dot(self.beads[bead][-1-i].T) * (layers[:-1][-1-i]) * (1- (layers[:-1][-1-i]))
            layersdelta.append(ldelta)
        
        
        for i in xrange(len(layers)-1):
            #print i
            #print self.beads[bead][-i-1]
            #rint layers[-i-2].T
            #print layersdelta[-i-1]
            #print layers[-i-2].T.dot(layersdelta[-i-1])
            if -i-1 != 0:
                self.beads[bead][-i-1] -= .1*layers[-i-2].T.dot(layersdelta[i])
            else:
                self.beads[bead][0] -= .1*X.T.dot(layersdelta[-1])

            
            
            #The code below regularizes the network so that they stay near each other in weight space
            '''if bead == 0:
                self.beads[bead][-i-1] -= (np.subtract(self.beads[bead][-i-1],self.w1[-i-1]) + np.subtract(self.beads[bead+1][-i-1],self.beads[bead][-i-1]))/regparam
                
            if bead == len(self.beads)-1:
                self.beads[bead][-i-1] -= (np.subtract(self.w2[-i-1],self.beads[bead][-i-1]) + np.subtract(self.beads[bead][-i-1],self.beads[bead-1][-i-1]))/regparam
            
            if (bead > 0 and bead < len(self.beads)-1):
                self.beads[bead][-i-1] -= (np.subtract(self.beads[bead+1][-i-1],self.beads[bead][-i-1]) + \
                                           np.subtract(self.beads[bead][-i-1],self.beads[bead-1][-i-1]))/regparam'''
            
        #layers.reverse()
        
        

        # how much did we miss the target value?
        NewSpringEnergy = self.SpringEnergy()
        finalerror = (layers[-1] - y) #(1./regparam)*NewSpringEnergy*np.ones(np.shape(y))
        
        NewMaxError = np.mean(np.abs(finalerror))
        #print "New SE: " + str(NewSpringEnergy)
        #print "Old Max: " + str(OldMax)
        ####print "Newmaxerror: " + str(NewMaxError)
        ne = NewSpringEnergy/100000. + NewMaxError
        #print "Newtotal: " + str(ne)
        ####print "\n"
        
        myrand = np.random.rand()
        ####print "rand is: " + str(myrand) + " and boltzmann weight is " + str(np.exp(-(NewSpringEnergy - OldSpringEnergy)/temperature))
        
        if NewSpringEnergy > OldSpringEnergy:
        #if NewSpringEnergy > self.InitialEnergy:
            if NewMaxError > OldMaxError:
                self.beads[bead]=oldweight
            else:
                if myrand > np.exp(-(NewSpringEnergy - OldSpringEnergy)/temperature):
                #if myrand > np.exp(-(NewSpringEnergy - self.InitialEnergy)/temperature):
                    #print "Rejecting proposal"
                    self.beads[bead]=oldweight
        
        
        
        return True    
    
    
    #def JUST MAKE A PURE KINETIC EVOLVER, SWAP BETWEEN KINETIC EVOLUTION AND GRADIENT DESCENT
    
    def UpdateKinetic(self, dt, k):
        
        
        for bead in xrange(len(self.beads)):
            for i in xrange(len(self.beads[bead])):
                self.beads[bead][i] += dt*self.velocity[bead][i]
        
        for bead in xrange(len(self.beads)):
            for i in xrange(len(self.beads[bead])):
        
                if bead == 0:
                    self.velocity[bead][i] += -dt*k*(np.subtract(self.beads[bead][i],self.w1[i]) + np.subtract(self.beads[bead+1][i],self.beads[bead][i]))

                if bead == len(self.beads)-1:
                    self.velocity[bead][i] += -dt*k*(np.subtract(self.w2[i],self.beads[bead][i]) + np.subtract(self.beads[bead][i],self.beads[bead-1][i]))

                if (bead > 0 and bead < len(self.beads)-1):
                    self.velocity[bead][i] += -dt*k*(np.subtract(self.beads[bead+1][i],self.beads[bead][i]) + \
                                               np.subtract(self.beads[bead][i],self.beads[bead-1][i]))
                #self.velocity[bead][i] -= .1*self.velocity[bead][i]
        

        
    
    
    #monte carlo update step
    def UpdateBeadPureKinetic(self, temperature, bead):
        
        OldSpringEnergy = self.SpringEnergy()
        #OldMax = [EvalNet(b,X)-y for b in self.beads]
        #OldMaxError = max([np.mean(np.abs(om)) for om in OldMax])
        #oe = OldSpringEnergy/100000. + OldMaxError
        
        ##print "Old SE: " + str(OldSpringEnergy)
        #print "Old Max: " + str(OldMax)
        #print "Oldmaxerror: " + str(OldMaxError)
        #print "Oldtotal: " + str(oe)
        
        oldweight = copy.deepcopy(self.beads[bead])
        
        randupdates = []
        
        for i,syn in enumerate(self.beads[bead]):
            #create random perturbation to weight matrix with correct shape
            addtobead = np.reshape(make_rand_vector(syn.size),syn.shape)
            #add it to this particular bead
            self.beads[bead][i]+=.1*addtobead
        
        
        NewSpringEnergy = self.SpringEnergy()
        #NewMax = [EvalNet(b,X)-y for b in self.beads]
        #NewMaxError = max([np.mean(np.abs(om)) for om in OldMax])
        ##print "New SE: " + str(OldSpringEnergy)
        #print "Old Max: " + str(OldMax)
        #print "Newmaxerror: " + str(OldMaxError)
        #ne = NewSpringEnergy/100000. + NewMaxError
        #print "Newtotal: " + str(ne)
        ##print "\n"
        
        #Gibbs sampling
        #if OldSpringError/100. + OldMaxError < NewSpringError/100. + NewMaxError:
        myrand = np.random.rand()
        ##print "rand is: " + str(myrand) + " and boltzmann weight is " + str(np.exp(-(NewSpringEnergy - OldSpringEnergy)/temperature))
        
        if NewSpringEnergy > OldSpringEnergy:
            if myrand > np.exp(-(NewSpringEnergy - OldSpringEnergy)/temperature):
                ##print "Rejecting proposal"
                self.beads[bead]=oldweight
        

        
        return True
        
        

test = WeightString(synapses[0],synapses[1],5,1,1)
    

#Simple function to evaluate network
def EvalNet(net, X):

    layer_1 = 1/(1+np.exp(-(np.dot(X,net[0]))))
    layer_2 = 1/(1+np.exp(-(np.dot(layer_1,net[1]))))
    #layer_3 = 1/(1+np.exp(-(np.dot(layer_2,net[2]))))
    
    # how much did we miss the target value?
    #layer_3_error = layer_3 - y

    return layer_2


def BeadError(X, y, bead):
    X= np.array(X)
    y=np.array(y)

    layer_1 = 1/(1+np.exp(-(np.dot(X,bead[0]))))
    layer_2 = 1/(1+np.exp(-(np.dot(layer_1,bead[1]))))
    #layer_3 = 1/(1+np.exp(-(np.dot(layer_2,bead[2]))))

    # how much did we miss the target value?
    layer_2_error = layer_2 - y
        
    return np.mean(np.abs(layer_2_error))



def InterpBeadError(X, y, bead1, bead2, write = False, name = "00"):
    '''X = []
    y = []
    for i in xrange(1000):
        a,b = generatecandidate3(.5,.25,.1)
        X.append(a)
        y.append(b)'''
        
    X= np.array(X)
    y=np.array(y)

    errors = []
    

    for tt in xrange(100):
        #Should make this architecture independent at some point
        t = tt/100.
        layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_interpolate(bead1[0],bead2[0],t)))))
        layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_interpolate(bead1[1],bead2[1],t)))))
        #layer_3 = 1/(1+np.exp(-(np.dot(layer_2,synapse_interpolate(bead1[2],bead2[2],t)))))

        # how much did we miss the target value?
        layer_2_error = layer_2 - y
        
        errors.append(np.mean(np.abs(layer_2_error)))

    if write == True:
        with open("f" + str(name) + ".out",'w+') as f:
            for e in errors:
                f.write(str(e) + "\n")
    
    return max(errors)

    
results = []

connecteddict = {}
for i1 in xrange(len(synapses)):
    connecteddict[i1] = 'not connected'


for i1 in xrange(len(synapses)):
    #print i1
    for i2 in xrange(len(synapses)):
        
        if i2 > i1 and ((connecteddict[i1] != connecteddict[i2]) or (connecteddict[i1] == 'not connected' or connecteddict[i2] == 'not connected')) :

            test = WeightString(synapses[i1],synapses[i2],1,1,1)

            training_threshold = thresh

            depth = 0
            d_max = 10

            #Check error between beads
            #Alg: for each bead at depth i, SGD until converged.
            #For beads with max error along path too large, add another bead between them, repeat


            while (depth < d_max):
                X, y = GenTest(X,y)
                counter = 0

                for i,c in enumerate(test.ConvergedList):
                    if c == False:
                        error = BeadError(X, y, test.AllBeads[i])
                        #print error
                        while error > .5 * training_threshold and counter < 40000:
                            counter += 1
                            error = test.SGDBead(i, X, y)
                            #if counter%5000==0:
                            #    print counter
                            #    print error
                        test.ConvergedList[i] = True

                #print test.ConvergedList

                interperrors = []
                for b in xrange(len(test.AllBeads)-1):
                    e = InterpBeadError(X,y,test.AllBeads[b],test.AllBeads[b+1])
                    interperrors.append(e)
                #print interperrors

                if max(interperrors) < training_threshold:
                    depth = 2*d_max
                    #print test.ConvergedList
                    #print test.SpringNorm(2)
                    #print "Done!"

                else:

                    #Interperrors stores the maximum error on the path between beads
                    #shift index to account for added beads
                    shift = 0
                    for i, ie in enumerate(interperrors):
                        if ie > training_threshold:

                            beaddata = []
                            for k in xrange(len(test.w1)):
                                beaddata.append(synapse_interpolate(test.AllBeads[i+shift][k],test.AllBeads[i+shift+1][k], .5))
                            test.AllBeads.insert(i+shift+1,beaddata)
                            test.ConvergedList.insert(i+shift+1, False)
                            shift+=1
                            #print test.ConvergedList
                            #print test.SpringNorm(2)


                    #print d_max
                    depth += 1
            if depth == 2*d_max:
                results.append([i1,i2,test.SpringNorm(2),"Connected"])
                if connecteddict[i1] == 'not connected' and connecteddict[i2] == 'not connected':
                    connecteddict[i1] = i1
                    connecteddict[i2] = i1

                if connecteddict[i1] == 'not connected':
                    connecteddict[i1] = connecteddict[i2]
                else:
                    if connecteddict[i2] == 'not connected':
                        connecteddict[i2] = connecteddict[i1]
                    else:
                        if connecteddict[i1] != 'not connected' and connecteddict[i2] != 'not connected':
                            hold = connecteddict[i2]
                            connecteddict[i2] = connecteddict[i1]
                            for h in xrange(len(synapses)):
                                if connecteddict[h] == hold:
                                    connecteddict[h] = connecteddict[i1]
                    
            else:
                results.append([i1,i2,test.SpringNorm(2),"Disconnected"])
            #print results[-1]
	
	
	

uniquecomps = []
totalcomps = 0
for i in xrange(len(synapses)):
    if not (connecteddict[i] in uniquecomps):
        uniquecomps.append(connecteddict[i])
    
    if connecteddict[i] == 'not connected':
        totalcomps += 1
        
    #print i,connecteddict[i]

notconoffset = 0

if 'not connected' in uniquecomps:
    notconoffset = -1
    
print "Thresh: " + str(thresh)
print "Comps: " + str(len(uniquecomps) + notconoffset + totalcomps)



#for i in xrange(len(synapses)):
#    print connecteddict[i]

connsum = []
for r in results:
    if r[3] == "Connected":
        connsum.append(r[2])
        #print r[2]
        
print "***"
print np.average(connsum)
print np.std(connsum)
print "Hidden dim: " + str(hidden_dim)
