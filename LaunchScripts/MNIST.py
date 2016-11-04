#This code implements Dynamic String Sampling for the MNIST digit recognition task.
#The convnet architecture is a modified version of the implementation at:
#https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist

#Specify the target failure percentage on the test set with the 'thresh' parameter.  So thresh=.01 would be 99% accuracy on test set.

#By C. Daniel Freeman, 2016


#Imports and model parameters

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


import copy


thresh = .00725

cost_thresh = 1.0

# Parameters
learning_rate = 0.0001
training_epochs = 15
batch_size = 100
display_step = 1


models = []


#Function definitions


def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def synapse_interpolate(synapse1, synapse2, t):
    return (synapse2-synapse1)*t + synapse1

def model_interpolate(w1,b1,w2,b2,t):
    
    m1w = w1
    m1b = b1
    m2w = w2 
    m2b = b2
    
    mwi = [synapse_interpolate(m1we,m2we,t) for m1we, m2we in zip(m1w,m2w)]
    mbi = [synapse_interpolate(m1be,m2be,t) for m1be, m2be in zip(m1b,m2b)]
    
    return mwi, mbi

def InterpBeadError(w1,b1, w2,b2, write = False, name = "00"):
    errors = []
        
    
    
    for tt in xrange(20):
        #print tt
        #accuracy = 0.
        t = tt/20.
        thiserror = 0

        weights, biases = model_interpolate(w1,b1,w2,b2, t)
        #interp_model = multilayer_perceptron(w=weights, b=biases)
        interp_model = convnet(w=weights, b=biases)

        with interp_model.g.as_default():

            x = tf.placeholder("float", [None, n_input])
            y = tf.placeholder("float", [None, n_classes])
            pred = interp_model.predict(x)
            init = tf.initialize_all_variables()


            with tf.Session() as sess:
                sess.run(init)
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print "Accuracy:", 1 - accuracy.eval({x: xdat, y: ydat, interp_model.keep_prob: 1.0})#,"\t",tt,weights[0][1][0],weights[0][1][1]
                thiserror = 1 - accuracy.eval({x: xdat, y: ydat, interp_model.keep_prob: 1.0})


        errors.append(thiserror)

    if write == True:
        with open("f" + str(name) + ".out",'w+') as f:
            for e in errors:
                f.write(str(e) + "\n")
    
    return max(errors), np.argmax(errors)
        

#Class definitions

class convnet():

    def __init__(self, w=0, b=0, ind='00'):


        self.index = ind

        learning_rate = .0001
        training_epochs = 15
        batch_size = 100
        display_step = 1

        # Network Parameters
        n_hidden_1 = 256 # 1st layer number of features
        n_hidden_2 = 256 # 2nd layer number of features
        n_input = 784 # Guess quadratic function
        n_classes = 10 # 
        self.g = tf.Graph()
        
        
        self.params = []
        
        with self.g.as_default():

            self.keep_prob = tf.placeholder(tf.float32)
        
            #Note that by default, weights and biases will be initialized to random normal dists
            if w==0:
                
                self.weights = {
                    'c1': tf.Variable(tf.truncated_normal([5,5,1,32],stddev=.1)),
                    'c2': tf.Variable(tf.truncated_normal([5,5,32,64],stddev=.1)),
                    'fc1': tf.Variable(tf.truncated_normal([7*7*64, 1024],stddev=.1)),
                    'out': tf.Variable(tf.truncated_normal([1024, 10],stddev=.1))
                }
                self.weightslist = [self.weights['c1'],self.weights['c2'],self.weights['fc1'],self.weights['out']]
                self.biases = {
                    'b1': tf.Variable(tf.constant(.1, shape=[32])),
                    'b2': tf.Variable(tf.constant(.1, shape=[64])),
                    'b3': tf.Variable(tf.constant(.1, shape=[1024])),
                    'out': tf.Variable(tf.constant(.1, shape=[10]))
                }
                self.biaseslist = [self.biases['b1'],self.biases['b2'],self.biases['b3'],self.biases['out']]
                
            else:
                
                self.weights = {
                    'c1': tf.Variable(w[0]),
                    'c2': tf.Variable(w[1]),
                    'fc1': tf.Variable(w[2]),
                    'out': tf.Variable(w[3])
                }
                self.weightslist = [self.weights['c1'],self.weights['c2'],self.weights['fc1'],self.weights['out']]
                self.biases = {
                    'b1': tf.Variable(b[0]),
                    'b2': tf.Variable(b[1]),
                    'b3': tf.Variable(b[2]),
                    'out': tf.Variable(b[3])
                }
                self.biaseslist = [self.biases['b1'],self.biases['b2'],self.biases['b3'],self.biases['out']]
            self.saver = tf.train.Saver()
    
    def predict(self, x):
        
        with self.g.as_default():

            x_image = tf.reshape(x, [-1,28,28,1])
            layer_1 = tf.nn.relu(tf.nn.conv2d(x_image, self.weights['c1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
            pool_1 = tf.nn.max_pool(layer_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            #layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
            #layer_1 = tf.nn.relu(layer_1)
            # Hidden layer with RELU activation
            layer_2 = tf.nn.relu(tf.nn.conv2d(pool_1, self.weights['c2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
            pool_2 = tf.nn.max_pool(layer_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

                #layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
            #layer_2 = tf.nn.relu(layer_2)
            # Output layer with linear activation
            
            pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64])
            fc_layer = tf.nn.relu(tf.matmul(pool_2_flat, self.weights['fc1']) + self.biases['b3'])
            
            
            fc_layer_dropout = tf.nn.dropout(fc_layer, self.keep_prob)

            out_layer = tf.nn.softmax(tf.matmul(fc_layer_dropout, self.weights['out']) + self.biases['out'])


            return out_layer
        
    def ReturnParamsAsList(self):
        
        with self.g.as_default():

            with tf.Session() as sess:
                # Restore variables from disk
                self.saver.restore(sess, "/home/dfreeman/PythonFun/tmp/model"+str(self.index)+".ckpt")                
                return sess.run(self.weightslist), sess.run(self.biaseslist)

        
        
class WeightString:
    
    def __init__(self, w1, b1, w2, b2, numbeads, threshold):
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2
        #self.w2, self.b2 = m2.params
        self.AllBeads = []

        self.threshold = threshold
        
        self.AllBeads.append([w1,b1])
        
        
        for n in xrange(numbeads):
            ws,bs = model_interpolate(w1,b1,w2,b2, (n + 1.)/(numbeads+1.))
            self.AllBeads.append([ws,bs])
            
        self.AllBeads.append([w2,b2])
        
        
        self.ConvergedList = [False for f in xrange(len(self.AllBeads))]
        self.ConvergedList[0] = True
        self.ConvergedList[-1] = True
    
    
    def SpringNorm(self, order):
        
        totalweights = 0.
        totalbiases = 0.
        totaltotal = 0.

        #Energy between mobile beads
        for i,b in enumerate(self.AllBeads):
            if i < len(self.AllBeads)-1:
                #print "Tallying energy between bead " + str(i) + " and bead " + str(i+1)
                subtotalw = 0.
                subtotalb = 0.
                #for j in xrange(len(b)):
                subtotalw += np.linalg.norm(np.subtract(flatten(self.AllBeads[i][0]),flatten(self.AllBeads[i+1][0])),ord=order)#/len(self.beads[0][j])
                #for j in xrange(len(b)):
                subtotalb += np.linalg.norm(np.subtract(flatten(self.AllBeads[i][1]),flatten(self.AllBeads[i+1][1])),ord=order)#/len(self.beads[0][j])
                totalweights+=subtotalw
                totalbiases+=subtotalb
                totaltotal+=subtotalw + subtotalb

        weightdist = np.linalg.norm(np.subtract(flatten(self.AllBeads[0][0]),flatten(self.AllBeads[-1][0])),ord=order)
        biasdist = np.linalg.norm(np.subtract(flatten(self.AllBeads[0][1]),flatten(self.AllBeads[-1][1])),ord=order)
        totaldist = np.linalg.norm(np.subtract(flatten(self.AllBeads[0]),flatten(self.AllBeads[-1])),ord=order)


        return [totalweights,totalbiases,totaltotal, weightdist, biasdist, totaldist]#/len(self.beads)
        
    
    
    def SGDBead(self, bead, thresh, maxindex):
        
        finalerror = 0.
        
        #thresh = .05

        # Parameters
        learning_rate = 0.0001
        training_epochs = 15
        batch_size = 100
        display_step = 1
        
        curWeights, curBiases = self.AllBeads[bead]
        #test_model = multilayer_perceptron(w=curWeights, b=curBiases)
        test_model = convnet(w=curWeights, b=curBiases)

        with test_model.g.as_default():

            x = tf.placeholder("float", [None, n_input])
            y = tf.placeholder("float", [None, n_classes])
            #keep_prob = tf.placeholder(tf.float32)
            pred = test_model.predict(x)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
            init = tf.initialize_all_variables()
            stopcond = True

            with tf.Session() as sess:
                sess.run(init)
                xtest = mnist.test.images
                ytest = mnist.test.labels
                
                thiserror = 0.
                j = 0
                while stopcond:
                    for epoch in range(training_epochs):
                        avg_cost = 0.
                        total_batch = int(mnist.train.num_examples/batch_size)
                        if (avg_cost > thresh or avg_cost == 0.) and stopcond:
                        # Loop over all batches
                            for i in range(total_batch):
                                batch_x, batch_y = mnist.train.next_batch(batch_size)
                                # Run optimization op (backprop) and cost op (to get loss value)
                                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                                              y: batch_y,
                                                                              test_model.keep_prob: 0.5})
                                # Compute average loss
                                avg_cost += c / total_batch
                            # Display logs per epoch step
                            #if epoch % display_step == 0:
                            #    print "Epoch:", '%04d' % (epoch+1), "cost=", \
                            #        "{:.9f}".format(avg_cost)
                            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                            # Calculate accuracy
                            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                            print "Accuracy:", accuracy.eval({x: xtest, y: ytest, test_model.keep_prob: 1.0})
                            thiserror = 1 - accuracy.eval({x: xtest, y: ytest, test_model.keep_prob: 1.0})
                            if thiserror < thresh:
                                stopcond = False
                    #print "Optimization Finished!"

                    # Test model
                    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                    # Calculate accuracy
                    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    #print "Accuracy:", accuracy.eval({x: xtest, y: ytest})

                    #if (j%5000) == 0:
                    #    print "Error after "+str(j)+" iterations:" + str(accuracy.eval({x: xtest, y: ytest}))

                    finalerror = 1 - accuracy.eval({x: xtest, y: ytest, test_model.keep_prob: 1.0})
                    
                    if finalerror < thresh or stopcond==False:# or j > maxindex:
                        #print "Changing stopcond!"
                        stopcond = False
                        #print "Final params:"
                        test_model.params = sess.run(test_model.weightslist), sess.run(test_model.biaseslist)
                        self.AllBeads[bead]=test_model.params
                        print "Final bead error: " + str(finalerror)
                        
                    j+=1

            return finalerror
        
                
#Model generation
#copy_model = multilayer_perceptron(ind=0)
copy_model = convnet(ind=0)

for ii in xrange(2):

    '''weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }'''

    # Construct model with different initial weights
    #test_model = multilayer_perceptron(ind=ii)
    test_model = convnet(ind=ii)


    #Construct model with same initial weights
    #test_model = copy.copy(copy_model)
    #test_model.index = ii
    
    
    
    
    #print test_model.weights
    

    
    models.append(test_model)
    with test_model.g.as_default():

        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])
        pred = test_model.predict(x)

        # Define loss and optimizer
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        #Was using RMSProp without the .1 * learning rate.  Was going too fast
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.initialize_all_variables()


        #remove the comment to get random initialization
        stopcond = True




        with tf.Session() as sess:
            sess.run(init)
            xtest = mnist.test.images
            ytest = mnist.test.labels
            while stopcond:
                #print 'epoch:' + str(e)
                #X = []
                #y = []
                j = 0
                # Training cycle
                for epoch in range(training_epochs):
                    avg_cost = 0.
                    total_batch = int(10000/batch_size)

                    if (avg_cost > thresh or avg_cost == 0.) and stopcond:
                    # Loop over all batches
                        for i in range(total_batch):
                            batch_x, batch_y = mnist.train.next_batch(batch_size)
                            # Run optimization op (backprop) and cost op (to get loss value)
                            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                                          y: batch_y,
                                                                          test_model.keep_prob: 0.5})
                            # Compute average loss
                            avg_cost += c / total_batch
                        # Display logs per epoch step
                        #if epoch % display_step == 0:
                        #    #print "Epoch:", '%04d' % (epoch+1), "cost=", \
                        #    #    "{:.9f}".format(avg_cost)
                        
                        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                        # Calculate accuracy
                        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                        #print "Accuracy:", accuracy.eval({x: xtest, y: ytest})
                        thiserror = 1 - accuracy.eval({x: xtest, y: ytest, test_model.keep_prob: 1.0})
                        print thiserror
                        if thiserror < thresh:
                            stopcond = False
                            
                print "Optimization Finished!"

                # Test model
                #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print "Accuracy:", accuracy.eval({x: xtest, y: ytest, test_model.keep_prob: 1.0})

                if (j%5000) == 0:
                    print "Error after "+str(j)+" iterations:" + str(accuracy.eval({x: xtest, y: ytest, test_model.keep_prob: 1.0}))

                if 1 - accuracy.eval({x: xtest, y: ytest, test_model.keep_prob: 1.0}) < thresh or stopcond == False:
                    #print "Changing stopcond!"
                    stopcond = False
                    print "Final params:"
                    test_model.params = sess.run(test_model.weightslist), sess.run(test_model.biaseslist)
                    #save_path = test_model.saver.save(sess,"/home/dfreeman/PythonFun/tmp/model" + str(ii) + ".ckpt")
                j+=1
    #remove the comment to get random initialization

    
    #synapses.append([synapse_0,synapse_1,synapse_2

                
#Connected components search


#Used for softening the training criteria.  There's some fuzz required due to the difference in 
#training error between test and training
thresh_multiplier = 1.02


results = []
tests = []

connecteddict = {}
for i1 in xrange(len(models)):
    connecteddict[i1] = 'not connected'


for i1 in xrange(len(models)):
    print i1
    for i2 in xrange(len(models)):
        
        if i2 > i1 and ((connecteddict[i1] != connecteddict[i2]) or (connecteddict[i1] == 'not connected' or connecteddict[i2] == 'not connected')) :
            #print "slow1?"
            #print i1,i2
            #print models[0]
            #print models[1]
            #print models[0].params
            #print models[1].params
            test = WeightString(models[i1].params[0],models[i1].params[1],models[i2].params[0],models[i2].params[1],1,1)

            training_threshold = thresh

            depth = 0
            d_max = 10

            #Check error between beads
            #Alg: for each bead at depth i, SGD until converged.
            #For beads with max error along path too large, add another bead between them, repeat

            
            #Keeps track of which indices to check the interpbeaderror between
            newindices = [0,1]
            
            while (depth < d_max):
                print newindices
                #print "slow2?"
                #X, y = GenTest(X,y)
                counter = 0

                for i,c in enumerate(test.ConvergedList):
                    if c == False:
                        #print "slow3?"
                        error = test.SGDBead(i, .98*training_threshold, 20)
                        #print "slow4?"
                            #if counter%5000==0:
                            #    print counter
                            #    print error
                        test.ConvergedList[i] = True

                print test.ConvergedList

                interperrors = []
                interp_bead_indices = []
                for b in xrange(len(test.AllBeads)-1):
                    if b in newindices:
                        e = InterpBeadError(test.AllBeads[b][0],test.AllBeads[b][1], test.AllBeads[b+1][0], test.AllBeads[b+1][1])

                        interperrors.append(e)
                        interp_bead_indices.append(b)
                print interperrors

                if max([ee[0] for ee in interperrors]) < thresh_multiplier*training_threshold:
                    depth = 2*d_max
                    #print test.ConvergedList
                    #print test.SpringNorm(2)
                    #print "Done!"

                else:
                    del newindices[:]
                    #Interperrors stores the maximum error on the path between beads
                    #shift index to account for added beads
                    shift = 0
                    for i, ie in enumerate(interperrors):
                        if ie[0] > thresh_multiplier*training_threshold:
                            k = interp_bead_indices[i]
                            
                            ws,bs = model_interpolate(test.AllBeads[k+shift][0],test.AllBeads[k+shift][1],\
                                                      test.AllBeads[k+shift+1][0],test.AllBeads[k+shift+1][1],\
                                                      ie[1]/20.)
                            
                            test.AllBeads.insert(k+shift+1,[ws,bs])
                            test.ConvergedList.insert(k+shift+1, False)
                            newindices.append(k+shift+1)
                            newindices.append(k+shift)
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
                            for h in xrange(len(models)):
                                if connecteddict[h] == hold:
                                    connecteddict[h] = connecteddict[i1]
                    
            else:
                results.append([i1,i2,test.SpringNorm(2),"Disconnected"])
            tests.append(test)
            #print results[-1]
        
        
        

uniquecomps = []
totalcomps = 0
for i in xrange(len(models)):
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
print len(tests[0].AllBeads)
print "\t".join([str(s) for s in connsum[0]])
#print np.average(connsum)
#print np.std(connsum)
        
        
