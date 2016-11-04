#This code implements Dynamic String Sampling for the PTB next word prediction task.
#The LSTM implementation is a modified version of the ''small'' configuration detailed at: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py

#Specify the model threshold with the 'thresh' parameter.

#By C. Daniel Freeman, 2016



#Imports and model parameters

import tensorflow as tf
import numpy as np

import copy
from datetime import datetime
import os.path
import time
import math


thresh = 135.


from __future__ import absolute_import
from __future__ import division

import gzip
import os
import re
import sys
import tarfile

import tensorflow as tf

#from tensorflow.models.rnn.ptb import reader
import reader #this is the same reader file as in tensorflow.models.rnn.ptb

flags = tf.flags
logging = tf.logging


for RNN_run in xrange(10):
    models = []

    data_path = "simple-examples/data"

    class SmallConfig(object):
        """Small config."""
        init_scale = 0.1
        learning_rate = 1.0
        max_grad_norm = 5
        num_layers = 2
        num_steps = 20
        hidden_size = 200
        max_epoch = 4
        max_max_epoch = 13
        keep_prob = 1.0
        lr_decay = 0.5
        batch_size = 20
        vocab_size = 10000



    def data_type():
        return tf.float32


    class PTBInput(object):
        """The input data."""

        def __init__(self, config, data, name=None):
            self.batch_size = batch_size= config.batch_size
            self.num_steps = num_steps = config.num_steps
            self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
            #print(data[0:10],batch_size,num_steps)
            self.input_data, self.targets = reader.ptb_producer(
                data, batch_size, num_steps)

    #Function definitions

    def get_config():
        if True:
            return SmallConfig()
        elif FLAGS.model == "medium":
            return MediumConfig()
        elif FLAGS.model == "large":
            return LargeConfig()
        elif FLAGS.model == "test":
            return TestConfig()
        else:
            raise ValueError("Invalid model: %s", FLAGS.model)

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
            t = tt/20.
            thiserror = 0

            weights, biases = model_interpolate(w1,b1,w2,b2, t)
            interp_model = PTBModel(config=config,w=weights, b=biases)

            with interp_model.g.as_default():


                with tf.name_scope("Test"):
                    test_input = PTBInput(config=config, data=test_data, name="TestInput")
                    with tf.variable_scope("Model", reuse=None):
                        inputs_for_testing = tf.nn.embedding_lookup(interp_model.weights['e'], test_input.input_data)
                        inputs_for_testing = [tf.squeeze(input_step, [1]) for input_step in tf.split(1, 20, inputs_for_testing)]
                        pred_test = interp_model.predict(inputs_for_testing)
                        loss_test = tf.nn.seq2seq.sequence_loss_by_example([pred_test],
                        [tf.reshape(test_input.targets, [-1])],
                        [tf.ones([interp_model.batch_size * interp_model.num_steps], dtype=data_type())])
                        cost_test = tf.reduce_sum(loss_test) / interp_model.batch_size

                init = tf.initialize_all_variables()

                sv = tf.train.Supervisor()
                with sv.managed_session() as session:
                    session.run(init)

                    test_perplexity = run_epoch(session, interp_model, test_input, cost_test, verbose=True)
                    print "Test perplexity: " + str(test_perplexity)
                    this_error= test_perplexity


            errors.append(thiserror)

        if write == True:
            with open("f" + str(name) + ".out",'w+') as f:
                for e in errors:
                    f.write(str(e) + "\n")

        return max(errors), np.argmax(errors)


    def InterpBeadErrorEval(w1,b1, w2,b2, tt):


        raw_data = reader.ptb_raw_data(data_path)
        train_data, _, test_data, _ = raw_data
        t = tt/20.
        thiserror = 0
        weights, biases = model_interpolate(w1,b1,w2,b2, t)
        interp_model = PTBModel(config=config,w=weights, b=biases)

        with interp_model.g.as_default():


            with tf.name_scope("Test"):
                test_input = PTBInput(config=config, data=test_data, name="TestInput")
                with tf.variable_scope("Interp", reuse=False):
                    inputs_for_testing = tf.nn.embedding_lookup(interp_model.weights['e'], test_input.input_data)
                    inputs_for_testing = [tf.squeeze(input_step, [1]) for input_step in tf.split(1, test_input.num_steps, inputs_for_testing)]
                    pred_test = interp_model.predict(inputs_for_testing)
                    loss_test = tf.nn.seq2seq.sequence_loss_by_example([pred_test],
                    [tf.reshape(test_input.targets, [-1])],
                    [tf.ones([test_input.batch_size * test_input.num_steps], dtype=data_type())])
                    cost_test = tf.reduce_sum(loss_test) / test_input.batch_size

            init = tf.initialize_all_variables()

            sv = tf.train.Supervisor()
            with sv.managed_session() as session:
                session.run(init)
                tv = tf.trainable_variables()

                test_perplexity = run_epoch(session, interp_model, test_input, cost_test, verbose=True)
                print "Test perplexity: " + str(test_perplexity)

    def run_epoch(session, model, inputs, cost, eval_op=None, verbose=False):
        """Runs the model on the given data."""
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = session.run(model.initial_state)

        fetches = {
        "cost": cost,
        "final_state": model.final_state,
        "initial_state": model.initial_state
        }
        if eval_op is not None:
            fetches["eval_op"] = eval_op

        print "Number of steps:"
        print inputs.epoch_size

        for step in range(inputs.epoch_size):
            #for step in range(200):
            feed_dict = {}
            for i, (c, h) in enumerate(model.initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            vals = session.run(fetches, feed_dict)
            this_cost = vals["cost"]
            state = vals["final_state"]
            init_state = vals["initial_state"]

            costs += this_cost
            iters += inputs.num_steps

            if verbose and step % (inputs.epoch_size // 10) == 10:
                print "%.3f perplexity: %.3f speed: %.0f wps" + \
                str(step * 1.0 / inputs.epoch_size) + " || " +  str(np.exp(costs / iters)) + " || " + \
                 str(iters * inputs.batch_size / (time.time() - start_time))

        return np.exp(costs / iters)


    class PTBModel():
        """The PTB model."""

        def __init__(self, config, w=0, b=0, is_training=True):

            self.g = tf.Graph()

            self.batch_size = 20
            self.num_steps = 20
            self.size = config.hidden_size
            self.vocab_size = config.vocab_size

            with self.g.as_default():

                #Note that by default, weights and biases will be initialized to random normal dists
                if w==0:

                    self.weights = {
                        'hw1': np.random.uniform(-.1,.1,size=(400,800)),
                        'hw2': np.random.uniform(-.1,.1,size=(400,800)),
                        'w1':  tf.Variable(tf.random_uniform([self.size, self.vocab_size],minval=-.1,maxval=.1),name="HiddenToOutW"),
                         'e':  tf.Variable(tf.random_uniform([10000, 200], minval=-.1, maxval=.1), name="embedding", dtype=data_type())               
                    }
                    self.weightslist = [self.weights['hw1'],self.weights['hw2'],self.weights['w1'],self.weights['e']]
                    self.biases = {
                        'hb1': np.full((800),0.),
                        'hb2': np.full((800),0.),
                        'b1': tf.Variable(tf.random_uniform([self.vocab_size],minval=-.1,maxval=.1),name="HiddenToOutB")
                    }
                    self.biaseslist = [self.biases['hb1'],self.biases['hb2'],self.biases['b1']]

                else:

                    self.weights = {
                        'hw1': w[0],
                        'hw2': w[1],
                        'w1' : tf.Variable(w[2]),
                        'e'  : tf.Variable(w[3])
                    }
                    self.weightslist = [self.weights['hw1'],self.weights['hw2'],self.weights['w1'],self.weights['e']]
                    self.biases = {
                        'hb1': b[0],
                        'hb2': b[1],
                        'b1' : tf.Variable(b[2])
                    }
                    self.biaseslist = [self.biases['hb1'],self.biases['hb2'],self.biases['b1']]
                self.saver = tf.train.Saver()

        def assign_lr(self, session, lr_value):
            session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


        @property
        def initial_state(self):
            return self._initial_state

        @property
        def test_state(self):
            return self._test_state

        @property
        def cost(self):
            return self._cost

        @property
        def final_state(self):
            return self._final_state

        @property
        def lr(self):
            return self._lr

        @property
        def train_op(self):
            return self._train_op

        def predict(self, inputs):

            with self.g.as_default():
                #note: I've used a slightly modified LSTMCell which I defined in the rnn_cell.py file within tensorflow.
                #The only change is that it explicitly reads in initializers for the hidden weights and biases
                #Either make this change yourself, or use the rnn_cell.py I've provided on the github.  Remember to delete the .pyc file!
                lstm_cell1 = tf.nn.rnn_cell.WritableLSTMCell(200, winitializer = tf.constant_initializer(self.weights['hw1']),\
                                                    binitializer = tf.constant_initializer(self.biases['hb1']),\
                                                    forget_bias=0.0, state_is_tuple=True)
                lstm_cell2 = tf.nn.rnn_cell.WritableLSTMCell(200, winitializer = tf.constant_initializer(self.weights['hw2']),\
                                                    binitializer = tf.constant_initializer(self.biases['hb2']),\
                                                    forget_bias=0.0, state_is_tuple=True)

                cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1,lstm_cell2], state_is_tuple=True)
                self._initial_state = cell.zero_state(self.batch_size, data_type())


                outputs = []
                state = self._initial_state

                outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
                output = tf.reshape(tf.concat(1, outputs), [-1, self.size])
                logits = tf.matmul(output, self.weights['w1']) + self.biases['b1']

                self._final_state = state

                return logits





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
            raw_data = reader.ptb_raw_data(data_path)
            train_data, _, test_data, _ = raw_data

            curWeights, curBiases = self.AllBeads[bead]
            test_model = PTBModel(config=config, w=curWeights, b=curBiases)


            with test_model.g.as_default():

                with tf.name_scope("Train"):
                    train_input = PTBInput(config=config, data=train_data, name="TrainInput")
                    with tf.variable_scope("Model", reuse=None):

                        inputs_for_training = tf.nn.embedding_lookup(test_model.weights['e'], train_input.input_data)

                        if True and config.keep_prob < 1: #True is a standin for is_training
                            inputs_for_training = tf.nn.dropout(inputs_for_training, config.keep_prob)

                        inputs_for_training = [tf.squeeze(input_step, [1]) for input_step in tf.split(1, 20, inputs_for_training)]        
                        pred = test_model.predict(inputs_for_training)
                        loss = tf.nn.seq2seq.sequence_loss_by_example([pred],
                        [tf.reshape(train_input.targets, [-1])],
                        [tf.ones([test_model.batch_size * test_model.num_steps], dtype=data_type())])
                        cost = tf.reduce_sum(loss) / test_model.batch_size


                        test_model._lr = tf.Variable(0.0, trainable=False)
                        tvars = tf.trainable_variables()
                        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                                  config.max_grad_norm)
                        optimizer = tf.train.GradientDescentOptimizer(test_model._lr)
                        train_op = optimizer.apply_gradients(
                        zip(grads, tvars),
                        global_step=tf.contrib.framework.get_or_create_global_step())

                        test_model._new_lr = tf.placeholder(
                          tf.float32, shape=[], name="new_learning_rate")
                        test_model._lr_update = tf.assign(test_model._lr, test_model._new_lr)


                        test_LSTM_weight = tf.trainable_variables()[-4]



                with tf.name_scope("Test"):
                    test_input = PTBInput(config=config, data=test_data, name="TestInput")
                    with tf.variable_scope("Model", reuse=True):



                        inputs_for_testing = tf.nn.embedding_lookup(test_model.weights['e'], test_input.input_data)
                        inputs_for_testing = [tf.squeeze(input_step, [1]) for input_step in tf.split(1, 20, inputs_for_testing)]
                        pred_test = test_model.predict(inputs_for_testing)
                        loss_test = tf.nn.seq2seq.sequence_loss_by_example([pred_test],
                        [tf.reshape(test_input.targets, [-1])],
                        [tf.ones([test_model.batch_size * test_model.num_steps], dtype=data_type())])
                        cost_test = tf.reduce_sum(loss_test) / test_model.batch_size


                init = tf.initialize_all_variables()

                sv = tf.train.Supervisor()
                with sv.managed_session() as session:
                    session.run(init)

                    stopcond = True
                    perplex_thres=10000.
                    i = 0
                    i_max = 100
                    while i < i_max and stopcond:

                        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                        test_model.assign_lr(session, config.learning_rate * lr_decay)

                        print "Epoch: %d Learning rate: %.3f" + str((i + 1, session.run(test_model.lr)))
                        train_perplexity = run_epoch(session, test_model, train_input,cost, eval_op=train_op, verbose=False)
                        print "Epoch: %d Train Perplexity: %.3f" + str((i + 1, train_perplexity))

                        test_perplexity = run_epoch(session, test_model, test_input, cost_test, verbose=False)
                        print "Test perplexity: " + str(test_perplexity)
                        i+=1


                        if test_perplexity < thresh:
                            stopcond = False
                            with tf.name_scope("Test"):
                                with tf.variable_scope("Model", reuse=True):
                                    tv = tf.trainable_variables()
                                    test_model.params = [session.run(tv[-4]),\
                                                     session.run(tv[-2]),\
                                                     session.run(tv[-7]),\
                                                     session.run(tv[-6])],\
                                                    [session.run(tv[-3]),\
                                                     session.run(tv[-1]),\
                                                     session.run(tv[-5])]
                            self.AllBeads[bead]=test_model.params


                            return test_perplexity

                                                        
    #A pair of models is generated in the below 'for' loop with perplexity close to 'thresh'.
    #By changing the xrange(2) to xrange(NUM) you'll generate more models, and the codeblock below will check pairwise connections between all models
    #in a semi-intelligent way.
        
    for ii in xrange(2):


        raw_data = reader.ptb_raw_data(data_path)
        train_data, valid_data, test_data, _ = raw_data

        config = get_config()
        eval_config = get_config()
        eval_config.batch_size = 1
        eval_config.num_steps = 1




        test_model = PTBModel(config = config)



        models.append(test_model)
        with test_model.g.as_default():

            with tf.name_scope("Train"):
                train_input = PTBInput(config=config, data=train_data, name="TrainInput")
                with tf.variable_scope("Model", reuse=None):
                    inputs_for_training = tf.nn.embedding_lookup(test_model.weights['e'], train_input.input_data)

                    if True and config.keep_prob < 1: #True is a standin for is_training
                        inputs_for_training = tf.nn.dropout(inputs_for_training, config.keep_prob)

                    inputs_for_training = [tf.squeeze(input_step, [1]) for input_step in tf.split(1, train_input.num_steps, inputs_for_training)]        
                    pred = test_model.predict(inputs_for_training)
                    loss = tf.nn.seq2seq.sequence_loss_by_example([pred],
                    [tf.reshape(train_input.targets, [-1])],
                    [tf.ones([train_input.batch_size * train_input.num_steps], dtype=data_type())])
                    cost = tf.reduce_sum(loss) / train_input.batch_size


                    test_model._lr = tf.Variable(0.0, trainable=False)
                    tvars = tf.trainable_variables()
                    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                              config.max_grad_norm)
                    optimizer = tf.train.GradientDescentOptimizer(test_model._lr)
                    train_op = optimizer.apply_gradients(
                    zip(grads, tvars),
                    global_step=tf.contrib.framework.get_or_create_global_step())

                    test_model._new_lr = tf.placeholder(
                      tf.float32, shape=[], name="new_learning_rate")
                    test_model._lr_update = tf.assign(test_model._lr, test_model._new_lr)

            with tf.name_scope("Test"):
                test_input = PTBInput(config=config, data=test_data, name="TestInput")
                with tf.variable_scope("Model", reuse=True):


                    inputs_for_testing = tf.nn.embedding_lookup(test_model.weights['e'], test_input.input_data)
                    inputs_for_testing = [tf.squeeze(input_step, [1]) for input_step in tf.split(1, test_input.num_steps, inputs_for_testing)]
                    pred_test = test_model.predict(inputs_for_testing)
                    loss_test = tf.nn.seq2seq.sequence_loss_by_example([pred_test],
                    [tf.reshape(test_input.targets, [-1])],
                    [tf.ones([test_input.batch_size * test_input.num_steps], dtype=data_type())])
                    cost_test = tf.reduce_sum(loss_test) / test_input.batch_size

            init = tf.initialize_all_variables()

            sv = tf.train.Supervisor()
            with sv.managed_session() as session:
                session.run(init)

                stopcond = True
                perplex_thres=10000.
                i = 0
                i_max = 100
                while i < i_max and stopcond:
                    lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                    test_model.assign_lr(session, config.learning_rate * lr_decay)

                    print "Epoch: %d Learning rate: %.3f" + str((i + 1, session.run(test_model.lr)))
                    train_perplexity = run_epoch(session, test_model, train_input,cost, eval_op=train_op, verbose=False)
                    print "Epoch: %d Train Perplexity: %.3f" + str((i + 1, train_perplexity))
                    test_perplexity = run_epoch(session, test_model, test_input, cost_test, verbose=False)
                    print "Test perplexity: " + str(test_perplexity)

                    i+=1
                    if test_perplexity < thresh:
                        stopcond = False

                        with tf.name_scope("Test"):
                            with tf.variable_scope("Model", reuse=True):


                                tv = tf.trainable_variables()
                                test_model.params = [session.run(tv[-4]),\
                                                     session.run(tv[-2]),\
                                                     session.run(tv[-7]),\
                                                     session.run(tv[-6])],\
                                                    [session.run(tv[-3]),\
                                                     session.run(tv[-1]),\
                                                     session.run(tv[-5])]



    #Connected components search
    #The below codeblock checks pairwise connectivity between the models trained above (for an arbitrary number of models).


    #Used for softening the training criteria.  There's some fuzz required due to the difference in 
    #training error between test and training.  The qualitative features in the numerics aren't heavily dependent on this fuzz-factor,
    #but for calculating real bonafide geometric properties, this should be enforced as 1.
    thresh_multiplier = 1.1


    results = []

    connecteddict = {}
    for i1 in xrange(len(models)):
        connecteddict[i1] = 'not connected'


    test = WeightString(models[0].params[0],models[0].params[1],models[1].params[0],models[1].params[1],1,1)

    for i1 in xrange(len(models)):
        print i1
        for i2 in xrange(len(models)):

            if i2 > i1 and ((connecteddict[i1] != connecteddict[i2]) or (connecteddict[i1] == 'not connected' or connecteddict[i2] == 'not connected')) :

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
                    counter = 0

                    for i,c in enumerate(test.ConvergedList):
                        if c == False:
                            error = test.SGDBead(i, .98*training_threshold, 20)
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

                    else:
                        del newindices[:]
                        #Interperrors stores the maximum error on the path between beads
                        #shift index to account for added beads
                        shift = 0
                        for i, ie in enumerate(interperrors):
                            if ie[0] > thresh_multiplier*training_threshold:
                                k = interp_bead_indices[i]
                                                                
                                #This is the step which adds new models to the string of connections.  Comment out the '.5' and use 'ie[1]/20. if you'd like
                                #to add models at the maximum of the interpolate loss
                                ws,bs = model_interpolate(test.AllBeads[k+shift][0],test.AllBeads[k+shift][1],\
                                                          test.AllBeads[k+shift+1][0],test.AllBeads[k+shift+1][1],\
                                                          .5)#ie[1]/20.)

                                test.AllBeads.insert(k+shift+1,[ws,bs])
                                test.ConvergedList.insert(k+shift+1, False)
                                newindices.append(k+shift+1)
                                newindices.append(k+shift)
                                shift+=1

                        depth += 1
                                
                #This if statement nest updates connectivities intelligently.  It basically just checks if m1 is connected to m2 and m2 is connected to m3, it enforces that m1 is also connected to m3.
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
    
    with open("DSSRNN." + str(RNN_run) + ".0."+str(thresh)+".out",'w+') as f:
        f.write("Thresh: " + str(thresh) + "\n")
        f.write("Comps: " + str(len(uniquecomps) + notconoffset + totalcomps) + "\n")


        connsum = []
        for r in results:
            if r[3]=="Connected":
                connsum.append(r[2])
        f.write( "***\n" )
        f.write( str(len(test.AllBeads)) + "\n")
        f.write("\t".join([str(s) for s in connsum[0]]))

