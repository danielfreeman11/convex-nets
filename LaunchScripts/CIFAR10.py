#Imports and model parameters

from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#Simple network: Given three integers a,b,c, [-100,100] chooses three random x-values, and evaluates
#the quadratic function a*x^2 + b*x + c at those values.

import copy
from datetime import datetime
import os.path
import time
import math
import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10_input
from tensorflow.models.image.cifar10 import cifar10


for num_run in xrange(1):    
	alpha,hidden_dim,hidden_dim2 = (.001,4,4)
	thresh = .95

	if num_run%4 == 0:
	    thresh = .8
	if num_run%4 == 1:
	    thresh = .6
	if num_run%4 == 2:
	    thresh = .4
	if num_run%4 == 3:
	    thresh = .35
	cost_thresh = 1.0

	# Parameters
	learning_rate = 0.001
	training_epochs = 15
	#batch_size = 100
	display_step = 1
	# Network Parameters
	n_hidden_1 = 256 # 1st layer number of features
	n_hidden_2 = 256 # 2nd layer number of features
	n_input = 784 # Guess quadratic function
	n_classes = 10 # 
	#synapses = []
	#from __future__ import print_function
	tf.logging.set_verbosity(tf.logging.FATAL)
	FLAGS = tf.app.flags.FLAGS
	# Basic model parameters.
	batch_size = 128
	data_dir = '/tmp/cifar10_data'
	use_fp16 = False
	train_dir= '/tmp/cifar10_train'
	max_steps=1000000
	num_examples=10000
	log_device_placement=False
	# Global constants describing the CIFAR-10 data set.
	IMAGE_SIZE = cifar10_input.IMAGE_SIZE
	NUM_CLASSES = cifar10_input.NUM_CLASSES
	NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
	# Constants describing the training process.
	MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
	NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
	LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
	INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
	# If a model is trained with multiple GPUs, prefix all Op names with tower_name
	# to differentiate the operations. Note that this prefix is removed from the
	# names of the summaries when visualizing a model.
	TOWER_NAME = 'tower'
	DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
	models = []
	#Testing starting in the same place
	#synapse0 = 2*np.random.random((1,hidden_dim)) - 1
	#synapse1 = 2*np.random.random((hidden_dim,hidden_dim2)) - 1
	#synapse2 = 2*np.random.random((hidden_dim2,1)) - 1
	#Function definitions

	def func(x,a,b,c):
		return x*x*a + x*b + c

	def flatten(x):
		result = []
		for el in x:
			if hasattr(el, "__iter__") and not isinstance(el, basestring):
				result.extend(flatten(el))
			else:
				result.append(el)
		return result


	def generatecandidate4(a,b,c,tot):
		
		candidate = [[np.random.random() for x in xrange(1)] for y in xrange(tot)]
		candidatesolutions = [[func(x[0],a,b,c)] for x in candidate]
		
		return (candidate, candidatesolutions)

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
		
		#xdat,ydat = generatecandidate4(.5, .25, .1, 1000)
		
		#xdat,ydat = mnist.train.next_batch(1000)
		
		#xdat = mnist.test.images
		#ydat = mnist.test.labels
		#xdat = np.array(xdat)
		#ydat = np.array(ydat)
		
		
		
		
		for tt in xrange(20):
			#print tt
			#accuracy = 0.
			t = tt/20.
			thiserror = 0

			#x0 = tf.placeholder("float", [None, n_input])
			#y0 = tf.placeholder("float", [None, n_classes])
			weights, biases = model_interpolate(w1,b1,w2,b2, t)
			#interp_model = multilayer_perceptron(w=weights, b=biases)
			interp_model = convnet(w=weights, b=biases)

			with interp_model.g.as_default():
				
				xdat, ydat = cifar10.inputs(eval_data='test')
				logit_test = interp_model.predict(xdat)
				top_k_op = tf.nn.in_top_k(logit_test, ydat, 1)
				pred = interp_model.predict(xdat)
				init = tf.initialize_all_variables()
				with tf.Session() as sess:
					sess.run(init)
					
					tf.train.start_queue_runners(sess=sess)
					
					num_iter = 20
					true_count = 0  # Counts the number of correct predictions.
					total_sample_count = num_iter * batch_size
					step = 0
					while step < num_iter:
						predictions = sess.run([top_k_op])
						true_count += np.sum(predictions)
						step += 1
					precision = true_count / total_sample_count
					print "Accuracy:", precision
					#,"\t",tt,weights[0][1][0],weights[0][1][1]
					thiserror = 1 - precision
					
			errors.append(thiserror)

		if write == True:
			with open("f" + str(name) + ".out",'w+') as f:
				for e in errors:
					f.write(str(e) + "\n")
		
		return max(errors), np.argmax(errors)
		

	def _activation_summary(x):
		"""Helper to create summaries for activations.
		Creates a summary that provides a histogram of activations.
		Creates a summary that measures the sparsity of activations.
		Args:
		x: Tensor
		Returns:
		nothing
		"""
		# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
		# session. This helps the clarity of presentation on tensorboard.
		tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
		tf.histogram_summary(tensor_name + '/activations', x)
		tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


	def _variable_on_cpu(name, shape, initializer):
		"""Helper to create a Variable stored on CPU memory.
		Args:
		name: name of the variable
		shape: list of ints
		initializer: initializer for Variable
		Returns:
		Variable Tensor
		"""
		with tf.device('/cpu:0'):
			dtype = tf.float16 if False else tf.float32
			var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
		return var


	def _variable_with_weight_decay(name, shape, stddev, wd):
		"""Helper to create an initialized Variable with weight decay.
		Note that the Variable is initialized with a truncated normal distribution.
		A weight decay is added only if one is specified.
		Args:
		name: name of the variable
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		wd: add L2Loss weight decay multiplied by this float. If None, weight
			decay is not added for this Variable.
		Returns:
		Variable Tensor
		"""
		dtype = tf.float16 if False else tf.float32
		var = _variable_on_cpu(
		  name,
		  shape,
		  tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
		if wd is not None:
			weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		return var


	def distorted_inputs():
		"""Construct distorted input for CIFAR training using the Reader ops.
		Returns:
		images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
		labels: Labels. 1D tensor of [batch_size] size.
		Raises:
		ValueError: If no data_dir
		"""
		if not FLAGS.data_dir:
			raise ValueError('Please supply a data_dir')
		data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
		images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
													  batch_size=FLAGS.batch_size)
		if False:
			images = tf.cast(images, tf.float16)
			labels = tf.cast(labels, tf.float16)
		return images, labels


	def inputs(eval_data):
		"""Construct input for CIFAR evaluation using the Reader ops.
		Args:
		eval_data: bool, indicating if one should use the train or eval data set.
		Returns:
		images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
		labels: Labels. 1D tensor of [batch_size] size.
		Raises:
		ValueError: If no data_dir
		"""
		if not FLAGS.data_dir:
			raise ValueError('Please supply a data_dir')
		data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
		images, labels = cifar10_input.inputs(eval_data=eval_data,
											data_dir=data_dir,
											batch_size=FLAGS.batch_size)
		if False:
			images = tf.cast(images, tf.float16)
			labels = tf.cast(labels, tf.float16)
		return images, labels
		
	#Class definitions

	class convnet():

		def __init__(self, w=0, b=0, ind='00'):


			self.index = ind

			learning_rate = .001
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
				#Note that by default, weights and biases will be initialized to random normal dists
				if w==0:
					
					self.weights = {
						'c1': _variable_with_weight_decay('c1',shape=[5, 5, 3, 64],stddev=5e-2,wd=0.0),
						'c2': _variable_with_weight_decay('c2',shape=[5, 5, 64, 64],stddev=5e-2,wd=0.0),
						'fc1': _variable_with_weight_decay('fc1', shape=[2304, 384],stddev=0.04, wd=0.004),
						'fc2': _variable_with_weight_decay('fc2', shape=[384, 192],stddev=0.04, wd=0.004),
						'out': _variable_with_weight_decay('out', [192, NUM_CLASSES],stddev=1/192.0, wd=0.0)
					}
					self.weightslist = [self.weights['c1'],self.weights['c2'],self.weights['fc1'],self.weights['fc2'],self.weights['out']]
					self.biases = {
						'b1':  _variable_on_cpu('b1', [64], tf.constant_initializer(0.0)),
						'b2':  _variable_on_cpu('b2', [64], tf.constant_initializer(0.1)),
						'b3':  _variable_on_cpu('b3', [384], tf.constant_initializer(0.1)),
						'b4': _variable_on_cpu('b4', [192], tf.constant_initializer(0.1)),
						'out': _variable_on_cpu('bo', [NUM_CLASSES],tf.constant_initializer(0.0))
					}
					self.biaseslist = [self.biases['b1'],self.biases['b2'],self.biases['b3'],self.biases['b4'],self.biases['out']]
					
				else:
					
					self.weights = {
						'c1': tf.Variable(w[0]),
						'c2': tf.Variable(w[1]),
						'fc1': tf.Variable(w[2]),
						'fc2': tf.Variable(w[3]),
						'out': tf.Variable(w[4])
					}
					self.weightslist = [self.weights['c1'],self.weights['c2'],self.weights['fc1'],self.weights['fc2'],self.weights['out']]
					self.biases = {
						'b1': tf.Variable(b[0]),
						'b2': tf.Variable(b[1]),
						'b3': tf.Variable(b[2]),
						'b4': tf.Variable(b[3]),
						'out': tf.Variable(b[4])
					}
					self.biaseslist = [self.biases['b1'],self.biases['b2'],self.biases['b3'],self.biases['b4'],self.biases['out']]
				self.saver = tf.train.Saver()
		
		def predict(self, x):
			
			with self.g.as_default():

				
				layer_1 = tf.nn.conv2d(x, self.weights['c1'], [1, 1, 1, 1], padding='SAME')
				layer_1 = tf.nn.bias_add(layer_1, self.biases['b1'])
				layer_1 = tf.nn.relu(layer_1, name='layer_1')
				#_activation_summary(layer_1)
				pool_1 = tf.nn.max_pool(layer_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')
				norm_1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
				
				layer_2 = tf.nn.conv2d(norm_1, self.weights['c2'], [1, 1, 1, 1], padding='SAME')
				layer_2 = tf.nn.bias_add(layer_2, self.biases['b2'])
				layer_2 = tf.nn.relu(layer_2, name='layer_2')
				#_activation_summary(layer_2)
				norm_2 = tf.nn.lrn(layer_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
				pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool2')
				
				
				reshape = tf.reshape(pool_2, [FLAGS.batch_size, -1])
				layer_3 = tf.nn.relu(tf.matmul(reshape, self.weights['fc1']) + self.biases['b3'], name='fc1')
				#_activation_summary(layer_3)
				
				layer_4 = tf.nn.relu(tf.matmul(layer_3, self.weights['fc2']) + self.biases['b4'], name='fc2')
				#_activation_summary(layer_4)
				
				out_layer = tf.add(tf.matmul(layer_4, self.weights['out']),  self.biases['out'], name='out')
				#_activation_summary(out)
				return out_layer
			
		def ReturnParamsAsList(self):
			
			with self.g.as_default():

				with tf.Session() as sess:
					# Restore variables from disk
					self.saver.restore(sess, "/home/dfreeman/PythonFun/tmp/model"+str(self.index)+".ckpt")                
					return sess.run(self.weightslist), sess.run(self.biaseslist)




	class multilayer_perceptron():
		
		#weights = {}
		#biases = {}
		
		def __init__(self, w=0, b=0, ind='00'):
			
			self.index = ind #used for reading values from file
			#See the filesystem convention below (is this really necessary?)
			#I'm going to eschew writing to file for now because I'll be generating too many files
			#Currently, the last value of the parameters is stored in self.params to be read
			
			learning_rate = 0.001
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
			
				#Note that by default, weights and biases will be initialized to random normal dists
				if w==0:
					
					self.weights = {
						'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
						'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
						'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
					}
					self.weightslist = [self.weights['h1'],self.weights['h2'],self.weights['out']]
					self.biases = {
						'b1': tf.Variable(tf.random_normal([n_hidden_1])),
						'b2': tf.Variable(tf.random_normal([n_hidden_2])),
						'out': tf.Variable(tf.random_normal([n_classes]))
					}
					self.biaseslist = [self.biases['b1'],self.biases['b2'],self.biases['out']]
					
				else:
					
					self.weights = {
						'h1': tf.Variable(w[0]),
						'h2': tf.Variable(w[1]),
						'out': tf.Variable(w[2])
					}
					self.weightslist = [self.weights['h1'],self.weights['h2'],self.weights['out']]
					self.biases = {
						'b1': tf.Variable(b[0]),
						'b2': tf.Variable(b[1]),
						'out': tf.Variable(b[2])
					}
					self.biaseslist = [self.biases['b1'],self.biases['b2'],self.biases['out']]
				self.saver = tf.train.Saver()
		
		
		def UpdateWeights(self, w, b):
			with self.g.as_default():
				self.weights = {
						'h1': tf.Variable(w[0]),
						'h2': tf.Variable(w[1]),
						'out': tf.Variable(w[2])
					}
				self.weightslist = [self.weights['h1'],self.weights['h2'],self.weights['out']]
				self.biases = {
					'b1': tf.Variable(b[0]),
					'b2': tf.Variable(b[1]),
					'out': tf.Variable(b[2])
				}
				self.biaseslist = [self.biases['b1'],self.biases['b2'],self.biases['out']]
				

			
		def predict(self, x):
			
			with self.g.as_default():
				layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
				layer_1 = tf.nn.relu(layer_1)
				# Hidden layer with RELU activation
				layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
				layer_2 = tf.nn.relu(layer_2)
				# Output layer with linear activation
				out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
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
			learning_rate = 0.001
			training_epochs = 15
			batch_size = 100
			display_step = 1
			
			curWeights, curBiases = self.AllBeads[bead]
			#test_model = multilayer_perceptron(w=curWeights, b=curBiases)
			test_model = convnet(w=curWeights, b=curBiases)

			
			with test_model.g.as_default():

				global_step = tf.Variable(0, trainable=False)

				# Get images and labels for CIFAR-10.
				images, labels = cifar10.distorted_inputs()
				test_images, test_labels = cifar10.inputs(eval_data='test')

				# Build a Graph that computes the logits predictions from the
				# inference model.
				logits = test_model.predict(images)
				logit_test = test_model.predict(test_images)

				# Calculate loss.
				loss = cifar10.loss(logits, labels)

				# Build a Graph that trains the model with one batch of examples and
				# updates the model parameters.
				train_op = cifar10.train(loss, global_step)


				top_k_op = tf.nn.in_top_k(logit_test, test_labels, 1)


				# Build an initialization operation to run below.
				init = tf.initialize_all_variables()

				# Start running operations on the Graph.
				#sess = tf.Session(config=tf.ConfigProto(
				#    log_device_placement=FLAGS.log_device_placement))

				with tf.Session(config=tf.ConfigProto(
					log_device_placement=False)) as sess:
					sess.run(init)

					tf.train.start_queue_runners(sess=sess)

					step = 0
					stopcond = True
					while step < max_steps and stopcond:


						start_time = time.time()
						_, loss_value = sess.run([train_op, loss])
						duration = time.time() - start_time

						assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

						if step % 10 == 0:
							num_examples_per_step = batch_size
							examples_per_sec = num_examples_per_step / duration
							sec_per_batch = float(duration)

							format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
									  'sec/batch)')
							print (format_str % (datetime.now(), step, loss_value,
											 examples_per_sec, sec_per_batch))

						if step % 100 == 0:

							num_iter = int(math.ceil(num_examples / batch_size))
							true_count = 0  # Counts the number of correct predictions.
							total_sample_count = num_iter * batch_size
							stepp = 0
							while stepp < num_iter:
								predictions = sess.run([top_k_op])
								true_count += np.sum(predictions)
								stepp += 1


							# Compute precision @ 1.
							precision = true_count / total_sample_count
							print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

							if precision > 1 - thresh:
								stopcond = False
								test_model.params = sess.run(test_model.weightslist), sess.run(test_model.biaseslist)
								self.AllBeads[bead]=test_model.params
								finalerror = 1 - precision
								print ("Final bead error: ",str(finalerror))
								
						step += 1        
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
			
			global_step = tf.Variable(0, trainable=False)

			# Get images and labels for CIFAR-10.
			images, labels = cifar10.distorted_inputs()
			test_images, test_labels = cifar10.inputs(eval_data='test')

			# Build a Graph that computes the logits predictions from the
			# inference model.
			logits = test_model.predict(images)
			logit_test = test_model.predict(test_images)

			# Calculate loss.
			loss = cifar10.loss(logits, labels)

			# Build a Graph that trains the model with one batch of examples and
			# updates the model parameters.
			train_op = cifar10.train(loss, global_step)


			top_k_op = tf.nn.in_top_k(logit_test, test_labels, 1)


			# Build an initialization operation to run below.
			init = tf.initialize_all_variables()

			# Start running operations on the Graph.
			#sess = tf.Session(config=tf.ConfigProto(
			#    log_device_placement=FLAGS.log_device_placement))
			
			with tf.Session(config=tf.ConfigProto(
				log_device_placement=False)) as sess:
				sess.run(init)

				tf.train.start_queue_runners(sess=sess)

				step = 0
				stopcond = True
				while step < max_steps and stopcond:
					
					
					start_time = time.time()
					_, loss_value = sess.run([train_op, loss])
					duration = time.time() - start_time

					assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

					if step % 10 == 0:
						num_examples_per_step = batch_size
						examples_per_sec = num_examples_per_step / duration
						sec_per_batch = float(duration)

						format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
								  'sec/batch)')
						print (format_str % (datetime.now(), step, loss_value,
										 examples_per_sec, sec_per_batch))
						
					if step % 100 == 0:

						num_iter = int(math.ceil(num_examples / batch_size))
						true_count = 0  # Counts the number of correct predictions.
						total_sample_count = num_iter * batch_size
						stepp = 0
						while stepp < num_iter:
							predictions = sess.run([top_k_op])
							true_count += np.sum(predictions)
							stepp += 1


						# Compute precision @ 1.
						precision = true_count / total_sample_count
						print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
						
						if precision > 1 - thresh:
							stopcond = False
							test_model.params = sess.run(test_model.weightslist), sess.run(test_model.biaseslist)
					step += 1

					
	#Connected components search


	#Used for softening the training criteria.  There's some fuzz required due to the difference in 
	#training error between test and training
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
				#print "slow1?"
				#print i1,i2
				#print models[0]
				#print models[1]
				#print models[0].params
				#print models[1].params
				
				#test = WeightString(models[i1].params[0],models[i1].params[1],models[i2].params[0],models[i2].params[1],1,1)

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
	#with open('DSSCIFAR.' + str(thresh) + '.' +  str(num_run) + '.out','w+') as f: 
        print "Thresh: " + str(thresh) + "\n"
        print "Comps: " + str(len(uniquecomps) + notconoffset + totalcomps) + "\n"
        connsum = []
        for r in results:
            if r[3] == "Connected":
                connsum.append(r[2])
        #print r[2]
        print "***\n"
        print  str(len(test.AllBeads)) + "\n"
        print "\t".join([str(s) for s in connsum[0]])
        #print np.average(connsum)
        #print np.std(connsum)	
