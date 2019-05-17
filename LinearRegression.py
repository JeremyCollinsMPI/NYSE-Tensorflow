import numpy as np
import tensorflow as tf
import time
from general import *
from OneHotEncoder import OneHotEncoder

class LinearRegression:
    steps = 1000000
    learn_rate = 0.03
    def __init__(self, target, dependent_variables, positive = True, include_intercept = False):
        self.target = np.array(target)
        self.dependent_variables = np.array(dependent_variables)
        self.positive = positive
        self.include_intercept = include_intercept
        self.one_hot_encodings = []	
        self.other_vectors = []

    def include_one_hot_encoding(self, one_hot_encoding):
        self.one_hot_encodings.append(one_hot_encoding)
    
    def create_linear_layers(self):
        self.x = tf.placeholder(tf.float32, [None, len(self.dependent_variables[0])], name = "x")
        self.W = tf.Variable(tf.zeros([len(self.dependent_variables[0]), 1]), name = "W")
        if self.include_intercept:
            self.b = tf.Variable(tf.zeros([1]), name = "b")
        self.product = tf.matmul(self.x, self.W)
        if self.include_intercept:
            self.y = self.product + self.b
        else:
            self.y = self.product
   
    def create_one_hot_layers(self):
        if len(self.one_hot_encodings) > 0:
            self.one_hot_encodings_placeholders = {}
            self.one_hot_encodings_encodings = {}
            self.one_hot_encodings_variables = {}
            for i in xrange(len(self.one_hot_encodings)):
                number_of_categories = self.one_hot_encodings[i].depth
                self.one_hot_encodings_placeholders[str(i)] = tf.placeholder(tf.int32, [None], name = "one_hot_placeholder" + str(i))
                self.one_hot_encodings_encodings[str(i)] = tf.one_hot(self.one_hot_encodings_placeholders[str(i)], number_of_categories)
                self.one_hot_encodings_variables[str(i)] = tf.Variable(tf.zeros([number_of_categories, 1], name = "one_hot_variable" + str(i)))  
        for i in xrange(len(self.one_hot_encodings)):
            print self.one_hot_encodings_encodings[str(i)]
            self.y = self.y + tf.matmul(self.one_hot_encodings_encodings[str(i)], self.one_hot_encodings_variables[str(i)])

    def create_other_vector_layers(self):
        if len(self.other_vectors) > 0:
            self.other_vectors_placeholders = {}
            self.other_vectors_variables = {}
            for i in xrange(len(self.other_vectors)):
                self.other_vectors_placeholders[str(i)] = tf.placeholder(tf.float32, [1, None], name = "other_vectors_placeholder" + str(i))
                self.other_vectors_variables[str(i)] = tf.Variable(tf.zeros([None, 1]), name = "other_vectors_variable" + str(i))
        for i in xrange(len(self.other_vectors)):
            self.y = self.y + tf.matmul(self.other_vectors_placeholders[str(i)], self.other_vectors_variables[str(i)])

    def train(self):      
        datapoint_size = len(self.target)
        batch_size = datapoint_size
        self.create_linear_layers()
        self.create_one_hot_layers()        
        self.y_ = tf.placeholder(tf.float32, [None, 1])
        self.cost = tf.reduce_mean(tf.square(self.y_ - self.y))
        self.cost_sum = tf.summary.scalar("cost", self.cost)
        self.train_step = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.cost)	
        self.clip_op = tf.assign(self.W, tf.clip_by_value(self.W, 0, np.infty))
        self.reduction_ops = {}
        for i in xrange(len(self.one_hot_encodings)):
            mean = tf.reduce_mean(self.one_hot_encodings_variables[str(i)])
            self.reduction_ops[str(i)] = tf.assign(self.one_hot_encodings_variables[str(i)], self.one_hot_encodings_variables[str(i)] - mean)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        for i in xrange(self.steps):
            print i
            if datapoint_size == batch_size:
                batch_start_idx = 0
            elif datapoint_size < batch_size:
                raise ValueError("datapoint_size: %d, must be greater than batch_size: %d" % (datapoint_size, batch_size))
            else:
                batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
            batch_end_idx = batch_start_idx + batch_size
            batch_x = self.dependent_variables[batch_start_idx:batch_end_idx]
            batch_y = self.target[batch_start_idx:batch_end_idx]
            feed = {self.x: batch_x, self.y_: batch_y}
            for j in xrange(len(self.one_hot_encodings)):
                to_feed = self.one_hot_encodings[j].encoding[batch_start_idx:batch_end_idx]
                feed[self.one_hot_encodings_placeholders[str(j)]] = to_feed
            sess.run(self.train_step, feed_dict = feed)
            if self.positive:
                sess.run(self.clip_op)
            for j in xrange(len(self.one_hot_encodings)):
            	sess.run(self.reduction_ops[str(j)])
            print("After %d iteration:" % i)
            print("W: %s" % sess.run(self.W))
            if self.include_intercept:
                print("b: %f" % sess.run(self.b))
            for j in xrange(len(self.one_hot_encodings)):
                print("one_hot_encodings_variable" + str(j) + " : %s" % sess.run(self.one_hot_encodings_variables[str(j)]))

class SequenceLinearRegression(LinearRegression):
	def __init__(self, sequences, window_size, offset = 0, positive = True, include_intercept = False, use_padding = True):
		self.window_size = window_size
		self.offset = offset
		self.sequences = sequences
		self.positive = positive
		self.include_intercept = include_intercept
		dependent_variables = []
		target_variable = []
		for sequence in self.sequences:
			normalised_sequence = []
			for n in range(1, len(sequence)):
				normalised_sequence.append([sequence[n] / sequence[n-1]])
				to_append = []
				for number in range(n - window_size, n - offset):
					if number < 0: 
						if use_padding == True:
							to_append.append(sequence[0])
					else:
						to_append.append(sequence[number])
				normaliser = to_append[-1]
				to_append = np.array(to_append)				
				to_append = to_append / normaliser
				dependent_variables.append(to_append)
			target_variable = target_variable + normalised_sequence
		self.target = np.array(target_variable)
		self.dependent_variables = np.array(dependent_variables)
		self.one_hot_encodings = []

class SequenceLinearRegressionIncludingTimestamps(SequenceLinearRegression):
    def include_timestamps(self, timestamps):    
        self.timestamps = timestamps
        self.calendar_months = [find_calendar_month(x) for x in timestamps]
        self.days_of_the_week = [find_day_of_the_week(x) for x in timestamps]
        calendar_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
        				'Oct', 'Nov', 'Dec']
        days_of_the_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        self.calendar_months_encoding = OneHotEncoder(self.calendar_months, calendar_months).encode()
        self.days_of_the_week_encoding = OneHotEncoder(self.days_of_the_week, days_of_the_week).encode()
        self.include_one_hot_encoding(self.calendar_months_encoding)
        self.include_one_hot_encoding(self.days_of_the_week_encoding)

class SequenceLinearRegressionIncludingConvolutionalNetwork(SequenceLinearRegression):
    def include_timestamps(self, timestamps):    
        self.timestamps = timestamps
        self.calendar_months = [find_calendar_month(x) for x in timestamps]
        self.days_of_the_week = [find_day_of_the_week(x) for x in timestamps]
        calendar_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
        				'Oct', 'Nov', 'Dec']
        days_of_the_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        self.calendar_months_encoding = OneHotEncoder(self.calendar_months, calendar_months).encode()
        self.days_of_the_week_encoding = OneHotEncoder(self.days_of_the_week, days_of_the_week).encode()
        self.include_one_hot_encoding(self.calendar_months_encoding)
        self.include_one_hot_encoding(self.days_of_the_week_encoding)

    def create_convolutional_network(self):
        self.cnn_input_layer = tf.placeholder(tf.float32, [len(self.dependent_variables), len(self.dependent_variables[0])])
        self.cnn_input_layer_reshaped = tf.reshape(self.cnn_input_layer, [-1, len(self.dependent_variables[0]), 1])
        self.cnn_conv1_filters = 5
        self.cnn_conv1 = tf.layers.conv1d(self.cnn_input_layer_reshaped, filters = self.cnn_conv1_filters, kernel_size = 2, padding = "same", activation = tf.nn.relu)
        self.cnn_last = self.cnn_conv1
        number_of_cnn_outputs = 1
        cnn_last_shape = tf.shape(self.cnn_last)
        cnn_output_length = len(self.dependent_variables[0]) * self.cnn_conv1_filters   
        self.cnn_last_flat = tf.reshape(self.cnn_last, [-1, cnn_output_length])
        self.cnn_output = tf.layers.dense(self.cnn_last_flat, units = number_of_cnn_outputs, activation = tf.nn.sigmoid)
        self.cnn_coefficients =  tf.Variable(tf.zeros([number_of_cnn_outputs, 1]))
        self.cnn_to_add = tf.matmul(self.cnn_output, self.cnn_coefficients)
        self.cnn_to_add = self.cnn_to_add - tf.reduce_mean(self.cnn_to_add)
        self.y = self.y + self.cnn_to_add 
	
    def train(self):      
        datapoint_size = len(self.target)
        batch_size = datapoint_size
        self.create_linear_layers()
        self.create_one_hot_layers()
        self.create_convolutional_network()        
        self.y_ = tf.placeholder(tf.float32, [None, 1])
        self.cost = tf.reduce_mean(tf.square(self.y_ - self.y))
        self.cost_sum = tf.summary.scalar("cost", self.cost)
        self.train_step = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.cost)	
        self.clip_op = tf.assign(self.W, tf.clip_by_value(self.W, 0, np.infty))
        self.reduction_ops = {}
        for i in xrange(len(self.one_hot_encodings)):
            mean = tf.reduce_mean(self.one_hot_encodings_variables[str(i)])
            self.reduction_ops[str(i)] = tf.assign(self.one_hot_encodings_variables[str(i)], self.one_hot_encodings_variables[str(i)] - mean)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        for i in xrange(self.steps):
            print i
            if datapoint_size == batch_size:
                batch_start_idx = 0
            elif datapoint_size < batch_size:
                raise ValueError("datapoint_size: %d, must be greater than batch_size: %d" % (datapoint_size, batch_size))
            else:
                batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
            batch_end_idx = batch_start_idx + batch_size
            batch_x = self.dependent_variables[batch_start_idx:batch_end_idx]
            batch_y = self.target[batch_start_idx:batch_end_idx]
            feed = {self.x: batch_x, self.y_: batch_y, self.cnn_input_layer: batch_x}
            for j in xrange(len(self.one_hot_encodings)):
                to_feed = self.one_hot_encodings[j].encoding[batch_start_idx:batch_end_idx]
                feed[self.one_hot_encodings_placeholders[str(j)]] = to_feed
            sess.run(self.train_step, feed_dict = feed)
            if self.positive:
                sess.run(self.clip_op)
            for j in xrange(len(self.one_hot_encodings)):
            	sess.run(self.reduction_ops[str(j)])
            print("After %d iterations:" % i)
            print("W: %s" % sess.run(self.W))
            if self.include_intercept:
                print("b: %f" % sess.run(self.b))
            for j in xrange(len(self.one_hot_encodings)):
                print("one_hot_encodings_variable" + str(j) + " : %s" % sess.run(self.one_hot_encodings_variables[str(j)]))
            print("Convolutional coefficient: %s" % sess.run(self.cnn_coefficients))




