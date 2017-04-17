#coding = utf-8
import tensorflow as tf
import numpy as np

class DataIterator(object):
    def __init__(self, datas, batch_size = 60, dtype = np.float64):
	if type(datas) == list:
		datas = np.array(datas, dtype = dtype)
        self.data_count = len(datas)
        self.batch_size = batch_size
        self.datas = datas
        self.batch_count = (self.data_count - 1)/batch_size + 1
        self.reset()

    def reset(self):
        self.next_batch_index = 0

    def has_next(self):
        return self.next_batch_index < self.batch_count

    def next(self):
        if not self.has_next():
            return None
        next_batch = self.datas[self.next_batch_index::self.batch_count]
        self.next_batch_index += 1
        return next_batch

def default_cost_listener(epoch, batch_index, cost_value):
    if epoch % 20 ==0 and batch_index % 10 == 0:
        print "epoch: {} batch: {} cost: {}".format(epoch, batch_index, cost_value)

class AutoEncoder(object):

    def __init__(self):
        pass

    # neuron_counts is the number of neuron for each hidden layer
    # the number of neuron of input layer and decoder is excluded
    def fit(self, 
            neuron_counts, 
            ite,
            learning_rate = 0.01,
            keep_prob = 0.95,
            max_epoch = 1000,
            cost_listener = default_cost_listener
            ):
        self.ws = neuron_counts 
        self.ite = ite
        self.input_dim = ite.next().shape[1]
        self.ws = [self.input_dim] + self.ws
        self.tf_vars = {}
        self.tf_vars["encoder"] = []
        self.tf_vars["decoder"] = []

        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.max_epoch = max_epoch
        self.cost_listener = cost_listener

        self.build()
        self.train()

    def init_parameters(self):
        for i in range(len(self.ws) - 1):
            W = tf.Variable(tf.truncated_normal([self.ws[i], self.ws[i + 1]]))
            b = tf.Variable(tf.truncated_normal([self.ws[i + 1]]))
            self.tf_vars["encoder"].append({"W": W, "b": b})

        for i in range(1, len(self.ws)):
            W = tf.Variable(tf.truncated_normal([self.ws[-i], self.ws[-(i+1)]]))
            b = tf.Variable(tf.truncated_normal([self.ws[-(i+1)]]))
            self.tf_vars["decoder"].append({"W": W, "b": b})


    def build_encoder(self, inputs):
        encoder = inputs
        for i, parameter in enumerate(self.tf_vars["encoder"]):
            W = parameter["W"]
            b = parameter["b"]
            encoder = tf.matmul(encoder, W) + b
            encoder = tf.nn.tanh(encoder)
        return encoder

    def build_decoder(self, inputs):
        decoder = inputs
        for i, parameter in enumerate(self.tf_vars["decoder"]):
            W = parameter["W"]
            b = parameter["b"]
            decoder = tf.matmul(decoder, W) + b
            if i < len(self.tf_vars["decoder"]) - 1:
                decoder = tf.nn.tanh(decoder)
            else:
                decoder = tf.nn.sigmoid(decoder)
        return decoder

    def build(self):
        self.init_parameters()
        self.inputs_holder = tf.placeholder(tf.float32, [None, self.ws[0]])
        
        self.keep_prob_holder = tf.placeholder(tf.float32)
        self.inputs_holder = tf.nn.dropout(self.inputs_holder, self.keep_prob_holder)
        self.encoder = self.build_encoder(self.inputs_holder)

        self.generator_inputs_holder = tf.placeholder(tf.float32, [None, self.ws[-1]])
        self.generator = self.build_decoder(self.generator_inputs_holder)
        self.decoder = self.build_decoder(self.encoder)
        self.cost = tf.reduce_mean(tf.pow(self.decoder - self.inputs_holder, 2))

    def encode(self, inputs):
        feed_dict = {self.inputs_holder: inputs, self.keep_prob_holder: 1.0}
        encoder_outputs = self.sess.run(self.encoder, feed_dict = feed_dict)
        return encoder_outputs

        
    def decode(self, inputs):
        feed_dict = {self.generator_inputs_holder: inputs}
        decoder_outputs = self.sess.run(self.generator, feed_dict = feed_dict)
        return decoder_outputs

    def output(self, inputs):
        feed_dict = {self.inputs_holder: inputs, self.keep_prob_holder: 1.0}
        outputs = self.sess.run(self.decoder, feed_dict = feed_dict)
        return outputs


    def train(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)
        for epoch in range(self.max_epoch):
            self.ite.reset()
            while self.ite.has_next():
                batch_index = self.ite.next_batch_index
                batch_inputs = self.ite.next()
                feed_dict = {self.inputs_holder: batch_inputs, self.keep_prob_holder: self.keep_prob}
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict = feed_dict)
                self.cost_listener(epoch, batch_index, c)
                    
    def close(self):
        self.sess.close()

