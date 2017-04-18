#coding = utf-8
import tensorflow as tf
import numpy as np

class DataIterator(object):
    def __init__(self, datas, labels = None, batch_size = 60, dtype = np.float64):
	if type(datas) == list:
	    datas = np.array(datas, dtype = dtype)
        if labels != None:
            if type(labels) == list:
                labels = np.array(labels)
            labels = labels.astype(np.int32)

        self.data_count = len(datas)
        if labels !=None and len(labels.shape) == 1:
            label_type_count = labels.max() + 1
            label_matrix = np.zeros((self.data_count, label_type_count))
            label_matrix[range(self.data_count), labels] = 1
            labels = label_matrix

        self.batch_size = batch_size
        self.datas = datas
        self.batch_count = (self.data_count - 1)/batch_size + 1
        self.labels = labels
        self.reset()

    def reset(self):
        self.next_batch_index = 0

    def has_next(self):
        return self.next_batch_index < self.batch_count

    def next(self):
        if not self.has_next():
            return None
        next_data_batch = self.datas[self.next_batch_index::self.batch_count]
        if self.labels == None:
            result = [next_data_batch, None]
        else:
            next_label_batch = self.labels[self.next_batch_index::self.batch_count]
            result = [next_data_batch, next_label_batch]
        self.next_batch_index += 1
        return result


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
            fine_tuning = False,
            learning_rate = 0.01,
            keep_prob = 1.0,
            max_epoch = 1000,
            fine_tuning_learning_rate = 0.01,
            fine_tuning_max_epoch = 1000,
            cost_listener = default_cost_listener
            ):
        self.ws = neuron_counts 
        self.ite = ite

        ite.reset()
        peek_batch = ite.next()
        self.input_dim = peek_batch[0].shape[1]
        if fine_tuning:
            self.label_type_count = peek_batch[1].shape[1]
        ite.reset()



        self.ws = [self.input_dim] + self.ws
        self.tf_vars = {}
        self.tf_vars["encoder"] = []
        self.tf_vars["decoder"] = []

        self.keep_prob = keep_prob
        self.cost_listener = cost_listener

        self.build(fine_tuning)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

        if fine_tuning:
            fine_tune_outputs_holder = tf.placeholder(tf.float32, [None, self.label_type_count])
            fine_tune_cost = tf.reduce_mean(-tf.reduce_sum(fine_tune_outputs_holder * tf.log(self.fine_tune_outputs), reduction_indices=1))
            fine_tune_optimizer = tf.train.AdamOptimizer(learning_rate = fine_tuning_learning_rate).minimize(fine_tune_cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)
        print "\nstart training autoencoder ============\n"
        for epoch in range(max_epoch):
            self.ite.reset()
            while self.ite.has_next():
                batch_index = self.ite.next_batch_index
                batch_inputs, _ = self.ite.next()
                feed_dict = {self.inputs_holder: batch_inputs, self.keep_prob_holder: self.keep_prob}
                _, c = self.sess.run([optimizer, self.cost], feed_dict = feed_dict)
                self.cost_listener(epoch, batch_index, c)
        if fine_tuning:
            print "\nstart fine tuning ============\n"
            for epoch in range(fine_tuning_max_epoch):
                self.ite.reset()
                while self.ite.has_next():
                    batch_index = self.ite.next_batch_index
                    batch_inputs, batch_labels = self.ite.next()
                    feed_dict = {
                            self.inputs_holder: batch_inputs,
                            fine_tune_outputs_holder: batch_labels,
                            self.keep_prob_holder: self.keep_prob
                            }
                    _, c = self.sess.run([fine_tune_optimizer, fine_tune_cost], feed_dict = feed_dict)
                    self.cost_listener(epoch, batch_index, c)

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



    def build(self, fine_tuning):
        self.init_parameters()
        self.inputs_holder = tf.placeholder(tf.float32, [None, self.ws[0]])
        
        self.keep_prob_holder = tf.placeholder(tf.float32)
        self.inputs_holder = tf.nn.dropout(self.inputs_holder, self.keep_prob_holder)
        self.encoder = self.build_encoder(self.inputs_holder)

        self.generator_inputs_holder = tf.placeholder(tf.float32, [None, self.ws[-1]])
        self.generator = self.build_decoder(self.generator_inputs_holder)
        self.decoder = self.build_decoder(self.encoder)
        self.cost = tf.reduce_mean(tf.pow(self.decoder - self.inputs_holder, 2))

        if fine_tuning:
            self.fine_tune_outputs = self.build_fine_tuning()

    def encode(self, inputs):
        feed_dict = {self.inputs_holder: inputs, self.keep_prob_holder: 1.0}
        encoder_outputs = self.sess.run(self.encoder, feed_dict = feed_dict)
        return encoder_outputs

        
    def decode(self, inputs):
        feed_dict = {self.generator_inputs_holder: inputs}
        decoder_outputs = self.sess.run(self.generator, feed_dict = feed_dict)
        return decoder_outputs

    def predict(self, inputs):
        feed_dict = {self.inputs_holder: inputs, self.keep_prob_holder: 1.0}
        predicted_outputs = self.sess.run(self.fine_tune_outputs, feed_dict = feed_dict)
        return predicted_outputs 

    def reconstruct(self, inputs):
        feed_dict = {self.inputs_holder: inputs, self.keep_prob_holder: 1.0}
        reconstruct_outputs = self.sess.run(self.decoder, feed_dict = feed_dict)
        return reconstruct_outputs

    def build_fine_tuning(self):
        self.tf_vars["output"] = {}
        W = tf.Variable(tf.truncated_normal([self.ws[-1], self.label_type_count]), name="W")
        b = tf.Variable(tf.truncated_normal([self.label_type_count]))
        self.tf_vars["output"]["W"] = W
        self.tf_vars["output"]["b"] = b
        fine_tune_outputs = tf.matmul(self.encoder, W) + b
        fine_tune_outputs = tf.nn.softmax(fine_tune_outputs)
        return fine_tune_outputs


        
        

                    
    def close(self):
        self.sess.close()

