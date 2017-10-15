import tensorflow as tf
import numpy as np
from util import *

class model():
    def __init__(self, args):
        self.args = args
        self.input = tf.placeholder(dtype=tf.int32, (None, args.max_time_step), name="input")
        self.indices = tf.placeholder(dtype=tf.float32, (args.batch_size), name="indices")

        h_in = []
        self.weight = tf.get_variable("embedding_weight", shape=(args.vacab_size, args.embedding_size), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        for t_step in range(args.max_time_step):
            if t_step != 0:
                tf.get_variable_scope().reuse_variables()

            embedded = tf.nn.embedding_lookup(self.weight, self.input[:, t_step])
            h_in.append(embedded)

        h_in = tf.reduce_sum(tf.convert_to_tensor(h_in, dtype=tf.float32), axis=0)
        logit = tf.layers.dense(h_in, args.vocab_size)

        target = tf.one_hot(self.indices, args.vocab_size, 1.0, 0.0)
        print(targe.get_shape().as_list())

        self.loss = tf.nn.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logit = logit, labels = target))

    def train(self):
        optimizer = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.loss)
        
        yield_data_function = mk_train_func(self.args.batch_size, self.args.max_time_step, self.args.data_path, self.args.dict_path) 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())

            for itr, (input_, label_) in enumerate(yield_data_function()):
                loss_, _ = sess.run([self.loss, optimizer], feed_dict={self.input:input_, self.indices:label_})
                
                if itr % 50 == 0:
                    print(itr, ":  ", loss_)

                if itr % 200 == 0:
                    saver.save(sess, "saved/model.ckpt", itr)
                    print("-----save model-----")

                if itr == self.args.itr:
                    break
            
    def get_embedding_weight(self):
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            saver.restore(sess, "saved/model.ckpt")
            weight = sess.run(self.weight)
        return weight
