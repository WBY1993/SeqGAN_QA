# -*- coding: utf-8 -*-
import tensorflow as tf


class Discriminator():
    def __init__(self, config):
        # initial
        self.batch_size = config.dis_batch_size
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.num_class = config.num_class
        self.l2_reg_lambda = config.l2_reg_lambda
        
        # build input
        self.build_input()
        
        # build network
        self.build_network()
        
    def build_input(self):
        with tf.variable_scope("discriminator_input"):
            self.encoder_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name="encoder_input")
            self.decoder_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name="decoder_input")
            self.encoder_seq_len = tf.reduce_sum(tf.sign(self.encoder_input), axis=1)
            self.decoder_seq_len = tf.reduce_sum(tf.sign(self.decoder_input), axis=1)
            self.label_input = tf.placeholder(dtype=tf.int32, shape=[None], name="label_input")
            self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
            
    def build_network(self):
        with tf.variable_scope("discriminator"):
            with tf.variable_scope("embedding"):
                embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embedding_size], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
                encoder_input_emb = tf.nn.embedding_lookup(embedding, self.encoder_input)
                decoder_input_emb = tf.nn.embedding_lookup(embedding, self.decoder_input)
            with tf.variable_scope("encoder"):
                en_lstm1 = tf.contrib.rnn.GRUCell(256)
                en_lstm1 = tf.contrib.rnn.DropoutWrapper(en_lstm1, output_keep_prob=self.keep_prob)
                en_lstm2 = tf.contrib.rnn.GRUCell(256)
                en_lstm2 = tf.contrib.rnn.DropoutWrapper(en_lstm2, output_keep_prob=self.keep_prob)
                encoder_cell = tf.contrib.rnn.MultiRNNCell([en_lstm1, en_lstm2])
                # outputs: batch * seqs * hidden_dim    state: num_layer * batch * hidden_dim
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_input_emb, sequence_length=self.encoder_seq_len, dtype=tf.float32)
                encoder_output = encoder_state[-1] # batch * hidden_dim
            with tf.variable_scope("decoder"):
                de_lstm1 = tf.contrib.rnn.GRUCell(256)
                de_lstm1 = tf.contrib.rnn.DropoutWrapper(de_lstm1, output_keep_prob=self.keep_prob)
                de_lstm2 = tf.contrib.rnn.GRUCell(256)
                de_lstm2 = tf.contrib.rnn.DropoutWrapper(de_lstm2, output_keep_prob=self.keep_prob)
                decoder_cell = tf.contrib.rnn.MultiRNNCell([de_lstm1, de_lstm2])
                # outputs: batch * seqs * hidden_dim    state: num_layer * batch * hidden_dim
                decoder_outputs, decoder_state = tf.nn.dynamic_rnn(decoder_cell, decoder_input_emb, sequence_length=self.decoder_seq_len, dtype=tf.float32)
                decoder_output = decoder_state[-1] # batch * hidden_dim
            with tf.variable_scope("output"):
                weights = tf.get_variable("weights", shape=[512, self.num_class], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                biases = tf.get_variable("baises", shape=[self.num_class], initializer=tf.constant_initializer(0.1))
                
                output = tf.concat([encoder_output, decoder_output], axis=1) # batch * hidden_dim*2
                logits = tf.nn.bias_add(tf.matmul(output, weights), biases) # batch * num_class
                self.softmax = tf.nn.softmax(logits)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_input, logits=logits)
                self.loss = tf.reduce_mean(cross_entropy)
                top_1 = tf.cast(tf.nn.in_top_k(logits, self.label_input, 1), tf.float32)
                self.acc = tf.reduce_mean(top_1)
            self.summary_op = tf.summary.merge([tf.summary.scalar("dis_loss", self.loss), tf.summary.scalar("dis_acc", self.acc)])
                
                
            