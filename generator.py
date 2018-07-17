# -*- coding: utf-8 -*-
import tensorflow as tf


class Generator():
    def __init__(self, config):
        # initial
        self.batch_size = config.gen_batch_size
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.GO_ID = config.GO_ID
        self.EOS_ID = config.EOS_ID
        
        # build input
        self.build_input()
        
        # build graph
        self.build_pretrain_network()
        self.build_sample_network()
        self.build_adversarial_network()
        self.build_rollout_network()
        
    def build_input(self):
        with tf.variable_scope("generator_input"):
            self.encoder_input_data = tf.placeholder(tf.int32, shape=[None, None], name="encoder_input_data")
            self.encoder_seq_len = tf.reduce_sum(tf.sign(self.encoder_input_data), axis=1)
            self.decoder_input_data = tf.placeholder(tf.int32, shape=[None, None], name="decoder_input_data")
            self.decoder_input_label = tf.placeholder(tf.int32, shape=[None, None], name="decoder_input_label")
            self.decoder_sample_data = tf.placeholder(tf.int32, shape=[None, None], name="decoder_sample_data")
            self.decoder_sample_label = tf.placeholder(tf.int32, shape=[None, None], name="decoder_sample_label")
            self.decoder_seq_len = tf.reduce_sum(tf.sign(self.decoder_input_data), axis=1)
            self.decoder_max_len = tf.reduce_max(self.decoder_seq_len)
            self.target_weight = tf.cast(tf.sequence_mask(self.decoder_seq_len, self.decoder_max_len), dtype=tf.float32)
            self.rewards = tf.placeholder(tf.float32, shape=[None, None], name="rewards")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            self.forcing_rate = tf.placeholder(tf.float32, name="forcing_rate")
        
    def build_pretrain_network(self):
        with tf.variable_scope("generator"):
            with tf.variable_scope("embedding"):
                embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embedding_size], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
                encoder_input_data_emb = tf.nn.embedding_lookup(embedding, self.encoder_input_data)
            with tf.variable_scope("encoder"):
                en_lstm1 = tf.contrib.rnn.LSTMCell(256)
                en_lstm1 = tf.contrib.rnn.DropoutWrapper(en_lstm1, output_keep_prob=self.keep_prob)
                en_lstm2 = tf.contrib.rnn.LSTMCell(256)
                en_lstm2 = tf.contrib.rnn.DropoutWrapper(en_lstm2, output_keep_prob=self.keep_prob)
                encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([en_lstm1])
                encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([en_lstm2])
                bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(encoder_cell_fw,
                                                                                       encoder_cell_bw,
                                                                                       encoder_input_data_emb,
                                                                                       sequence_length=self.encoder_seq_len,
                                                                                       dtype=tf.float32) # 2 * batch_size * seq_len * hidden_dim
                encoder_outputs = tf.concat(bi_encoder_outputs, -1)
                encoder_state = []
                for layer_id in range(1):
                    # layer num
                    encoder_state.append(bi_encoder_state[0][layer_id]) # forward
                    encoder_state.append(bi_encoder_state[1][layer_id]) # backward
                encoder_state = tuple(encoder_state)
            with tf.variable_scope("decoder"):
                with tf.variable_scope("attention"):
                    de_lstm1 = tf.contrib.rnn.LSTMCell(256)
                    de_lstm1 = tf.contrib.rnn.DropoutWrapper(de_lstm1, output_keep_prob=self.keep_prob)
                    de_lstm2 = tf.contrib.rnn.LSTMCell(256)
                    de_lstm2 = tf.contrib.rnn.DropoutWrapper(de_lstm2, output_keep_prob=self.keep_prob)
                    decoder_cell = tf.contrib.rnn.MultiRNNCell([de_lstm1, de_lstm2])
                    
                    attention_mechanism = tf.contrib.seq2seq.LuongAttention(256, encoder_outputs, self.encoder_seq_len)
                    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, 256)
                    decoder_initial_state = decoder_cell.zero_state(self.batch_size, dtype=tf.float32)
                    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
                with tf.variable_scope("output"):
                    weights = tf.get_variable("weights", shape=[256, self.vocab_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                    biases = tf.get_variable("baises", shape=[self.vocab_size], initializer=tf.constant_initializer(0.1))
                    
                    def cond(time, state, max_len, logits_list_pre):
                        return time < max_len
                    def body(time, state, max_len, logits_list_pre):
                        decoder_in = tf.nn.embedding_lookup(embedding, self.decoder_input_data[:, time]) # batch * embedding_size
                        output, state = decoder_cell(decoder_in, state) # batch * hidden_dim
                        logits = tf.nn.bias_add(tf.matmul(output, weights), biases) # batch * vocab_size
                        logits_list_pre = logits_list_pre.write(time, logits)
                        return time+1, state, max_len, logits_list_pre
                        
                    logits_list_pre = tf.TensorArray(dtype=tf.float32, size=self.decoder_max_len, name="logits_list_pre")
                    _, _, _, logits_list_pre = tf.while_loop(cond=cond, body=body, loop_vars=[0, decoder_initial_state, self.decoder_max_len, logits_list_pre])
                    logits_list_pre = logits_list_pre.stack() # seqs * batch * vocab_size
                    logits_list_pre = tf.transpose(logits_list_pre, perm=[1, 0, 2]) # batch * seqs * vocab_size
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.decoder_input_label, [-1]), logits=tf.reshape(logits_list_pre, [-1, self.vocab_size]))
                    #cross_entropy = -tf.reduce_sum(tf.one_hot(tf.reshape(self.decoder_input_label, [-1]), self.vocab_size) * tf.log(tf.clip_by_value(tf.nn.softmax(tf.reshape(logits_list_pre, [-1, self.vocab_size])), 1e-20, 1.0)), 1)
                    self.pretrained_loss = tf.reduce_mean(cross_entropy)
            self.pre_summary_op = tf.summary.scalar("gen_pretrain_loss", self.pretrained_loss)
                    
    def build_sample_network(self):
        with tf.variable_scope("generator", reuse=True):
            with tf.variable_scope("embedding"):
                embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embedding_size], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
                encoder_input_data_emb = tf.nn.embedding_lookup(embedding, self.encoder_input_data)
            with tf.variable_scope("encoder"):
                en_lstm1 = tf.contrib.rnn.LSTMCell(256)
                en_lstm1 = tf.contrib.rnn.DropoutWrapper(en_lstm1, output_keep_prob=self.keep_prob)
                en_lstm2 = tf.contrib.rnn.LSTMCell(256)
                en_lstm2 = tf.contrib.rnn.DropoutWrapper(en_lstm2, output_keep_prob=self.keep_prob)
                encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([en_lstm1])
                encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([en_lstm2])
                bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(encoder_cell_fw,
                                                                                       encoder_cell_bw,
                                                                                       encoder_input_data_emb,
                                                                                       sequence_length=self.encoder_seq_len,
                                                                                       dtype=tf.float32) # 2 * batch_size * seq_len * hidden_dim
                encoder_outputs = tf.concat(bi_encoder_outputs, -1)
                encoder_state = []
                for layer_id in range(1):
                    # layer num
                    encoder_state.append(bi_encoder_state[0][layer_id]) # forward
                    encoder_state.append(bi_encoder_state[1][layer_id]) # backward
                encoder_state = tuple(encoder_state)
            with tf.variable_scope("decoder"):
                with tf.variable_scope("attention"):
                    de_lstm1 = tf.contrib.rnn.LSTMCell(256)
                    de_lstm1 = tf.contrib.rnn.DropoutWrapper(de_lstm1, output_keep_prob=self.keep_prob)
                    de_lstm2 = tf.contrib.rnn.LSTMCell(256)
                    de_lstm2 = tf.contrib.rnn.DropoutWrapper(de_lstm2, output_keep_prob=self.keep_prob)
                    decoder_cell = tf.contrib.rnn.MultiRNNCell([de_lstm1, de_lstm2])
                    
                    attention_mechanism = tf.contrib.seq2seq.LuongAttention(256, encoder_outputs, self.encoder_seq_len)
                    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, 256)
                    decoder_initial_state = decoder_cell.zero_state(self.batch_size, dtype=tf.float32)
                    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
                with tf.variable_scope("output"):
                    weights = tf.get_variable("weights", shape=[256, self.vocab_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                    biases = tf.get_variable("baises", shape=[self.vocab_size], initializer=tf.constant_initializer(0.1))
                    
                    def cond(time, sample_word, state, max_len, sample_data, sample_label):
                        return time < max_len
                    def body(time, sample_word, state, max_len, sample_data, sample_label):
                        # teaching force by changing decoder input
                        decoder_in = tf.cond(tf.random_uniform(shape=[], minval=0, maxval=1)<self.forcing_rate, 
                                             lambda: self.decoder_input_data[:, time],
                                             lambda: sample_word)
                        decoder_in_emb = tf.nn.embedding_lookup(embedding, decoder_in) # batch * embedding_size
                        output, state = decoder_cell(decoder_in_emb, state) # batch * hidden_dim
                        logits = tf.nn.bias_add(tf.matmul(output, weights), biases) # batch * vocab_size
                        softmax = tf.nn.softmax(logits)
                        sample_word = tf.reshape(tf.multinomial(tf.log(softmax), 1), shape=[self.batch_size])
                        sample_word = tf.cast(sample_word, dtype=tf.int32)
                        sample_data = sample_data.write(time, decoder_in)
                        sample_label = sample_label.write(time, sample_word)
                        return time+1, sample_word, state, max_len, sample_data, sample_label
                    
                    sample_data = tf.TensorArray(dtype=tf.int32, size=self.decoder_max_len, name="sample_data")
                    sample_label = tf.TensorArray(dtype=tf.int32, size=self.decoder_max_len, name="sample_label")
                    _, _, _, _, sample_data, sample_label = tf.while_loop(cond=cond, body=body, loop_vars=[0, tf.constant([self.GO_ID]*self.batch_size, dtype=tf.int32), decoder_initial_state, self.decoder_max_len, sample_data, sample_label])
                    sample_data = sample_data.stack() # seqs * batch
                    self.sample_data = tf.transpose(sample_data, perm=[1, 0]) # batch * seqs
                    sample_label = sample_label.stack() # seqs * batch
                    self.sample_label = tf.transpose(sample_label, perm=[1, 0]) # batch * seqs

    def build_adversarial_network(self):
        with tf.variable_scope("generator", reuse=True):
            with tf.variable_scope("embedding"):
                embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embedding_size], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
                encoder_input_data_emb = tf.nn.embedding_lookup(embedding, self.encoder_input_data)
            with tf.variable_scope("encoder"):
                en_lstm1 = tf.contrib.rnn.LSTMCell(256)
                en_lstm1 = tf.contrib.rnn.DropoutWrapper(en_lstm1, output_keep_prob=self.keep_prob)
                en_lstm2 = tf.contrib.rnn.LSTMCell(256)
                en_lstm2 = tf.contrib.rnn.DropoutWrapper(en_lstm2, output_keep_prob=self.keep_prob)
                encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([en_lstm1])
                encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([en_lstm2])
                bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(encoder_cell_fw,
                                                                                       encoder_cell_bw,
                                                                                       encoder_input_data_emb,
                                                                                       sequence_length=self.encoder_seq_len,
                                                                                       dtype=tf.float32) # 2 * batch_size * seq_len * hidden_dim
                encoder_outputs = tf.concat(bi_encoder_outputs, -1)
                encoder_state = []
                for layer_id in range(1):
                    # layer num
                    encoder_state.append(bi_encoder_state[0][layer_id]) # forward
                    encoder_state.append(bi_encoder_state[1][layer_id]) # backward
                encoder_state = tuple(encoder_state)
            with tf.variable_scope("decoder"):
                with tf.variable_scope("attention"):
                    de_lstm1 = tf.contrib.rnn.LSTMCell(256)
                    de_lstm1 = tf.contrib.rnn.DropoutWrapper(de_lstm1, output_keep_prob=self.keep_prob)
                    de_lstm2 = tf.contrib.rnn.LSTMCell(256)
                    de_lstm2 = tf.contrib.rnn.DropoutWrapper(de_lstm2, output_keep_prob=self.keep_prob)
                    decoder_cell = tf.contrib.rnn.MultiRNNCell([de_lstm1, de_lstm2])
                    
                    attention_mechanism = tf.contrib.seq2seq.LuongAttention(256, encoder_outputs, self.encoder_seq_len)
                    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, 256)
                    decoder_initial_state = decoder_cell.zero_state(self.batch_size, dtype=tf.float32)
                    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
                with tf.variable_scope("output"):
                    weights = tf.get_variable("weights", shape=[256, self.vocab_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                    biases = tf.get_variable("baises", shape=[self.vocab_size], initializer=tf.constant_initializer(0.1))
                    
                    def cond(time, state, max_len, logits_list_adv):
                        return time < max_len
                    def body(time, state, max_len, logits_list_adv):
                        decoder_in_emb = tf.nn.embedding_lookup(embedding, self.decoder_sample_data[:, time]) # batch * embedding_size
                        output, state = decoder_cell(decoder_in_emb, state) # batch * hidden_dim
                        logits = tf.nn.bias_add(tf.matmul(output, weights), biases) # batch * vocab_size
                        logits_list_adv = logits_list_adv.write(time, logits)
                        return time+1, state, max_len, logits_list_adv
                        
                    logits_list_adv = tf.TensorArray(dtype=tf.float32, size=self.decoder_max_len, name="logits_list_adv")
                    _, _, _, logits_list_adv = tf.while_loop(cond=cond, body=body, loop_vars=[0, decoder_initial_state, self.decoder_max_len, logits_list_adv])
                    logits_list_adv = logits_list_adv.stack() # seqs * batch * vocab_size
                    logits_list_adv = tf.transpose(logits_list_adv, perm=[1, 0, 2]) # batch * seqs * vocab_size
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.decoder_sample_label, [-1]), logits=tf.reshape(logits_list_adv, [-1, self.vocab_size]))
                    #cross_entropy = -tf.reduce_sum(tf.one_hot(tf.reshape(self.decoder_sample_label, [-1]), self.vocab_size) * tf.log(tf.clip_by_value(tf.nn.softmax(tf.reshape(logits_list_adv, [-1, self.vocab_size])), 1e-20, 1.0)), 1)
                    self.adversarial_loss = tf.reduce_mean(cross_entropy * tf.reshape(self.rewards, [-1]))
            self.adv_summary_op = tf.summary.scalar("gen_adversarial_loss", self.adversarial_loss)
                    
    def build_rollout_network(self):
        with tf.variable_scope("generator", reuse=True):
            with tf.variable_scope("embedding"):
                embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embedding_size], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
                encoder_input_data_emb = tf.nn.embedding_lookup(embedding, self.encoder_input_data)
            with tf.variable_scope("encoder"):
                en_lstm1 = tf.contrib.rnn.LSTMCell(256)
                en_lstm1 = tf.contrib.rnn.DropoutWrapper(en_lstm1, output_keep_prob=self.keep_prob)
                en_lstm2 = tf.contrib.rnn.LSTMCell(256)
                en_lstm2 = tf.contrib.rnn.DropoutWrapper(en_lstm2, output_keep_prob=self.keep_prob)
                encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([en_lstm1])
                encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([en_lstm2])
                bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(encoder_cell_fw,
                                                                                       encoder_cell_bw,
                                                                                       encoder_input_data_emb,
                                                                                       sequence_length=self.encoder_seq_len,
                                                                                       dtype=tf.float32) # 2 * batch_size * seq_len * hidden_dim
                encoder_outputs = tf.concat(bi_encoder_outputs, -1)
                encoder_state = []
                for layer_id in range(1):
                    # layer num
                    encoder_state.append(bi_encoder_state[0][layer_id]) # forward
                    encoder_state.append(bi_encoder_state[1][layer_id]) # backward
                encoder_state = tuple(encoder_state)
            with tf.variable_scope("decoder"):
                with tf.variable_scope("attention"):
                    de_lstm1 = tf.contrib.rnn.LSTMCell(256)
                    de_lstm1 = tf.contrib.rnn.DropoutWrapper(de_lstm1, output_keep_prob=self.keep_prob)
                    de_lstm2 = tf.contrib.rnn.LSTMCell(256)
                    de_lstm2 = tf.contrib.rnn.DropoutWrapper(de_lstm2, output_keep_prob=self.keep_prob)
                    decoder_cell = tf.contrib.rnn.MultiRNNCell([de_lstm1, de_lstm2])
                    
                    attention_mechanism = tf.contrib.seq2seq.LuongAttention(256, encoder_outputs, self.encoder_seq_len)
                    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, 256)
                    decoder_initial_state = decoder_cell.zero_state(self.batch_size, dtype=tf.float32)
                    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
                with tf.variable_scope("output"):
                    weights = tf.get_variable("weights", shape=[256, self.vocab_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                    biases = tf.get_variable("baises", shape=[self.vocab_size], initializer=tf.constant_initializer(0.1))
                    
                    def cond(time, max_len, sample_rollout):
                        return time < max_len
                    def body(time, max_len, sample_rollout):
                        sample_left = self.decoder_sample_label[:, :time]
                        sample_right = tf.TensorArray(dtype=tf.int32, size=max_len-time, name="sample_right")
                        def cond1(step, state):
                            return step < time
                        def body1(step, state):
                            decoder_in = tf.nn.embedding_lookup(embedding, self.decoder_sample_data[:, step]) # batch * embedding_size
                            output, state = decoder_cell(decoder_in, state) # batch * hidden_dim
                            return step+1, state
                        _, state = tf.while_loop(cond=cond1, body=body1, loop_vars=[0, decoder_initial_state])
                        def cond2(step, decoder_in, state, sample_right):
                            return step < max_len
                        def body2(step, decoder_in, state, sample_right):
                            decoder_in = tf.nn.embedding_lookup(embedding, decoder_in) # batch * embedding_size
                            output, state = decoder_cell(decoder_in, state)
                            logits = tf.nn.bias_add(tf.matmul(output, weights), biases) # batch * vocab_size
                            softmax = tf.nn.softmax(logits)
                            sample_word = tf.reshape(tf.multinomial(tf.log(softmax), 1), shape=[self.batch_size])
                            sample_word = tf.cast(sample_word, dtype=tf.int32)
                            sample_right = sample_right.write(step-time, sample_word)
                            return step+1, sample_word, state, sample_right
                        _, _, _, sample_right = tf.while_loop(cond=cond2, body=body2, loop_vars=[time, self.decoder_sample_data[:, time], state, sample_right])
                        sample_right = sample_right.stack() # seqs-time * batch
                        sample_right = tf.transpose(sample_right, perm=[1, 0]) # batch * seqs-time
                        sample_rollout_step = tf.concat([sample_left, sample_right], axis=1) # batch * seqs
                        sample_rollout = sample_rollout.write(time-1, sample_rollout_step)
                        return time+1, max_len, sample_rollout
                        
                    sample_rollout = tf.TensorArray(dtype=tf.int32, size=self.decoder_max_len, name="sample_rollout")
                    _, _, sample_rollout = tf.while_loop(cond=cond, body=body, loop_vars=[1, self.decoder_max_len, sample_rollout])
                    sample_rollout = sample_rollout.write(self.decoder_max_len-1, self.decoder_sample_label)
                    self.sample_rollout = sample_rollout.stack() # seqs * batch * seqs
