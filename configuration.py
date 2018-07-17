# -*- coding: utf-8 -*-
class Config():
    def __init__(self):
        self.vocab_file = "./data/vocab.pkl"
        self.save_dir = ".\model"
        self.PAD_ID = 0
        self.GO_ID = 1
        self.EOS_ID = 2
        self.NUM_ID = 3
        self.UNK_ID = 4
        self.gen_batch_size = 2
        self.dis_batch_size = 4
        self.shuffle_size = 1000
        self.vocab_size = 699
        self.embedding_size = 32
        self.keep_prob = 0.75
        self.num_class = 2
        self.gen_learning_rate = 1e-2
        self.dis_learning_rate = 1e-4
        self.forcing_rate = 1.0
        self.rollout_num = 10 # Monte Carlo Search Num
        self.l2_reg_lambda = 0.2
        self.total_epoch = 50
        self.gen_pretrain_epoch = 30
        self.dis_pretrain_step = 20
        self.dis_pretrain_epoch = 3
        self.gen_update_step = 1
        self.dis_update_step = 5
        self.dis_update_epoch = 3
        self.grad_clip = 5.0
