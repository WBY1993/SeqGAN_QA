# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import preprocess
import configuration
import dataloader
import discriminator
import generator
import util
import os


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.60
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
resume = True

def main(unused_argv):
    conf = configuration.Config()
#    ####################Preprocess#######################
#    pre_model = preprocess.Preprocess(conf)
#    pre_model.build_vocab(["data/query.txt", "data/answer.txt"])
#    pre_model.convert_qa("data/query.txt", "data/answer.txt", "data/data.tra")
#    #####################################################
    
    ####################Initialize#######################
    data_model = dataloader.Data_loader(conf)
    gen_model = generator.Generator(conf)
    dis_model = discriminator.Discriminator(conf)
    #####################################################
    
    ####################Build Graph#######################
    global_step = tf.Variable(0, trainable=False, name="global_step")
    with tf.variable_scope("gen_optimizer"):
        with tf.variable_scope("pretrain"):
            # build optimizer op for pre-train
            pre_optimizer_gen = tf.train.AdamOptimizer(conf.gen_learning_rate)
            pre_var_gen = [v for v in tf.trainable_variables() if "generator" in v.name]
            grad1, var1 = zip(*pre_optimizer_gen.compute_gradients(gen_model.pretrained_loss, var_list=pre_var_gen))
            grad1, _ = tf.clip_by_global_norm(grad1, conf.grad_clip)
            pre_trainop_gen = pre_optimizer_gen.apply_gradients(zip(grad1, var1), global_step)
        with tf.variable_scope("advtrain"):
            # build optimizer op for adv-train
            adv_optimizer_gen = tf.train.AdamOptimizer(conf.gen_learning_rate)
            adv_var_gen = [v for v in tf.trainable_variables() if "generator" in v.name]
            grad2, var2 = zip(*adv_optimizer_gen.compute_gradients(gen_model.adversarial_loss, var_list=adv_var_gen))
            grad2, _ = tf.clip_by_global_norm(grad2, conf.grad_clip)
            adv_trainop_gen = adv_optimizer_gen.apply_gradients(zip(grad2, var2), global_step)
        
    with tf.variable_scope("dis_optimizer"):
        optimizer_dis = tf.train.AdamOptimizer(conf.dis_learning_rate)
        var_dis = [v for v in tf.trainable_variables() if "discriminator" in v.name]
        grad3, var3 = zip(*optimizer_dis.compute_gradients(dis_model.loss, var_list=var_dis))
        grad3, _ = tf.clip_by_global_norm(grad3, conf.grad_clip)
        trainop_dis = optimizer_dis.apply_gradients(zip(grad3, var3), global_step)
    ######################################################
    
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(os.path.join(conf.save_dir, "tra"), sess.graph)
    ####################Resume#######################
    if resume:
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(conf.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            last_global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, last_global_step is %s' % last_global_step)
        else:
            print('No checkpoint file found')
    #################################################
    
    ####################Pretrain Generator#######################
    print("Start pre-training generator.....")
    for epoch in range(conf.gen_pretrain_epoch):
        for encoder_input_data, decoder_input_data, decoder_input_label in data_model.create_batches_gen("data/data.tra", conf.gen_batch_size, conf.shuffle_size):
            _, loss, summary_str = sess.run([pre_trainop_gen, gen_model.pretrained_loss, gen_model.pre_summary_op],
                                            feed_dict={gen_model.encoder_input_data: encoder_input_data,
                                                       gen_model.decoder_input_data: decoder_input_data,
                                                       gen_model.decoder_input_label:decoder_input_label,
                                                       gen_model.keep_prob: conf.keep_prob})
            step = global_step.eval(session=sess)
            if step % 1 == 0:
                summary_writer.add_summary(summary_str, global_step=step)
                print("#Gen_Pre# Epoch %d, Step %d, loss: %.4f" % (epoch, step, loss))
        saver.save(sess, os.path.join(conf.save_dir, "model.ckpt"), global_step=global_step)
    #############################################################
    
    ####################Pretrain Discriminator#######################
    print("Start pre-training discriminator.....")
    for time in range(conf.dis_pretrain_step):
        file_real = open("data/real.tra", "w", encoding="utf-8")
        file_fake = open("data/fake.tra", "w", encoding="utf-8")
        for encoder_input_data, decoder_input_data, decoder_input_label in data_model.create_batches_gen("data/data.tra", conf.gen_batch_size, conf.shuffle_size):
            decoder_sample_label = sess.run(gen_model.sample_label,
                                            feed_dict={gen_model.encoder_input_data: encoder_input_data,
                                                       gen_model.decoder_input_data: decoder_input_data,
                                                       gen_model.forcing_rate: conf.forcing_rate,
                                                       gen_model.keep_prob: 1.0})
            for i in range(len(encoder_input_data)):
                print([list(encoder_input_data[i]), list(decoder_input_label[i])], file=file_real)
                print([list(encoder_input_data[i]), list(decoder_sample_label[i])], file=file_fake)
        file_real.close()
        file_fake.close()
        for epoch in range(conf.dis_pretrain_epoch):
            for encoder_input_data, decoder_input_label, dis_label in data_model.create_batches_dis("data/real.tra", "data/fake.tra", conf.dis_batch_size):
                _, loss, acc, summary_str = sess.run([trainop_dis, dis_model.loss, dis_model.acc, dis_model.summary_op],
                                                     feed_dict={dis_model.encoder_input: encoder_input_data,
                                                                dis_model.decoder_input: decoder_input_label,
                                                                dis_model.label_input: dis_label,
                                                                dis_model.keep_prob:conf.keep_prob})
                step = global_step.eval(session=sess)
                if step % 1 == 0:
                    summary_writer.add_summary(summary_str, global_step=step)
                    print("#Dis_Pre# Times: %d, Epoch %d, Step %d, loss: %.4f, acc: %.4f" % (time, epoch, step, loss, acc))
            saver.save(sess, os.path.join(conf.save_dir, "model.ckpt"), global_step=global_step)
    ################################################################
    
    ####################Start adversarial training#######################
    print("Start adversarial training.....")
    for total_epoch in range(conf.total_epoch):
        for encoder_input_data, decoder_input_data, decoder_input_label in data_model.create_batches_gen("data/data.tra", conf.gen_batch_size, conf.shuffle_size):
            for gen_step in range(conf.gen_update_step):
                decoder_sample_data, decoder_sample_label = sess.run([gen_model.sample_data, gen_model.sample_label],
                                                                     feed_dict={gen_model.encoder_input_data: encoder_input_data,
                                                                                gen_model.decoder_input_data: decoder_input_data,
                                                                                gen_model.forcing_rate: conf.forcing_rate,
                                                                                gen_model.keep_prob: 1.0})
                decoder_sample_data_unforce = np.insert(decoder_sample_label[:, :-1], 0, [conf.GO_ID]*conf.gen_batch_size, axis=1)
                reward_rollout = []
                for rollout_num in range(conf.rollout_num):
                    sample_rollout = sess.run(gen_model.sample_rollout,
                                              feed_dict={gen_model.encoder_input_data: encoder_input_data,
                                                         gen_model.decoder_input_data: decoder_input_data,
                                                         gen_model.decoder_sample_data: decoder_sample_data_unforce,
                                                         gen_model.decoder_sample_label: decoder_sample_label,
                                                         gen_model.keep_prob: 1.0})
                    reward_tmp = []
                    for sample in sample_rollout:
                        reward_step = sess.run(dis_model.softmax,
                                               feed_dict={dis_model.encoder_input: encoder_input_data,
                                                          dis_model.decoder_input: sample,
                                                          dis_model.keep_prob: 1.0})
                        reward_step = reward_step[:, 1] # batch
                        reward_tmp.append(reward_step)
                    reward_tmp = np.transpose(reward_tmp, axes=[1, 0]) # batch * seq
                    reward_rollout.append(reward_tmp) # rollout_num * batch * seq
                rewards = np.mean(reward_rollout, axis=0)
                rewards = rewards - 0.35
                _, loss, summary_str = sess.run([adv_trainop_gen, gen_model.adversarial_loss, gen_model.adv_summary_op],
                                                feed_dict={gen_model.encoder_input_data: encoder_input_data,
                                                           gen_model.decoder_input_data: decoder_input_data,
                                                           gen_model.decoder_sample_data: decoder_sample_data,
                                                           gen_model.decoder_sample_label: decoder_sample_label,
                                                           gen_model.rewards: rewards,
                                                           gen_model.keep_prob: 1.0})
                step = global_step.eval(session=sess)
                if step % 1 == 0:
                    summary_writer.add_summary(summary_str, global_step=step)
                    print("#Gen_Adv# Total_Epoch %d, Step %d, loss: %.4f" % (total_epoch, step, loss))
                
                rewards = np.ones_like(decoder_input_label, dtype=np.float32)
                _, loss, summary_str = sess.run([adv_trainop_gen, gen_model.adversarial_loss, gen_model.adv_summary_op],
                                                feed_dict={gen_model.encoder_input_data: encoder_input_data,
                                                           gen_model.decoder_input_data:decoder_input_data,
                                                           gen_model.decoder_sample_data:decoder_input_data,
                                                           gen_model.decoder_sample_label: decoder_input_label,
                                                           gen_model.rewards: rewards,
                                                           gen_model.keep_prob: 1.0})
                if step % 1 == 0:
                    summary_writer.add_summary(summary_str, global_step=step)
                    print("#Gen_Tea# Total_Epoch %d, Step %d, loss: %.4f" % (total_epoch, step, loss))
            
            for dis_step in range(conf.dis_update_step):
                file_real = open("data/real.tra", "w", encoding="utf-8")
                file_fake = open("data/fake.tra", "w", encoding="utf-8")
                for encoder_input_data, decoder_input_data, decoder_input_label in data_model.create_batches_gen("data/data.tra", conf.gen_batch_size, conf.shuffle_size):
                    decoder_sample_label = sess.run(gen_model.sample_label,
                                                    feed_dict={gen_model.encoder_input_data: encoder_input_data,
                                                               gen_model.decoder_input_data: decoder_input_data,
                                                               gen_model.forcing_rate: conf.forcing_rate,
                                                               gen_model.keep_prob: 1.0})
                    for i in range(len(encoder_input_data)):
                        print([list(encoder_input_data[i]), list(decoder_input_label[i])], file=file_real)
                        print([list(encoder_input_data[i]), list(decoder_sample_label[i])], file=file_fake)
                file_real.close()
                file_fake.close()
                for epoch in range(conf.dis_update_epoch):
                    for encoder_input, decoder_input, dis_label in data_model.create_batches_dis("data/real.tra", "data/fake.tra", conf.dis_batch_size):
                        _, loss, acc, summary_str = sess.run([trainop_dis, dis_model.loss, dis_model.acc, dis_model.summary_op],
                                                             feed_dict={dis_model.encoder_input: encoder_input,
                                                                        dis_model.decoder_input: decoder_input,
                                                                        dis_model.label_input: dis_label,
                                                                        dis_model.keep_prob:conf.keep_prob})
                        step = global_step.eval(session=sess)
                        if step % 10 == 0:
                            summary_writer.add_summary(summary_str, global_step=step)
                            print("#Dis_Adv# Total_epoch: %d, Times: %d, Epoch %d, Step %d, loss: %.4f, acc: %.4f" % (total_epoch, dis_step, epoch, step, loss, acc))
        saver.save(sess, os.path.join(conf.save_dir, "model.ckpt"), global_step=global_step)
    ###################Start adversarial training#######################    
    sess.close()
    

if __name__=="__main__":
    tf.app.run()
