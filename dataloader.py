# -*- coding: utf-8 -*-
import numpy as np


class Data_loader():
    def __init__(self, config):
        self.PAD_ID = config.PAD_ID
        self.GO_ID = config.GO_ID
        self.EOS_ID = config.EOS_ID
        self.NUM_ID = config.NUM_ID
        self.UNK_ID = config.UNK_ID
        
    def padding_gen(self, encoder_input, decoder_input):
        input_max_len = max([len(i) for i in encoder_input])
        output_max_len = max([len(i) for i in decoder_input])
        encoder_input_data = []
        decoder_input_data = []
        decoder_input_label = []
        for i in range(len(encoder_input)):
            encoder_input_data.append(np.array(encoder_input[i] + [self.PAD_ID] * (input_max_len-len(encoder_input[i])), dtype=np.int32))
            data = decoder_input[i] + [self.EOS_ID] + [self.PAD_ID] * (output_max_len-len(decoder_input[i]))
            decoder_input_data.append(np.array([self.GO_ID] + data[:-1], dtype=np.int32))
            decoder_input_label.append(np.array(data, dtype=np.int32))
        return encoder_input_data, decoder_input_data, decoder_input_label
    
    def create_batches_gen(self, data_file, batch_size, shuffle_size):
        with open(data_file, "r", encoding="utf-8") as file:
            text_list = []
            while True:
                data = file.readline()
                if data:
                    data = eval(data)
                    text_list.append(data)
                    if len(text_list)==shuffle_size:
                        np.random.shuffle(text_list)
                        encoder_input = []
                        decoder_input = []
                        for _ in range(batch_size):
                            sample = text_list.pop()
                            encoder_input.append(sample[0])
                            decoder_input.append(sample[1])
                        encoder_input_data, decoder_input_data, decoder_input_label = self.padding_gen(encoder_input, decoder_input)
                        yield encoder_input_data, decoder_input_data, decoder_input_label
                else:
                    np.random.shuffle(text_list)
                    while len(text_list) >= batch_size:
                        encoder_input = []
                        decoder_input = []
                        for _ in range(batch_size):
                            sample = text_list.pop()
                            encoder_input.append(sample[0])
                            decoder_input.append(sample[1])
                        encoder_input_data, decoder_input_data, decoder_input_label = self.padding_gen(encoder_input, decoder_input)
                        yield encoder_input_data, decoder_input_data, decoder_input_label
                    break
    
    def padding_dis(self, encoder_input, decoder_input, label_input):
        input_max_len = max([len(i) for i in encoder_input])
        output_max_len = max([len(i) for i in decoder_input])
        encoder_input_data = []
        decoder_input_label = []
        dis_label = []
        for i in range(len(encoder_input)):
            encoder_input_data.append(np.array(encoder_input[i] + [self.PAD_ID] * (input_max_len-len(encoder_input[i])), dtype=np.int32))
            decoder_input_label.append(np.array(decoder_input[i] + [self.PAD_ID] * (output_max_len-len(decoder_input[i])), dtype=np.int32))
            dis_label.append(label_input[i])
        return encoder_input_data, decoder_input_label, dis_label
                
    def create_batches_dis(self, real_data_file, fake_data_file, batch_size):
        file_real = open(real_data_file, "r", encoding="utf-8")
        file_fake = open(fake_data_file, "r", encoding="utf-8")
        text_list = []
        while True:
            real_data = file_real.readline()
            fake_data = file_fake.readline()
            if not real_data or not fake_data:
                break
            else:
                real_data = eval(real_data)
                real_data.append(1)
                text_list.append(real_data)
                
                fake_data = eval(fake_data)
                fake_data.append(0)
                text_list.append(fake_data)
                if len(text_list)>=batch_size:
                    np.random.shuffle(text_list)
                    encoder_input = []
                    decoder_input = []
                    label_input = []
                    for _ in range(batch_size):
                        sample = text_list.pop()
                        encoder_input.append(sample[0])
                        decoder_input.append(sample[1])
                        label_input.append(sample[2])
                    encoder_input_data, decoder_input_label, dis_label = self.padding_dis(encoder_input, decoder_input, label_input)
                    yield encoder_input_data, decoder_input_label, dis_label
        file_real.close()
        file_fake.close()
                
                
