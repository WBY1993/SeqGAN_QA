# -*- coding: utf-8 -*-
import pickle
import jieba
import Pinyin2Hanzi


class Preprocess():
    def __init__(self, config):
        self.vocab = {}
        self.vocab_file = config.vocab_file
        self.NUM_ID = config.NUM_ID
        self.UNK_ID = config.UNK_ID

    def build_vocab(self, file_list, vocab_size=6000):
        vocab_dict = {}
        for file_name in file_list:
            with open(file_name, "r", encoding="utf-8") as file:
                line_num = 0
                for line in file:
                    line_num += 1
                    if line_num % 10000 == 0:
                        print("file_name:%s line:%d" % (file_name, line_num))
                    line = line.strip()
                    for word in jieba.cut(line):
                        if Pinyin2Hanzi.is_chinese(word):
                            if word not in vocab_dict.keys():
                                vocab_dict[word] = 1
                            else:
                                vocab_dict[word] += 1
                        elif word in [" ", ":", ";", "\"", "'", "[", "]", "{", "}", ",", ".", "/", "?", "~", "!", "@", "#", "$",
                                      "%", "^", "&", "*", "(", ")", "-", "_", "+", "="]:
                            if word not in vocab_dict.keys():
                                vocab_dict[word] = 1
                            else:
                                vocab_dict[word] += 1
                        elif word in [" ", "：", "；", "“", "”", "‘", "「", "」", "{", "}", "，", "。", "/", "？", "～", "！", "@", "#", "￥",
                                      "%", "…", "&", "×", "（", "）", "-", "—", "+", "="]:
                            if word not in vocab_dict.keys():
                                vocab_dict[word] = 1
                            else:
                                vocab_dict[word] += 1
        
        vocab_list = sorted(vocab_dict.items(), key=lambda x:x[1], reverse=True)
        if len(vocab_list) > vocab_size:
            vocab_list = vocab_list[:vocab_size]
        vocab2id = {}
        id2vocab = {}
        for i,v in enumerate(vocab_list):
            vocab2id[v[0]] = i + 5
            id2vocab[i+5] = v[0]
        self.vocab = {"vocab2id": vocab2id,"id2vocab": id2vocab}
        pickle.dump(self.vocab, open(self.vocab_file, "wb"))
        
    def load_vocab(self):
        self.vocab = pickle.load(open(self.vocab_file, "rb"))
        
    def convert_qa(self, query_file, answer_file, data_file):
        query_len = []
        answer_len = []
        file_q = open(query_file, "r", encoding="utf-8")
        file_a = open(answer_file, "r", encoding="utf-8")
        line_num = 0
        with open(data_file, "w", encoding="utf-8") as f:
            while True:
                line_num += 1
                if line_num % 10000 == 0:
                    print(line_num)
                query = file_q.readline()
                answer = file_a.readline()
                if not query or not answer:
                    break
                query = query.strip()
                answer = answer.strip()
                if not query or not answer:
                    continue
                
                query_id = []
                answer_id = []
                for word in jieba.cut(query):
                    try:
                        query_id.append(self.vocab["vocab2id"][word])
                    except KeyError:
                        if word.replace(".", "", 1).isdigit():
                            query_id.append(self.NUM_ID)
                        else:
                            query_id.append(self.UNK_ID)
                for word in jieba.cut(answer):
                    try:
                        answer_id.append(self.vocab["vocab2id"][word])
                    except KeyError:
                        if word.replace(".", "", 1).isdigit():
                            answer_id.append(self.NUM_ID)
                        else:
                            answer_id.append(self.UNK_ID)
                            
                query_len.append(len(query_id))
                answer_len.append(len(answer_id))
                print([query_id, answer_id], file=f)
        file_q.close()
        file_a.close()
        print("Query length: %d--%d" % (min(query_len), max(query_len)))
        print("Answer length: %d--%d" % (min(answer_len), max(answer_len)))
