# -*- coding: utf-8 -*-
import math
import os
import parser
import pickle
from tqdm import tqdm
import random
import copy
import numpy as np
import argparse
from hnswlib import Index
import torch
import torch.nn as nn
import gensim
from modules import Encoder, LayerNorm
from src.utils import cosine_similarity, get_user_seqs
from datasketch import MinHash, MinHashLSH


class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()

        # self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)

        # 替换成预训练后的embedding
        if args.data_name == 'ml-1m':
            item_embed = torch.from_numpy(
                np.load(args.data_dir + args.data_name + '_item_embedding.npz')['item_embed']).float()
            self.item_embeddings = nn.Embedding.from_pretrained(item_embed, padding_idx=0)
            self.item_embeddings.requires_grad = False  # 将 requires_grad 设置为 False

        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def transformer_encoder(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class OfflineItemSimilarity:
    def __init__(self, args, data_file=None, similarity_path=None, model_name='ItemCF', \
                 dataset_name='Sports_and_Outdoors'):
        """
        :param data_file: 原始的数据集txt文件
        :param similarity_path:
        :param model_name:
        :param dataset_name:
        """
        self.args = args
        self.dataset_name = dataset_name
        self.similarity_path = similarity_path
        # train_data_list used for item2vec, train_data_dict used for itemCF and itemCF-IUF
        self.train_data_list, self.train_item_list, self.train_data_dict = self._load_train_data(data_file)
        """
            train_data_list:
            [[],[],[]],每个[]为不含后3个item的序列
            
            train_data将每个用户喜欢的物品转化为一个列表
            [('1', '1', 1), ('1', '2', 1), ('1', '3', 1), ('1', '4', 1), ('1', '5', 1)]
            ()中第一个代表userId
            第二个代表itemId
            第三个每个都是1,表示喜欢

            train_data_dict
            数据为{'1': {'1': 1, '2': 1, '3': 1, '4': 1, '5': 1}, 
            '2': {'9': 1, '10': 1, '11': 1}}
            表示用户1对物品1态度为1(喜欢)
            调用方法为train_data[1][1]

        """

        # 选取模型
        self.model_name = model_name
        self.similarity_model = self.load_similarity_model(self.similarity_path)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()
        self.data_path = data_file

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item in self.similarity_model.keys():
            for neig in self.similarity_model[item]:
                sim_score = self.similarity_model[item][neig]
                max_score = max(max_score, sim_score)
                min_score = min(min_score, sim_score)
        return max_score, min_score

    def _convert_data_to_dict(self, data):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        train_data_dict = {}
        for user, item, record in data:
            train_data_dict.setdefault(user, {})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path='./similarity.pkl'):
        print("saving data to ", save_path)
        with open(save_path, 'wb') as write_file:
            # pickle.dump：将obj对象序列化存入已经打开的file中
            pickle.dump(dict_data, write_file)

    def _load_train_data(self, data_file=None):
        """
        read the data from the data file which is a data set
        """
        train_data = []
        train_data_list = []
        train_data_set_list = []
        for line in open(data_file).readlines():
            userid, items = line.strip().split(' ', 1)
            # only use training data
            items = items.split(' ')[:-3]
            train_data_list.append(items)
            train_data_set_list += items
            for itemid in items:
                train_data.append((userid, itemid, int(1)))
        return train_data_list, set(train_data_set_list), self._convert_data_to_dict(train_data)
        """
            train_data_list:
            [[],[],[]],每个[]为不含后3个item的序列
            train_data将每个用户喜欢的物品转化为一个列表
            [('1', '1', 1), ('1', '2', 1), ('1', '3', 1), ('1', '4', 1), ('1', '5', 1)]
            ()中第一个代表userId
            第二个代表itemId
            第三个每个都是1,表示喜欢

            _convert_data_to_dict(train_data)
            数据为{'1': {'1': 1, '2': 1, '3': 1, '4': 1, '5': 1}, 
            '2': {'9': 1, '10': 1, '11': 1}}
            表示用户1对物品1态度为1(喜欢)
            调用方法为train_data[1][1]
            
        """

    def _generate_item_similarity(self, train=None, save_path='./'):
        """
        calculate co-rated users between items
        """
        print("getting item similarity...")
        train = train or self.train_data_dict
        """
        train_data_dict
            数据为{'1': {'1': 1, '2': 1, '3': 1, '4': 1, '5': 1}, 
            '2': {'9': 1, '10': 1, '11': 1}}
            表示用户1对物品1态度为1(喜欢)
            调用方法为train_data[1][1]
        """
        C = dict()
        """
            {'1': {'2': 2, '3': 1, '4': 1, '5': 1, '10': 1, '11': 1}, 
                '10': {'9': 1, '11': 2, '1': 1, '2': 1}, 
                '11': {'9': 1, '10': 2, '1': 1, '2': 1}}
            字典C的作用是统计每个物品的相似物品,比如与物品10最相似的物品是11,在同一序列中出现了两次
        """

        N = dict()  # N统计了每个物品出现的次数

        if self.model_name in ['ItemCF', 'ItemCF_IUF']:
            print("Step 1: Compute Statistics")
            data_iter = tqdm(enumerate(train.items()), total=len(train.items()))
            for idx, (u, items) in data_iter:
                """
                    idx为迭代次数从0开始,u为用户,
                    items格式为{'9': 1, '10': 1, '11': 1} '物品9':喜欢
                """

                if self.model_name == 'ItemCF':
                    for i in items.keys():
                        """
                        i为物品编号
                        setdefault(key[, default]),如果没有 key,会加入这个key,值设为0
                        有这个key,直接返回字典中对应的key 的值    
                        """
                        N.setdefault(i, 0)
                        N[i] += 1  # N统计了每个物品出现的次数
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1
                elif self.model_name == 'ItemCF_IUF':
                    for i in items.keys():
                        N.setdefault(i, 0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1 / math.log(1 + len(items) * 1.0)
            self.itemSimBest = dict()
            """
                itemSimBest 是一个嵌套字典，其中第一层的键表示当前物品的编号，
                第二层的键表示与当前物品相似的其他物品的编号，
                第二层的值则表示这些相似物品与当前物品之间的相似度分数。
            """
            print("Step 2: Compute co-rate matrix")
            c_iter = tqdm(enumerate(C.items()), total=len(C.items()))
            for idx, (cur_item, related_items) in c_iter:
                self.itemSimBest.setdefault(cur_item, {})
                for related_item, score in related_items.items():
                    self.itemSimBest[cur_item].setdefault(related_item, 0);
                    self.itemSimBest[cur_item][related_item] = score / math.sqrt(N[cur_item] * N[related_item])
            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == 'Item2Vec':
            # details here: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
            print("Step 1: train item2vec model")
            item2vec_model = gensim.models.Word2Vec(sentences=self.train_data_list,
                                                    vector_size=20, window=5, min_count=0,
                                                    epochs=100)
            self.itemSimBest = dict()
            total_item_nums = len(item2vec_model.wv.index_to_key)
            print("Step 2: convert to item similarity dict")
            total_items = tqdm(item2vec_model.wv.index_to_key, total=total_item_nums)
            for cur_item in total_items:
                related_items = item2vec_model.wv.most_similar(positive=[cur_item], topn=20)
                self.itemSimBest.setdefault(cur_item, {})
                for (related_item, score) in related_items:
                    self.itemSimBest[cur_item].setdefault(related_item, 0)
                    self.itemSimBest[cur_item][related_item] = score
            print("Item2Vec model saved to: ", save_path)
            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == 'cosine_similarity':
            item_embed = np.load(os.path.join(self.args.data_dir, self.args.data_name+'_item_embedding.npz'))['item_embed']
            item_similarity_matrix = cosine_similarity(item_embed)

            self.itemSimBest = {}
            for i in range(item_similarity_matrix.shape[0]):
                self.itemSimBest[i] = {}
                for j in range(item_similarity_matrix.shape[1]):
                    if i == j:
                        continue
                    self.itemSimBest[i][j] = item_similarity_matrix[i, j]

            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == 'HNSW':
            item_embed = np.load(os.path.join(self.args.data_dir, self.args.data_name+'_item_embedding.npz'))['item_embed']

            # 创建 HNSW 索引
            hnsw_index = Index(space='cosine', dim=item_embed.shape[1])
            hnsw_index.init_index(max_elements=item_embed.shape[0], ef_construction=200, M=16)
            hnsw_index.add_items(item_embed)

            # 计算物品之间的相似度
            itemSimBest = {}
            for i in range(item_embed.shape[0]):
                neighbors, _ = hnsw_index.knn_query(item_embed[i], k=item_embed.shape[0])
                itemSimBest[i] = {}
                for j, score in zip(neighbors, _):
                    if i == j:
                        continue
                    itemSimBest[i][j] = 1 - score

            self._save_dict(itemSimBest, save_path=save_path)
        elif self.model_name == 'MinHash':
            item_embed = np.load(os.path.join(self.model_dir, 'ml-1m_item_embedding.npz'))['item_embed']

            # 创建 MinHashLSH 对象
            num_perm = 128  # 设置哈希函数的数量
            lsh = MinHashLSH(num_perm=num_perm)

            # 计算物品的 MinHash 签名并添加到 LSH 索引中
            for i in range(item_embed.shape[0]):
                item_vector = item_embed[i]
                minhash = MinHash(num_perm=num_perm)
                for j in range(item_vector.shape[0]):
                    minhash.update(str(item_vector[j]).encode('utf-8'))
                lsh.insert(i, minhash)

            # 构建相似度字典
            itemSimBest = {}
            for i in range(item_embed.shape[0]):
                itemSimBest[i] = {}
                query_minhash = lsh[i]
                results = lsh.query(query_minhash)
                for j in results:
                    if i == j:
                        continue
                    itemSimBest[i][j] = 1

            self._save_dict(itemSimBest, save_path=save_path)

    def load_similarity_model(self, similarity_model_path):
        if not similarity_model_path:
            raise ValueError('invalid path')
        elif not os.path.exists(similarity_model_path):
            print("the similirity dict not exist, generating...")
            """
                generate_item_similarity
            """
            self._generate_item_similarity(save_path=self.similarity_path)
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec',  'cosine_similarity', 'HNSW']:
            with open(similarity_model_path, 'rb') as read_file:
                similarity_dict = pickle.load(read_file)
            return similarity_dict
        elif self.model_name == 'Random':
            similarity_dict = self.train_item_list
            return similarity_dict

    def most_similar(self, item, top_k=1, with_score=False):
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'cosine_similarity', 'HNSW']:
            """TODO: handle case that item not in keys"""
            if str(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[str(item)].items(), key=lambda x: x[1], \
                                                reverse=True)[0:top_k]
                if with_score:
                    return list(
                        map(lambda x: (int(x[0]), (float(x[1]) - self.min_score) / (self.max_score - self.min_score)),
                            top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            elif int(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[int(item)].items(), key=lambda x: x[1], \
                                                reverse=True)[0:top_k]
                if with_score:
                    return list(
                        map(lambda x: (int(x[0]), (float(x[1]) - self.min_score) / (self.max_score - self.min_score)),
                            top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            else:
                item_list = list(self.similarity_model.keys())
                random_items = random.sample(item_list, k=top_k)
                if with_score:
                    return list(map(lambda x: (int(x), 0.0), random_items))
                return list(map(lambda x: int(x), random_items))
        elif self.model_name == 'Random':
            random_items = random.sample(self.similarity_model, k=top_k)
            if with_score:
                return list(map(lambda x: (int(x), 0.0), random_items))
            return list(map(lambda x: int(x), random_items))

