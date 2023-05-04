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

import torch
import torch.nn as nn
import gensim
from modules import Encoder, LayerNorm
from src.utils import cosine_similarity, get_user_seqs


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
    def __init__(self, data_file=None, similarity_path=None, model_name='ItemCF', \
                 dataset_name='Sports_and_Outdoors'):
        """
        :param data_file: 原始的数据集txt文件
        :param similarity_path:
        :param model_name:
        :param dataset_name:
        """
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
        elif self.model_name == 'graph_embedding':
            # 读取图嵌入的物品embedding向量
            item_emb_file = os.path.join(self.data_path, '_knowledgeGraph_embedding.txt')
            item_emb_dict = {}
            with open(item_emb_file, 'r') as f:
                for line in f.readlines():
                    cols = line.strip().split()
                    item_id = cols[0]
                    item_emb = list(map(float, cols[1].split(',')))
                    item_emb_dict[item_id] = item_emb

            # 构建相似度矩阵
            item_ids = list(item_emb_dict.keys())
            item_num = len(item_ids)
            item_sim_mat = np.zeros((item_num, item_num))
            for i in range(item_num):
                for j in range(i + 1, item_num):
                    sim = cosine_similarity([item_emb_dict[item_ids[i]]], [item_emb_dict[item_ids[j]]])[0][0]
                    item_sim_mat[i][j] = sim
                    item_sim_mat[j][i] = sim

            # 获取最相似的物品
            self.itemSimBest = {}
            for i in range(item_num):
                sim_scores = list(enumerate(item_sim_mat[i]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:self.n_similar_items + 1]
                self.itemSimBest[item_ids[i]] = [item_ids[j] for j, _ in sim_scores]

            # 保存字典到磁盘
            dict_file = os.path.join(self.data_path, 'item_sim_best.pkl')
            self._save_dict(self.itemSimBest, dict_file)

    def load_similarity_model(self, similarity_model_path):
        if not similarity_model_path:
            raise ValueError('invalid path')
        elif not os.path.exists(similarity_model_path):
            print("the similirity dict not exist, generating...")
            """
                generate_item_similarity
            """
            self._generate_item_similarity(save_path=self.similarity_path)
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN']:
            with open(similarity_model_path, 'rb') as read_file:
                similarity_dict = pickle.load(read_file)
            return similarity_dict
        elif self.model_name == 'Random':
            similarity_dict = self.train_item_list
            return similarity_dict

    def most_similar(self, item, top_k=1, with_score=False):
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN']:
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


if __name__ == '__main__':
    if True:
        parser = argparse.ArgumentParser()
        # system args
        parser.add_argument('--data_dir', default='../data/', type=str)
        parser.add_argument('--output_dir', default='output/', type=str)
        parser.add_argument('--data_name', default='ml-1m', type=str)
        parser.add_argument('--do_eval', action='store_true', default=False)
        parser.add_argument('--model_idx', default=0, type=int, help="model idenfier 10, 20, 30...")
        parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

        # data augmentation args
        parser.add_argument('--noise_ratio', default=0.0, type=float,
                            help="percentage of negative interactions in a sequence - robustness analysis")
        parser.add_argument('--training_data_ratio', default=1.0, type=float,
                            help="percentage of training samples used for training - robustness analysis")
        parser.add_argument('--augment_threshold', default=4, type=int,
                            help="control augmentations on short and long sequences.\
                                default:-1, means all augmentations types are allowed for all sequences.\
                                For sequence length < augment_threshold: Insert, and Substitute methods are allowed \
                                For sequence length > augment_threshold: Crop, Reorder, Substitute, and Mask \
                                are allowed.")
        parser.add_argument('--similarity_model_name', default='ItemCF', type=str,
                            help="Method to generate item similarity score. choices: \
                                Random, ItemCF, ItemCF_IUF(Inverse user frequency), Item2Vec, LightGCN")
        parser.add_argument("--augmentation_warm_up_epoches", type=float, default=160,
                            help="number of epochs to switch from \
                                memory-based similarity model to \
                                hybrid similarity model.")
        parser.add_argument('--insert-first', action='store_true', default=True,
                            help='whether to perform insert augmentation first')
        parser.add_argument('--base_augment_type', default='crop', type=str,
                            help="default data augmentation types. Chosen from: \
                                mask, crop, reorder, substitute, insert, random, \
                                combinatorial_enumerate (for multi-view).")
        parser.add_argument('--augment_type_for_short', default='SIM', type=str,
                            help="data augmentation types for short sequences. Chosen from: \
                                SI, SIM, SIR, SIC, SIMR, SIMC, SIRC, SIMRC.")
        parser.add_argument("--tao", type=float, default=0.2, help="crop ratio for crop operator")
        parser.add_argument("--gamma", type=float, default=0.7, help="mask ratio for mask operator")
        parser.add_argument("--beta", type=float, default=0.2, help="reorder ratio for reorder operator")
        parser.add_argument("--substitute_rate", type=float, default=0.1,
                            help="substitute ratio for substitute operator")
        parser.add_argument("--insert_rate", type=float, default=0.4,
                            help="insert ratio for insert operator")
        parser.add_argument("--max_insert_num_per_pos", type=int, default=1,
                            help="maximum insert items per position for insert operator - not studied")

        # contrastive learning task args
        parser.add_argument('--temperature', default=1.0, type=float,
                            help='softmax temperature (default:  1.0) - not studied.')
        parser.add_argument('--n_views', default=2, type=int, metavar='N',
                            help='Number of augmented data for each sequence - not studied.')

        # model args
        parser.add_argument("--model_name", default='CoSeRec', type=str)
        parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
        parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
        parser.add_argument('--num_attention_heads', default=2, type=int)
        parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
        parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
        parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
        parser.add_argument("--initializer_range", type=float, default=0.02)
        parser.add_argument('--max_seq_length', default=50, type=int)

        # train args
        parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
        parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
        parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
        parser.add_argument("--no_cuda", action="store_true")
        parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
        parser.add_argument("--seed", default=1, type=int)
        parser.add_argument("--cf_weight", type=float, default=0.1, \
                            help="weight of contrastive learning task")
        parser.add_argument("--rec_weight", type=float, default=1.0, \
                            help="weight of contrastive learning task")

        # learning related
        parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
        parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
        parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

        args = parser.parse_args()

        args = parser.parse_args()
        args.data_file = args.data_dir + args.data_name + '.txt'
        user_seq, max_item, valid_rating_matrix, test_rating_matrix = \
            get_user_seqs(args.data_file)
        # print(user_seq)
        args.item_size = max_item + 2

        args.mask_id = max_item + 1

        # save model args
        args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}'
        args.log_file = os.path.join(args.output_dir, args_str + '.txt')
        args.train_matrix = valid_rating_matrix
        checkpoint = args_str + '.pt'
        args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

        # -----------   pre-computation for item similarity   ------------ #
        args.similarity_model_path = os.path.join(args.data_dir, \
                                                  args.data_name + '_' + args.similarity_model_name + '_similarity.pkl')
    # 实例化模型并打印模型结构
    # 实例化模型并打印模型结构
    model = SASRecModel(args)
    print(model)

    # 随机初始化embedding
    # random_embedding = model.item_embeddings.weight.detach().clone()
    # print('Randomly initialized embedding shape:', random_embedding.shape)
    # print(random_embedding[:10])

    # 预训练embedding
    pretrained_embedding = model.item_embeddings.weight.detach().clone()
    print('Pretrained embedding shape:', pretrained_embedding.shape)
    print(pretrained_embedding[:10])