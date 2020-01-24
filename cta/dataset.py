import pandas as pd
import numpy as np
import torch
import random

import pickle

class Dataset(object):

    def __init__(self, data_prefix, data_name, observed_threshold, window_size, itemKey, timeKey):
        time_filename = data_prefix + timeKey
        time_seq_arr_total = pickle.load(open(time_filename, "rb"))

        item_filename = data_prefix + itemKey
        data_file = open(item_filename, "rb")
        m_seq_list = pickle.load(data_file) 
        
        seq_num = len(m_seq_list)
        print("seq num", seq_num)
        
        self.m_input_action_seq_list = []
        self.m_target_action_seq_list = []
        self.m_input_seq_idx_list = []
        self.m_time_action_seq_list = []
        self.m_features_seq_list = []
        
        self.m_itemmap = set()
        print("loading data")
        for seq_index in range(seq_num):
            action_seq_arr = m_seq_list[seq_index]
            time_seq_arr =  time_seq_arr_total[seq_index]
            
            feature_arr = np.zeros( len(action_seq_arr) )
            feature_map = {}
            idx = 0
            for item in action_seq_arr:
#                 if item == 0: print(action_seq_arr)
                self.m_itemmap.add(item)
                
                if item not in feature_map: feature_map[item] = 0
                feature_map[item] += 1
                
#                 feature_arr[idx] = feature_map[item]
                subspace_size = 20
                feature_arr[idx] = feature_map[item] if feature_map[item] < subspace_size else subspace_size
                idx += 1
            
#             rank_freq = dict(sorted(feature_map.items(), key=lambda kv: kv[1], reverse=True) )

#             for item in action_seq_arr:  
#                 feature_arr[idx] = rank_freq[item] if rank_freq[item] < 10 else 10
#                 idx += 1

                
            action_num_seq = len(action_seq_arr)

            if action_num_seq < window_size :
                window_size = action_num_seq

            for action_index in range(observed_threshold, window_size):
                self.m_input_action_seq_list.append( action_seq_arr[:action_index])
                self.m_target_action_seq_list.append( action_seq_arr[action_index])
                self.m_input_seq_idx_list.append( action_index)
                self.m_time_action_seq_list.append(  np.asarray( time_seq_arr[action_index]) - np.asarray(time_seq_arr[:action_index]) )
                self.m_features_seq_list.append( feature_arr[:action_index])

            for action_index in range(window_size, action_num_seq):
                self.m_input_action_seq_list.append( action_seq_arr[action_index-window_size+1:action_index])
                self.m_target_action_seq_list.append( action_seq_arr[action_index])
                self.m_input_seq_idx_list.append(action_index)
                self.m_time_action_seq_list.append(  np.asarray(time_seq_arr[action_index]) - np.asarray(time_seq_arr[action_index-window_size+1:action_index]) )
                self.m_features_seq_list.append( feature_arr[action_index-window_size+1:action_index])

    def __len__(self):
        return len(self.m_input_action_seq_list)

    def __getitem__(self, index):
        x = self.m_input_action_seq_list[index]
        y = self.m_target_action_seq_list[index]
        t = self.m_time_action_seq_list[index]

        x_tensor = torch.LongTensor(np.asarray(x))
        y_tensor = torch.LongTensor(np.asarray(y))
        t_tensor = torch.FloatTensor(t)
        
        return x_tensor, y_tensor, t_tensor

    @property
    def items(self):
        return self.m_itemmap


class DataLoader():
    def __init__(self, dataset, batch_size):
        self.m_dataset = dataset
        self.m_batch_size = batch_size

    def __iter__(self):
        print("shuffling")

        
        batch_size = self.m_batch_size
        input_action_seq_list = self.m_dataset.m_input_action_seq_list
        target_action_seq_list = self.m_dataset.m_target_action_seq_list
        input_time_seq_list = self.m_dataset.m_time_action_seq_list
        input_seq_idx_list = self.m_dataset.m_input_seq_idx_list
        features_seq_list = self.m_dataset.m_features_seq_list
        
#         temp = list(zip(input_action_seq_list, target_action_seq_list, input_time_seq_list, input_seq_idx_list, features_seq_list))
#         random.shuffle(temp)
#         input_action_seq_list, target_action_seq_list, input_time_seq_list, input_seq_idx_list, features_seq_list = zip(*temp)
        
        temp = np.arange(len(input_action_seq_list) )
        np.random.shuffle( temp )
        
        input_num = len(input_action_seq_list)
        batch_num = int(input_num/batch_size)

        for batch_index in range(batch_num):
            x_batch = []
            y_batch = []
            t_batch = []
            idx_batch = []
            feature_batch = []

            for seq_index_batch in range(batch_size):
                seq_index = batch_index*batch_size+seq_index_batch
                seq_index = temp[seq_index]

                x_batch.append(input_action_seq_list[seq_index])
                y_batch.append(target_action_seq_list[seq_index])
                t_batch.append(input_time_seq_list[seq_index])
                idx_batch.append(input_seq_idx_list[seq_index])
                feature_batch.append(features_seq_list[seq_index])
                
#             x_batch = input_action_seq_list[ batch_index*batch_size : batch_index*(batch_size+1)]
#             y_batch = target_action_seq_list[ batch_index*batch_size : batch_index*(batch_size+1)]
#             t_batch = input_time_seq_list[ batch_index*batch_size : batch_index*(batch_size+1)]
#             idx_batch = input_seq_idx_list[ batch_index*batch_size : batch_index*(batch_size+1)]              
#             feature_batch = feature_batch[ batch_index*batch_size : batch_index*(batch_size+1)]              
            
            x_batch, y_batch, t_batch, idx_batch, feature_batch = self.batchifyData(x_batch, y_batch, t_batch, idx_batch, feature_batch)

            x_batch_tensor = torch.LongTensor(x_batch)
            y_batch_tensor = torch.LongTensor(y_batch)
            t_batch_tensor = torch.FloatTensor(t_batch)
            idx_batch_tensor = torch.LongTensor(idx_batch)
#             feature_batch_tensor = torch.FloatTensor(feature_batch)
            feature_batch_tensor = torch.LongTensor(feature_batch)

            yield x_batch_tensor, y_batch_tensor, t_batch_tensor, idx_batch_tensor, feature_batch_tensor

    def batchifyData(self, input_action_seq_batch, target_action_seq_batch, input_time_seq_batch, idx_batch, feature_batch):
        longest_len_batch = max([len(seq_i) for seq_i in input_action_seq_batch])
        batch_size = len(input_action_seq_batch)

        pad_input_action_seq_batch = np.zeros((batch_size, longest_len_batch))
        pad_target_action_seq_batch = np.zeros(batch_size)
        pad_input_time_seq_batch = np.zeros((batch_size, longest_len_batch))
        pad_idx_batch = np.zeros(batch_size)
        pad_feature_batch = np.zeros((batch_size, longest_len_batch, 1))
        
        zip_batch = sorted(zip(idx_batch, input_action_seq_batch, target_action_seq_batch, input_time_seq_batch, feature_batch), key=lambda x: x[0], reverse=True)

        for seq_i, (seq_idx, input_action_seq_i, target_action_seq_i, input_time_seq_i, feature_i) in enumerate(zip_batch):
            seq_len_i = len(input_action_seq_i)
            pad_input_action_seq_batch[seq_i, 0:seq_len_i] = input_action_seq_i
            pad_input_time_seq_batch[seq_i, 0:seq_len_i] = input_time_seq_i
            pad_target_action_seq_batch[seq_i] = target_action_seq_i
            pad_idx_batch[seq_i] = seq_idx
            pad_feature_batch[seq_i, 0:seq_len_i] = np.expand_dims(feature_i, axis=-1)

        return pad_input_action_seq_batch, pad_target_action_seq_batch, pad_input_time_seq_batch, pad_idx_batch, pad_feature_batch

def load_sample():
    window_size = 8
    observed_threshold = 5
    data_name = 'xing'
    train_data_dir = '/af12/jw7jb/data/xing/standard/test'
    train_data = Dataset(train_data_dir, data_name, observed_threshold, window_size, '_item.pickle', '_time.pickle')
    batch_size = 1
    train_data_loader = DataLoader(train_data, batch_size)
    
    for input_x_batch, target_y_batch, input_t_batch, x_len_batch in train_data_loader:
        torch.save(input_x_batch, 'checkpoint/input_x_sample')
        torch.save(target_y_batch, 'checkpoint/target_y_sample')
        torch.save(input_t_batch, 'checkpoint/input_t_sample')
        torch.save(x_len_batch, 'checkpoint/x_len_sample')
        break
    
    
if __name__ == '__main__':
    load_sample()