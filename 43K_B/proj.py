#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:01:34 2018

@author: admin
"""

import os
import re
import numpy as np
import pandas as pd
import igraph
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score, recall_score, make_scorer, \
    classification_report, matthews_corrcoef

#pip install imblearn
#conda update scikit-learn
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier

# To be able to plot with igraph is a bit troublesome,
    # please do the below steps
from igraph.drawing import plot
import cairo #conda install pycairo
# follow instruction here: https://lists.nongnu.org/archive/html/igraph-help/2017-01/msg00024.html

import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import pydot
import pickle

pd.set_option('display.max_columns', 100)

class TwitterData():
    
    def __init__(self, dir_path, struct_file_path):
        
        #self.num_cores = multiprocessing.cpu_count()-1
        
        if (not dir_path):
            self.dir_ref = r"./data/twitter"
        else:
            self.dir_ref = dir_path
            
        if (not struct_file_path):
            self.struct_file_path = r"twitter_combined.txt"
        else:
            self.struct_file_path = struct_file_path
        
        self.struct_data_pos = None
        self.struct_data_pos_orig = None
        self.struct_data_neg = None
        self.struct_data_full = None
        self.graph_nodes = None
        self.graph_struct_n2id = None
        self.graph_struct_pos = None
        self.graph_struct_neg = None
        self.graph_struct_full = None
        self.node_features = dict()
        
    ###############################################
    ### Load data with only positive labels
    ###############################################
    def load_struct_data(self, p=0.01, n=None):
        print("in method: load_struct_data()")
        self.struct_data_pos = pd.read_csv(self.dir_ref + "/" + self.struct_file_path, 
                                           names=["from", "to"], header=None, sep=" ")
        # subset some rows for use - TESTING purpose
        if p is not None:
            print("subsetted {} records".format(int(p*len(self.struct_data_pos))))
            self.struct_data_pos = self.struct_data_pos[:int(p*len(self.struct_data_pos))].drop_duplicates()
        elif n is not None:
            print("subsetted {} records".format(n))
            self.struct_data_pos = self.struct_data_pos[:n].drop_duplicates()
            
    ###############################################
    ### Init graph with positive labels data
    ###############################################
    def init_graph(self):
        print("in method: init_graph()")
        self.graph_struct_pos = igraph.Graph(directed=True)
        nodes_duplicates = pd.Series.to_frame(
                pd.concat([self.struct_data_pos["from"],self.struct_data_pos["to"]]))
        nodes_duplicates.columns = ["nodes"]
        self.graph_nodes = nodes_duplicates.nodes.unique()
        self.graph_struct_pos.add_vertices(self.graph_nodes)
        
        self.graph_struct_n2id = dict(zip(self.graph_struct_pos.vs["name"],
                                                [v.index for v in self.graph_struct_pos.vs]))
        self.struct_data_pos["from_id"] = self.struct_data_pos.apply (lambda row: \
                            self.graph_struct_n2id[row["from"]], axis=1)
        self.struct_data_pos["to_id"] = self.struct_data_pos.apply (lambda row: \
                            self.graph_struct_n2id[row["to"]], axis=1)
        
        pos_tuples = [tuple(x) for x in self.struct_data_pos[['from_id', 'to_id']].values]
        self.graph_struct_pos.add_edges(pos_tuples)
        
    ###############################################
    ### Remove low degree nodes < 3
    ###############################################
    def remove_low_degree_nodes(self):
        print("in method: remove_low_degree_nodes()")
        to_delete_names_all = []
        round_index = 0
#        print(self.graph_struct_pos.summary())
        
        while True:
#            print("round {}".format(round_index))
#            node_list_less_than_3_d = pd.DataFrame({
#            "node": 
#                self.graph_struct_pos.degree(mode="all")
#            })
            #print("nodes with < 3 degree: {}".format(node_list_less_than_3_d.loc[(node_list_less_than_3_d['node'] < 3)].index))
            #print("length of nodes < 3 degree - updated: {}" \
                  #.format(len(node_list_less_than_3_d.loc[(node_list_less_than_3_d['node'] < 3)].index)))
            
            to_delete_ids = [v.index for v in self.graph_struct_pos.vs if v.degree(mode="all") < 3]
#            print("length of ids to delete: {}".format(round_index, len(to_delete_ids)))
            
            if(len(to_delete_ids) == 0) :
                break
            
            # Note that the indexes after each deletion will be shifted, 
                # so u store the names instead of indexes
            to_delete_names_all =  to_delete_names_all \
               + [k["name"] for k in self.graph_struct_pos.vs if k.index in to_delete_ids]
#            print("consolidated names to delete: ".format(len(to_delete_names_all)))
            
            self.graph_struct_pos.delete_vertices(to_delete_ids)
#            print(self.graph_struct_pos.summary())
            
            round_index = round_index+1

        
        self.graph_struct_n2id = dict(zip(self.graph_struct_pos.vs["name"],
                                                    [v.index for v in self.graph_struct_pos.vs]))

#        print("before filter len: {}".format(len(self.struct_data_pos)))
        self.struct_data_pos = self.struct_data_pos[~self.struct_data_pos['from'].isin(to_delete_names_all)]
        self.struct_data_pos = self.struct_data_pos[~self.struct_data_pos['to'].isin(to_delete_names_all)]
#        print("after filter len: {}".format(len(self.struct_data_pos)))
        
        self.struct_data_pos["from_id"] = self.struct_data_pos.apply (lambda row: \
                            self.graph_struct_n2id[row["from"]], axis=1)
        self.struct_data_pos["to_id"] = self.struct_data_pos.apply (lambda row: \
                            self.graph_struct_n2id[row["to"]], axis=1)
        
        self.graph_nodes = [v["name"] for v in self.graph_struct_pos.vs]
        

    ###############################################
    ### Remove 10% of the edges (previous state)
    ###############################################
    def remove_random_edges(self, percent):
        print("in method: remove_random_edges()")
        # store original data
        self.struct_data_pos_orig = self.struct_data_pos.copy()
        self.graph_struct_pos_orig = self.graph_struct_pos.copy()
        
        random.seed(123)
        # this is sample without replacement - confirmed
        to_remove = random.sample(range(self.struct_data_pos.shape[0]), 
                                  k=int(round(self.struct_data_pos.shape[0] * (percent))))
        
        self.struct_data_pos_removed = self.struct_data_pos.iloc[to_remove]
#        self.struct_data_pos_removed.to_csv("struct_data_pos_removed.csv")
        
        self.struct_data_pos = self.struct_data_pos.iloc[~self.struct_data_pos.index.isin(to_remove)]
        print("removed edges len: {}".format(len(self.struct_data_pos_removed))) #4939 
        
        self.init_graph() # create the pos graph again
    
    ###############################################################
    ### Create complementary graph with negative labels data
    ###############################################################
    def complement_pos_graph(self):
        print("in method: complement_pos_graph()")
        self.graph_struct_full = igraph.Graph.Full(len(self.graph_nodes),directed=True)
        self.graph_struct_full.vs["name"] = self.graph_nodes
        
        # find all the complement edges not in new graph
        self.graph_struct_neg = igraph.Graph(directed=True)
        self.graph_struct_neg.add_vertices(self.graph_nodes)
        self.graph_struct_neg.add_edges(list(set(self.graph_struct_full.get_edgelist()) ^ \
                                             set(self.graph_struct_pos.get_edgelist())))
        
        neg_tuples = [e.tuple for e in self.graph_struct_neg.es]
        self.struct_data_neg = pd.DataFrame(data = [(self.graph_struct_neg.vs[t[0]]["name"], self.graph_struct_neg.vs[t[1]]["name"], t[0], t[1]) \
                                for t in neg_tuples], columns=["from", "to", "from_id", "to_id"])
        self.struct_data_full = pd.concat([self.struct_data_pos, self.struct_data_neg], ignore_index=True)
        
        # add in all the complementary edges and removed edges to predict
        self.struct_data_to_pred = pd.concat([self.struct_data_pos_removed, self.struct_data_neg], ignore_index=True)
        
        
        self.struct_data_to_pred = self.struct_data_to_pred.assign(label = 0)
#        print(len(self.struct_data_pos_removed))
#        print(len(self.struct_data_pos_removed.drop_duplicates()))
#        print(len(self.struct_data_neg))
#        print(len(self.struct_data_neg.drop_duplicates()))
#        print(len(self.struct_data_to_pred))
#        print(len(self.struct_data_to_pred.drop_duplicates()))
#        
#        self.struct_data_to_pred.to_csv("struct_data_to_pred.csv")
#        self.struct_data_pos_orig.to_csv("struct_data_pos_orig.csv")
        
        # update that removed edges to label = 1
        df = pd.merge(self.struct_data_to_pred, \
                      self.struct_data_pos_orig, on=['from_id','to_id'], \
                      how='left', indicator='Exist')
        
#        df.to_csv("merged_df.csv")
        
        self.struct_data_to_pred.loc[df[df['Exist']=='both'].index.values.astype(int),"label"] = 1
        
        # reason for not balancing here is cos after calculating common neighbours,
            # negative eg set can get lesser than positive eg set for those with common neighbours
 
    ############################################################
    ### Create topographical structure features for these data
    ############################################################
    def create_struct_feat(self):
        print("in method: create_struct_feat()")
        
        df = self.struct_data_to_pred
        
        df = df.assign(d_in_u = 0, d_out_u = 0,
                       d_in_u_div_d_out_u = 0., d_out_u_div_d_in_u = 0.,
                       d_in_v = 0, d_out_v = 0,
                       d_in_v_div_d_out_v = 0., d_out_v_div_d_in_v = 0.,
                       t_1 = 0, t_2 = 0, t_3 = 0, t_4 = 0, 
                       t_1_div_cn = 0., t_2_div_cn = 0., t_3_div_cn = 0., t_4_div_cn = 0.)
        
        for index, row in df.iterrows():
            #print("**{}** from:{}, to:{}, from_id:{}, to_id:{}"
                  #.format(index, row["from"], row["to"], row["from_id"], row["to_id"]))
            from_in = set(self.graph_struct_pos.neighbors(int(row["from_id"]), mode = "in"))
            from_out = set(self.graph_struct_pos.neighbors(int(row["from_id"]), mode = "out"))
            to_in = set(self.graph_struct_pos.neighbors(int(row["to_id"]), mode = "in"))
            to_out = set(self.graph_struct_pos.neighbors(int(row["to_id"]), mode = "out"))
            
            len_from_in = len(from_in)
            len_from_out = len(from_out)
            len_to_in = len(to_in)
            len_to_out = len(to_out)
            
            common_from_out_to_in = from_out & to_in
            common_from_out_to_out = from_out & to_out
            common_from_in_to_in = from_in & to_in
            common_from_in_to_out = from_in & to_out
            common_from_to = (from_out | from_in) & (to_out | to_in)
            
            len_common_from_out_to_in = len(common_from_out_to_in)
            len_common_from_out_to_out = len(common_from_out_to_out)
            len_common_from_in_to_in = len(common_from_in_to_in)
            len_common_from_in_to_out = len(common_from_in_to_out)
            len_common_from_to = len(common_from_to)
            
            # 1. d_in(u)
            df.at[index,"d_in_u"] = len_from_in
            # 2. d_out(u)
            df.at[index,"d_out_u"] = len_from_out
            # 3. d_in(u)/d_out(u)
            if len_from_out != 0:
                df.at[index,"d_in_u_div_d_out_u"] = float(len_from_in) / float(len_from_out)
            # 4. d_out(u)/d_in(u)
            if len_from_in != 0:
                df.at[index,"d_out_u_div_d_in_u"] = float(len_from_out) / float(len_from_in)
            
            # 5. d_in(v)
            df.at[index,"d_in_v"] = len_to_in
            # 6. d_out(v)
            df.at[index,"d_out_v"] = len_to_out
            # 7. d_in(v)/d_out(v)
            if len_to_out != 0:
                df.at[index,"d_in_v_div_d_out_v"] = float(len_to_in) / float(len_to_out)
            # 8. d_out(v)/d_in(v)
            if len_to_in != 0:
                df.at[index,"d_out_v_div_d_in_v"] = float(len_to_out) / float(len_to_in)
                
            # 9. t_1
            df.at[index,"t_1"] = len_common_from_out_to_in
            # 10. t_2
            df.at[index,"t_2"] = len_common_from_out_to_out
            # 11. t_3
            df.at[index,"t_3"] = len_common_from_in_to_in
            # 12. t_4
            df.at[index,"t_4"] = len_common_from_in_to_out
            
            # common neighbours
            df.at[index,"cn"] = len_common_from_to
            
            if len_common_from_to != 0:
            # 13. t_1 / C(u,v)
                df.at[index,"t_1_div_cn"] = len_common_from_out_to_in / len_common_from_to
            # 14. t_2 / C(u,v)
                df.at[index,"t_2_div_cn"] = len_common_from_out_to_out / len_common_from_to
            # 15. t_3 / C(u,v)
                df.at[index,"t_3_div_cn"] = len_common_from_in_to_in / len_common_from_to
            # 16. t_4 / C(u,v)
                df.at[index,"t_4_div_cn"] = len_common_from_in_to_out / len_common_from_to
        
        # filter only those for prediction if common neighbours != 0
        df = df.loc[df['cn'] != 0]
        self.struct_data_to_pred = df.drop(columns=['cn'])
        
    
#    ############################################################################################
#    ### Balance positive and negative examples
#    ############################################################################################
#    def balance_pred_set(self):
#        print("in method: balance_pred_set()")
#        
#        self.struct_data_to_pred_pos = self.struct_data_to_pred.loc[self.struct_data_to_pred['label'] == 1]
#        print("pos eg: {}".format(len(self.struct_data_to_pred_pos)))
#        self.struct_data_to_pred_neg = self.struct_data_to_pred.loc[self.struct_data_to_pred['label'] == 0]
#        print("neg eg - before: {}".format(len(self.struct_data_to_pred_neg)))
#        
#        self.struct_data_to_pred_pos.reset_index(drop=True)
#        self.struct_data_to_pred_neg.reset_index(drop=True)
#
#        neg_subset_indexes = random.sample(range(self.struct_data_to_pred_neg.shape[0]), 
#                              k=int(self.struct_data_to_pred_pos.shape[0]))
#        self.struct_data_to_pred_neg = self.struct_data_to_pred_neg.iloc[neg_subset_indexes]
#        self.struct_data_to_pred_neg.reset_index(drop=True)
#        print("neg eg - after: {}".format(len(self.struct_data_to_pred_neg)))
#        
#        self.struct_data_to_pred = pd.concat([self.struct_data_to_pred_pos, self.struct_data_to_pred_neg], ignore_index=True)
#        print("total eg: {}".format(len(self.struct_data_to_pred)))
        
    ############################################################################################
    ### Create heuristic features for these data - Adamic/Adar，Katz & Common Neighbours
    ### Note that here we only use followers
    ############################################################################################
    def create_heuristic_feat(self):
        print("in method: create_heuristic_feat()")
        
        df = self.struct_data_to_pred
        df = df.assign(cn = 0, jaccard_sim = 0., adamic_adar = 0.,
                       prefer_att = 0., katz = 0.)
        
        jaccard_sim_for_all = self.graph_struct_pos.similarity_jaccard(mode="out")
        aa_for_all = self.graph_struct_pos.similarity_inverse_log_weighted(mode="out")
        katz_sim_for_all = katz_similarity(self.graph_struct_pos)
        
        # here we only use common follower
        for index, row in df.iterrows():
            
            from_out = set(self.graph_struct_pos.neighbors(int(row["from_id"]), mode = "out"))
            to_out = set(self.graph_struct_pos.neighbors(int(row["to_id"]), mode = "out"))
        
            common_from_out_to_out = from_out & to_out
        
            # Common Neighbour - we redefine this from the one defined in create_struct_feat
            df.at[index,"cn"] = len(common_from_out_to_out)
            
            # Jaccard Similarity
            df.at[index,"jaccard_sim"] = jaccard_sim_for_all[int(row["from_id"])][int(row["to_id"])]
            
            # Adamic / Adar
            df.at[index,"adamic_adar"] = aa_for_all[int(row["from_id"])][int(row["to_id"])]
            
            # Preferential Attachment
            df.at[index,"prefer_att"] = len(from_out) * len(to_out)
            
            # Katz
            df.at[index,"katz"] = katz_sim_for_all[int(row["from_id"]), int(row["to_id"])]
            
        self.struct_data_to_pred = df
        

    ############################################################
    ### Load domain features for the data
    ############################################################
    def loadlocFeatKey(self, ego):
        filename = os.path.join(self.dir_ref,str(ego)+'.featnames')
        ego_features = dict()
        with open(filename, 'r') as document:
            feat_names = document.readlines()
        for feat in feat_names:
            feat = feat.split(' ')
            local_id = int(feat[0])
            feat_raw = feat[1].rstrip('\n')
            # remove pure number
            #remove_number = re.sub(r'^[0-9]*$','', feat_raw[1:])
            # remove trailing non-alphanumeric characters
            remove_character = re.sub(r'[^a-zA-Z\d]*$','', feat_raw[1:])
            if remove_character:
                feat_name = feat_raw[0]+ remove_character
                #if feat_name not in ego_features.values():
                ego_features[local_id] = feat_name
        return ego_features
            
    def loadAnEgoFeat(self, ego, ego_features):
        filename = os.path.join(self.dir_ref, str(ego)+'.feat')
        
        with open(filename, "r") as document:
            nodes_feat = document.readlines()
        for node_feat in nodes_feat:
            feats = node_feat.split(' ')
            feats[len(feats)-1]=feats[len(feats)-1].rstrip('\n')
            node_id = str(feats[0])
            del feats[0]
        
            if node_id not in self.node_features:
                self.node_features[node_id] = set()
            for i in range(len(feats)):
                if feats[i] == '1' and (i in ego_features) and (ego_features[i] not in self.node_features[node_id]):
                    self.node_features[node_id].add(ego_features[i])
            
        filename_ego = os.path.join(self.dir_ref, str(ego)+'.egofeat')
        with open(filename_ego, "r") as document:
            nodes_feat = document.readlines()
        for node_feat in nodes_feat:
            feats = node_feat.split(' ')
            feats[len(feats)-1]=feats[len(feats)-1].rstrip('\n')
            node_id = str(ego)
            if node_id not in self.node_features:
                self.node_features[node_id] = set()
            for i in range(len(feats)):
                if feats[i] == '1' and (i in ego_features) and (ego_features[i] not in self.node_features[node_id]):
                    self.node_features[node_id].add(ego_features[i])

    def load_domain_data(self):
        print("in method: load_domain_data()")
        files_all = os.listdir(self.dir_ref)
        regex_feat = re.compile(r"\d+\.feat$")
        regex_egofeat = re.compile(r"\d+\.egofeat")
        regex_featnames = re.compile(r"\d+\.featnames")
        files_feat = list(filter(regex_feat.search, files_all))
        files_egofeat = list(filter(regex_egofeat.search, files_all))
        files_featnames = list(filter(regex_featnames.search, files_all))
        egos = [i[:-5] for i in files_feat]
        
        for ego in egos:
            ego_features = self.loadlocFeatKey(ego)
            self.loadAnEgoFeat(ego, ego_features)
    
    ############################################################
    ### Create domain features for these data
    ############################################################
    def create_domain_feat(self):
        print("in method: create_domain_feat()")
        df = self.struct_data_to_pred
        
        for index, row in df.iterrows():
            from_node = str(df.loc[index, 'from']) 
            to_node = str(df.loc[index, 'to'])
            from_hash = {i for i in self.node_features[from_node] if i[0]=='#'}
            from_mention = {i for i in self.node_features[from_node] if i[0]=='@'}
            to_hash = {i for i in self.node_features[to_node] if i[0]=='#'}
            to_mention = {i for i in self.node_features[to_node] if i[0]=='@'}
            union_hash = from_hash | to_hash
            union_mention = from_mention | to_mention
            if union_hash:    
                common_hash = len(from_hash & to_hash) / len(union_hash)
            else:
                common_hash = 0
            if union_mention:
                common_mention = len(from_mention & to_mention) / len(union_mention)
            else:
                common_mention = 0
            df.at[index,"common_hash"] = common_hash
            df.at[index,"common_mention"] = common_mention
        self.struct_data_to_pred = df
    
#############
#  Similarity measure based on all paths in a graph. 
#  This function counts all the paths between given pair of nodes, with shorter
#  paths counting more heavily. Weigths are exponential.
#    Katz centrality for node $i$ is
#    .. math::
#        x_i = \alpha \sum_{j} A_{ij} x_j + \beta,
#    where $A$ is the adjacency matrix of graph G with eigenvalues $\lambda$.
#    The parameter $\beta$ controls the initial centrality
#############
def katz_similarity(graph, alpha = 0.001, beta=1.0):
    print("in method: katz_similarity()")
    
    A  = np.matrix(graph.get_adjacency().data) # returns the directed adjacency
    I = np.identity(A.shape[0]) #form an Identity matrix with no of rows of A
    tmp = beta * (I - alpha * A)
    score = np.linalg.inv(tmp) #calculate inverse
    np.fill_diagonal(score, 0)
    return (score)


#####################################################################
### Massage data for running models
#####################################################################

def massage_dataset(dataset, test_proportion, col_to_drop, target_col, 
                    has_struct_partial=False, has_struct_full=False,
                    has_heuristic=False, has_domain=False,
                    sampling_strategy=None, random_state=0):
    print("in method: massage_dataset()")
    
    dataset_tmp = dataset.drop(columns=col_to_drop).copy()
    print(dataset_tmp.columns.values)
    
    dataset_1 = dataset_tmp[['d_in_u', 'd_out_u', 'd_in_u_div_d_out_u', 'd_out_u_div_d_in_u',
                   'd_in_v', 'd_out_v', 'd_in_v_div_d_out_v', 'd_out_v_div_d_in_v']]  
    dataset_1_full = dataset_tmp[['d_in_u', 'd_out_u', 'd_in_u_div_d_out_u', 'd_out_u_div_d_in_u',
                   'd_in_v', 'd_out_v', 'd_in_v_div_d_out_v', 'd_out_v_div_d_in_v',
                   't_1', 't_2', 't_3', 't_4',
                   't_1_div_cn', 't_2_div_cn', 't_3_div_cn', 't_4_div_cn']]
    dataset_2 = dataset_tmp[['cn', 'jaccard_sim', 'adamic_adar', 'prefer_att', 'katz']]
    dataset_3 = dataset_tmp[['common_hash', 'common_mention']]
    
    if(has_struct_partial):
        has_struct = True
    elif(has_struct_full):
        has_struct = True
        dataset_1 = dataset_1_full
    else:
        has_struct = False
        
    if(has_struct and not has_heuristic and not has_domain) :  # only struct (partial / full)
        dataset = dataset_1
    elif(not has_struct and has_heuristic and not has_domain) :  # only heuristic
        dataset = dataset_2
    elif(not has_struct and not has_heuristic and has_domain) :  # only domain
        dataset = dataset_3
    elif(has_struct and has_heuristic and not has_domain) :  # struct + heuristic
        dataset = pd.concat([dataset_1.reset_index(drop=True), dataset_2.reset_index(drop=True)], axis=1)
    elif(has_struct and not has_heuristic and has_domain) :  # struct + domain
        dataset = pd.concat([dataset_1.reset_index(drop=True), dataset_3.reset_index(drop=True)], axis=1)
    elif(not has_struct and has_heuristic and has_domain) :  # heuristic + domain
        dataset = pd.concat([dataset_2.reset_index(drop=True), dataset_3.reset_index(drop=True)], axis=1)
    else: # all or none
        dataset = dataset_tmp
        
    if 'label' not in dataset.columns.values:
        dataset = pd.concat([dataset_tmp[['label']].reset_index(drop=True), dataset.reset_index(drop=True)], axis=1)

    X = dataset.loc[:, dataset.columns != target_col].values
    y = dataset.loc[:, target_col].values
    dataset_col_names = np.delete(dataset.columns.values, 
                                  np.where(dataset.columns.values == "label"), axis=0)
    print(dataset_col_names)
    
    # splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = test_proportion, random_state = random_state)
    
    # feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    if(sampling_strategy is not None):
        print("%s sampling..." % sampling_strategy)
        if sampling_strategy == "under-NearMiss":
            clf = NearMiss(random_state=random_state)
        elif sampling_strategy == "under-RandomOverSampler":
            clf = RandomOverSampler(random_state=random_state)
        elif sampling_strategy == "over-SMOTE":
            clf = SMOTE(random_state=random_state)
        elif sampling_strategy == "over-RandomUnderSampler":
            clf = RandomUnderSampler(random_state=random_state)
        elif sampling_strategy == "combine-SMOTEENN":
            clf = SMOTEENN(random_state=random_state, ratio='minority')

        X_train, y_train = clf.fit_sample(X_train, y_train)
    
        
    return (X_train, X_test, y_train, y_test, dataset_col_names)


#####################################################################
### Run Random Forest model
#####################################################################
        
def run_random_forest(X_train, X_test, y_train, y_test, 
                      type_of_forest, n_estimators=100, max_features=3,
                      class_weight="balanced",
                      criterion='gini', bootstrap=True,
                      run_grid_search=False, search_grid={},
                      n_iter=10, cv=3, verbose=0, scoring="accuracy", 
                      random_state=0, n_jobs=None):
    print("in method: run_random_forest()")
    
    if(type_of_forest == "BalancedRandomForestClassifier"):
        classifier = BalancedRandomForestClassifier(
                n_estimators = n_estimators, 
                max_features = max_features,
                criterion=criterion, 
                bootstrap=bootstrap,
                random_state=random_state
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators = n_estimators, 
            max_features = max_features,
            criterion=criterion, 
            bootstrap=bootstrap,
            random_state=random_state,
            class_weight=class_weight
        )
    
    if(run_grid_search) :
        random_grid_search = RandomizedSearchCV(
                estimator=classifier, 
                param_distributions = search_grid,
                n_iter=n_iter, cv=cv, verbose=verbose, 
                scoring=scoring, random_state=random_state, n_jobs=n_jobs)
        
        fitted_results = random_grid_search.fit(X_train, y_train)
        best_score = fitted_results.best_score_
        print("best score: {}".format(best_score))
        best_params = fitted_results.best_params_
        print("best params: {}".format(best_params))
        
        if(type_of_forest == "BalancedRandomForestClassifier"):
            classifier = BalancedRandomForestClassifier(
                max_depth = best_params['max_depth'],
                max_features = best_params['max_features'],
                min_samples_leaf = best_params['min_samples_leaf'],
                min_samples_split = best_params['min_samples_split'],
                min_weight_fraction_leaf = best_params['min_weight_fraction_leaf'],
                n_estimators = best_params['n_estimators'],
                criterion=criterion, 
                bootstrap=bootstrap,
                random_state = random_state
            )
        else:
            classifier = RandomForestClassifier(
                max_depth = best_params['max_depth'],
                max_features = best_params['max_features'],
                min_samples_leaf = best_params['min_samples_leaf'],
                min_samples_split = best_params['min_samples_split'],
                min_weight_fraction_leaf = best_params['min_weight_fraction_leaf'],
                n_estimators = best_params['n_estimators'],
                criterion=criterion, 
                bootstrap=bootstrap,
                random_state = random_state
            )
    
    # fit with training data
    classifier.fit(X_train, y_train)
    
    # predicting the test set results
    y_pred = classifier.predict(X_test)
    
    # confusion matrix and accuracy
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(cm)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    F1 = 2 * recall * precision / (recall + precision)
    print("Accuracy: {} \nRecall: {} \nPrecision: {} \nF1: {}".format(
            accuracy, recall, precision, F1))
    print("ROC: {}".format(roc_auc_score(y_test, y_pred)))
    print("Classification report: \n{}".format(classification_report(y_test, y_pred)))
    print("MCC: {}".format(matthews_corrcoef(y_test, y_pred)))
    
    return classifier


#####################################################################
### Display Results
#####################################################################
def plot_feat_importance (classifier, col_names):
    print("in method: plot_feat_importance()")
    
    feat_imp = pd.DataFrame({'importance':classifier.feature_importances_})    
    feat_imp['feature'] = col_names
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title="Feature Importances",figsize=(8,8))
    plt.xlabel('Feature Importance Score')
    plt.show()

#####################################################################
### Create Twitter Data
#####################################################################
#if __name__ == "__main__":

start_time = time.time()
data = TwitterData("./data/twitter", "twitter_combined.txt")
data.load_struct_data(p=None, n=5000)

data.init_graph()
data.remove_low_degree_nodes()
data.remove_random_edges(0.1)
data.complement_pos_graph()
data.create_struct_feat()
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
data.create_heuristic_feat()
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
data.load_domain_data()
data.create_domain_feat()
print("--- %s seconds ---" % (time.time() - start_time))

## Write in and Read from Pickle File - uncomment when neeeded
#pickle_file = open('data_5000.pickle', 'wb')
#pickle.dump(data, pickle_file)
#pickle_file = open('data_5000.pickle', 'rb') 
#data1 = pickle.load(pickle_file)
#pickle_file.close()

#####################################################################
### Run Random Forest and Display Results
#####################################################################

def mcc_score(estimator, X_vals, y_val):
    return matthews_corrcoef(y_val, estimator.predict(X_vals))

common_random_state = 123
common_ds = data.struct_data_to_pred
common_test_proportion = 0.2
common_col_to_drop = ['from','to','from_id','to_id'] 
common_target_col = 'label'
common_sampling_strategy = None #"combine-SMOTEENN"
common_random_forest = "BalancedRandomForestClassifier"
common_run_grid_search = True
common_search_grid = {
        "n_estimators": [int(x) for x in np.linspace(start = 31, stop = 101, num = 10)], #for 100000 - 201, 20
        "max_depth": [int(x) for x in np.linspace(50, 250, num = 10)], #for 100000 - 500
        "max_features": ['auto', 'sqrt'],
        "min_samples_split": [int(x) for x in np.linspace(3, 15, num = 5)],
        "min_samples_leaf": [int(x) for x in np.linspace(1, 9, num = 3)],
        "min_weight_fraction_leaf": [0.05, 0.1, 0.15]
        } 
common_n_iter = 50
common_cv= 5
common_verbose = 2
common_scoring = "f1"
common_n_jobs = -1

#1. Structural (8)
start_time = time.time()
X_train_1, X_test_1, y_train_1, y_test_1, col_names_1 = massage_dataset(
        dataset=common_ds, test_proportion=common_test_proportion, 
        col_to_drop=common_col_to_drop, target_col=common_target_col,
        has_struct_partial=True, has_struct_full=False, has_heuristic=False, has_domain=False,
        sampling_strategy=common_sampling_strategy, random_state=common_random_state)
search_grid_1 = common_search_grid.copy()
search_grid_1["max_features"] = search_grid_1["max_features"] + [x for x in range(1,int(len(X_train_1[0])/2+1)) if x % 2 != 0]
classifier_1 = run_random_forest(
        X_train_1, X_test_1, y_train_1, y_test_1, common_random_forest,
        run_grid_search = common_run_grid_search, search_grid = search_grid_1,
        n_iter=common_n_iter, cv=common_cv, verbose=common_verbose, scoring=common_scoring, 
        random_state=common_random_state, n_jobs=common_n_jobs)
plot_feat_importance(classifier_1, col_names_1)
print("--- %s seconds ---" % (time.time() - start_time))

#2. Structural (16)
start_time = time.time()
X_train_2, X_test_2, y_train_2, y_test_2, col_names_2 = massage_dataset(
        dataset=common_ds, test_proportion=common_test_proportion, 
        col_to_drop=common_col_to_drop, target_col=common_target_col,
        has_struct_partial=False, has_struct_full=True, has_heuristic=False, has_domain=False,
        sampling_strategy=common_sampling_strategy, random_state=common_random_state)
search_grid_2 = common_search_grid.copy()
search_grid_2["max_features"] = search_grid_2["max_features"] + [x for x in range(1,int(len(X_train_2[0])/2+1)) if x % 2 != 0]
classifier_2 = run_random_forest(
        X_train_2, X_test_2, y_train_2, y_test_2, common_random_forest,
        run_grid_search = common_run_grid_search, search_grid = search_grid_2,
        n_iter=common_n_iter, cv=common_cv, verbose=common_verbose, scoring=common_scoring, 
        random_state=common_random_state, n_jobs=common_n_jobs)
plot_feat_importance(classifier_2, col_names_2)
print("--- %s seconds ---" % (time.time() - start_time))

#3. Heuristic (5)
start_time = time.time()
X_train_3, X_test_3, y_train_3, y_test_3, col_names_3 = massage_dataset(
        dataset=common_ds, test_proportion=common_test_proportion, 
        col_to_drop=common_col_to_drop, target_col=common_target_col,
        has_struct_partial=False, has_struct_full=False, has_heuristic=True, has_domain=False,
        sampling_strategy=common_sampling_strategy, random_state=common_random_state)
search_grid_3 = common_search_grid.copy()
search_grid_3["max_features"] = search_grid_3["max_features"] + [x for x in range(1,int(len(X_train_3[0])/2+1)) if x % 2 != 0]
classifier_3 = run_random_forest(
        X_train_3, X_test_3, y_train_3, y_test_3, common_random_forest,
        run_grid_search = common_run_grid_search, search_grid = search_grid_3,
        n_iter=common_n_iter, cv=common_cv, verbose=common_verbose, scoring=common_scoring, 
        random_state=common_random_state, n_jobs=common_n_jobs)
plot_feat_importance(classifier_3, col_names_3)
print("--- %s seconds ---" % (time.time() - start_time))

#4. Domain (2)
start_time = time.time()
X_train_4, X_test_4, y_train_4, y_test_4, col_names_4 = massage_dataset(
        dataset=common_ds, test_proportion=common_test_proportion, 
        col_to_drop=common_col_to_drop, target_col=common_target_col,
        has_struct_partial=False, has_struct_full=False, has_heuristic=False, has_domain=True,
        sampling_strategy=common_sampling_strategy, random_state=common_random_state)
search_grid_4 = common_search_grid.copy()
search_grid_4["max_features"] = search_grid_4["max_features"] + [x for x in range(1,int(len(X_train_4[0])/2+1)) if x % 2 != 0]
classifier_4 = run_random_forest(
        X_train_4, X_test_4, y_train_4, y_test_4, common_random_forest,
        run_grid_search = common_run_grid_search, search_grid = search_grid_4,
        n_iter=common_n_iter, cv=common_cv, verbose=common_verbose, scoring=common_scoring, 
        random_state=common_random_state, n_jobs=common_n_jobs)
plot_feat_importance(classifier_4, col_names_4)
print("--- %s seconds ---" % (time.time() - start_time))
    
#5. Structural (16) + Heuristic (5)
start_time = time.time()
X_train_5, X_test_5, y_train_5, y_test_5, col_names_5 = massage_dataset(
        dataset=common_ds, test_proportion=common_test_proportion, 
        col_to_drop=common_col_to_drop, target_col=common_target_col,
        has_struct_partial=True, has_struct_full=False, has_heuristic=True, has_domain=False,
        sampling_strategy=common_sampling_strategy, random_state=common_random_state)
search_grid_5 = common_search_grid.copy()
search_grid_5["max_features"] = search_grid_5["max_features"] + [x for x in range(1,int(len(X_train_5[0])/2+1)) if x % 2 != 0]
classifier_5 = run_random_forest(
        X_train_5, X_test_5, y_train_5, y_test_5, common_random_forest,
        run_grid_search = common_run_grid_search, search_grid = search_grid_5,
        n_iter=common_n_iter, cv=common_cv, verbose=common_verbose, scoring=common_scoring, 
        random_state=common_random_state, n_jobs=common_n_jobs)
plot_feat_importance(classifier_5, col_names_5)
print("--- %s seconds ---" % (time.time() - start_time))

#6. Structural (16) + Domain (2)
start_time = time.time()
X_train_6, X_test_6, y_train_6, y_test_6, col_names_6 = massage_dataset(
        dataset=common_ds, test_proportion=common_test_proportion, 
        col_to_drop=common_col_to_drop, target_col=common_target_col,
        has_struct_partial=False, has_struct_full=True, has_heuristic=False, has_domain=True,
        sampling_strategy=common_sampling_strategy, random_state=common_random_state)
search_grid_6 = common_search_grid.copy()
search_grid_6["max_features"] = search_grid_6["max_features"] + [x for x in range(1,int(len(X_train_6[0])/2+1)) if x % 2 != 0]
classifier_6 = run_random_forest(
        X_train_6, X_test_6, y_train_6, y_test_6, common_random_forest,
        run_grid_search = common_run_grid_search, search_grid = search_grid_6,
        n_iter=common_n_iter, cv=common_cv, verbose=common_verbose, scoring=common_scoring, 
        random_state=common_random_state, n_jobs=common_n_jobs)
plot_feat_importance(classifier_6, col_names_6)
print("--- %s seconds ---" % (time.time() - start_time))

#7. Heuristic (5) + Domain (2)
start_time = time.time()
X_train_7, X_test_7, y_train_7, y_test_7, col_names_7 = massage_dataset(
        dataset=common_ds, test_proportion=common_test_proportion, 
        col_to_drop=common_col_to_drop, target_col=common_target_col,
        has_struct_partial=False, has_struct_full=False, has_heuristic=True, has_domain=True,
        sampling_strategy=common_sampling_strategy, random_state=common_random_state)
search_grid_7 = common_search_grid.copy()
search_grid_7["max_features"] = search_grid_7["max_features"] + [x for x in range(1,int(len(X_train_7[0])/2+1)) if x % 2 != 0]
classifier_7 = run_random_forest(
        X_train_7, X_test_7, y_train_7, y_test_7, common_random_forest,
        run_grid_search = common_run_grid_search, search_grid = search_grid_7,
        n_iter=common_n_iter, cv=common_cv, verbose=common_verbose, scoring=common_scoring, 
        random_state=common_random_state, n_jobs=common_n_jobs)
plot_feat_importance(classifier_7, col_names_7)
print("--- %s seconds ---" % (time.time() - start_time))

#8. Structural (16) + Heuristic (5) + Domain (2)
start_time = time.time()
X_train_8, X_test_8, y_train_8, y_test_8, col_names_8 = massage_dataset(
        dataset=common_ds, test_proportion=common_test_proportion, 
        col_to_drop=common_col_to_drop, target_col=common_target_col,
        has_struct_partial=True, has_struct_full=True, has_heuristic=True, has_domain=True,
        sampling_strategy=common_sampling_strategy, random_state=common_random_state)
search_grid_8 = common_search_grid.copy()
search_grid_8["max_features"] = search_grid_8["max_features"] + [x for x in range(1,int(len(X_train_8[0])/2+1)) if x % 2 != 0]
classifier_8 = run_random_forest(
        X_train_8, X_test_8, y_train_8, y_test_8, common_random_forest,
        run_grid_search = common_run_grid_search, search_grid = search_grid_8,
        n_iter=common_n_iter, cv=common_cv, verbose=common_verbose, scoring=common_scoring, 
        random_state=common_random_state, n_jobs=common_n_jobs)
plot_feat_importance(classifier_8, col_names_8)
print("--- %s seconds ---" % (time.time() - start_time))


######################################################################
#### Plot other graphs
######################################################################
## randomly pull out one tree from the forest to visualise
#tree = classifier.estimators_[5]
#export_graphviz(tree, out_file = 'tree.dot', 
#                feature_names = dataset.columns[1:], 
#                rounded = True, precision = 1)
#(graph, ) = pydot.graph_from_dot_file('tree.dot')
#graph.write_png('tree.png')


# Network graph
plot(data.graph_struct_pos_orig)

# In degree distribution
in_distribution = data.graph_struct_pos_orig.degree_distribution(mode="in")
in_dist_data = []
for left, right, count in in_distribution.bins():
    in_dist_data.append([int(right), count])
in_dist_df = pd.DataFrame(data=in_dist_data, columns=["in_degree", "count"])
plt.figure(figsize=(10,3))
plt.plot(in_dist_df["in_degree"], in_dist_df["count"])
plt.xlabel('No of In-Degree')
plt.ylabel('Count')
plt.title('In-Degree Distribution')
plt.show()

# Out degree distribution
out_distribution = data.graph_struct_pos_orig.degree_distribution(mode="out")
out_dist_data = []
for left, right, count in out_distribution.bins():
    out_dist_data.append([int(right), count])
out_dist_df = pd.DataFrame(data=out_dist_data, columns=["out_degree", "count"])
plt.figure(figsize=(10,3))
plt.plot(out_dist_df["out_degree"], out_dist_df["count"])
plt.xlabel('No of Out-Degree')
plt.ylabel('Count')
plt.title('Out-Degree Distribution')
plt.show()

# Get no of degree by nodes
in_by_node = data.graph_struct_pos_orig.neighborhood_size(mode = "in")
node_degrees_df = pd.DataFrame(in_by_node, columns=["in_degree"])
node_degrees_df = node_degrees_df.reset_index().set_index('index', drop=False)
node_degrees_df.index.name = None
node_degrees_df[['index']] = node_degrees_df[['index']].astype(str)
out_by_node = data.graph_struct_pos_orig.neighborhood_size(mode = "out")
node_degrees_df = pd.concat([node_degrees_df, pd.DataFrame(out_by_node, columns=["out_degree"])], axis=1)

# In degree edges by node
node_degrees_df = node_degrees_df.sort_values(by=['in_degree'], ascending=False)
# In degree edges by node (All)
plt.figure(figsize=(10,3))
plt.plot(node_degrees_df["index"], node_degrees_df["in_degree"])
plt.xticks([])
plt.xlabel("Node ID")
plt.ylabel('Node Degree')
plt.title('Nodes In-Degree (All)')
plt.show()
# In degree edges by node (Top 20)
plt.figure(figsize=(10,3))
plt.bar(node_degrees_df.head(20)["index"], node_degrees_df.head(20)["in_degree"])
plt.xticks(fontsize=8)
plt.xlabel("Node ID")
plt.ylabel('Degree')
plt.title('Nodes In-Degree (Top 20)')
plt.show()

# Out degree edges by node
node_degrees_df = node_degrees_df.sort_values(by=['out_degree'], ascending=False)
# Out degree edges by node (All)
plt.figure(figsize=(10,3))
plt.plot(node_degrees_df["index"], node_degrees_df["out_degree"])
plt.xticks([])
plt.xlabel("Node ID")
plt.ylabel('Node Degree')
plt.title('Nodes Out-Degree (All)')
plt.show()
# Out degree edges by node (Top 20)
plt.figure(figsize=(10,3))
plt.bar(node_degrees_df.head(20)["index"], node_degrees_df.head(20)["out_degree"])
plt.xlabel("Node ID")
plt.ylabel('Degree')
plt.title('Nodes Out-Degree')
plt.show()

#in_by_node = data1.graph_struct_pos_orig.neighborhood_size(mode = "in")
#Some summary statistics
print(data.graph_struct_pos_orig.summary()) # (nodes, edges)
print(len(data.graph_struct_pos_orig.cliques(min=3,max=3))) # closed triangles
print(np.mean(in_by_node)) # avg no of followers/followees






