
library(tidyverse, quietly = TRUE, warn.conflicts = FALSE) # for data wrangling
library(igraph, quietly = TRUE, warn.conflicts = FALSE)    # for network data structures and tools
library(ggraph, quietly = TRUE, warn.conflicts = FALSE)       # for network plotting

SEA_edges <- read.csv("NodesNEdges/SEA_2010-2015_edges_for_link_pred.csv")
CHN_edges <- read.csv("NodesNEdges/CHN_2010-2015_edges_for_link_pred.csv")

head(SEA_edges)
head(CHN_edges)

# timestamp:
### 2010-01-01: 1262304000
### 2011-01-01: 1293840000
### 2012-01-01: 1325376000
### 2013-01-01: 1356998400
### 2014-01-01: 1388534400
### 2015-01-01: 1420070400
### 2016-01-01: 1451606400

YEARS <- 6 #2010-2015
timestamp_from_2011 <- c(1293840000, 1325376000, 1356998400, 1388534400, 1420070400, 1451606400)
SEA_igraph <- list()
CHN_igraph <- list()
SEA_igraph_pos_df <- list()
SEA_igraph_neg_df <- list()
CHN_igraph_pos_df <- list()
CHN_igraph_neg_df <- list()

for(year in 1:YEARS) { 
#     print(paste0("#####", year, "#####"))
    # as the original data is in individual investment, we have to group by from and to nodes
    
    ##### for SEA graph and df #####
    SEA_igraph_pos_df[[year]] <- SEA_edges %>%
      filter(funded_at_ts<timestamp_from_2011[year]) %>%
      group_by(from, to, investor_type, company_origin, company_category_grp) %>%
      summarise(company_funding_to_date = max(company_funding_to_date),
                investor_funding_to_date = max(investor_funding_to_date)) %>%
      select(to, from, company_origin, company_category_grp, company_funding_to_date,
             investor_type, investor_funding_to_date) %>% as.data.frame
    SEA_igraph_pos_df[[year]]$label <- 1
    SEA_igraph[[year]] <- graph_from_data_frame(SEA_igraph_pos_df[[year]], directed=FALSE)
    V(SEA_igraph[[year]])$type <- bipartite_mapping(SEA_igraph[[year]])$type

    SEA_company_nodes <- SEA_igraph_pos_df[[year]] %>%
      group_by(to, company_origin, company_category_grp) %>%
      summarise(company_funding_to_date = max(company_funding_to_date)) %>% ungroup
    SEA_investor_nodes <- SEA_igraph_pos_df[[year]] %>%
      group_by(from, investor_type) %>%
      summarise(investor_funding_to_date = max(investor_funding_to_date)) %>% ungroup

    SEA_igraph_neg_df[[year]] <- expand.grid(to = SEA_igraph_pos_df[[year]]$to,
                                             from = SEA_igraph_pos_df[[year]]$from) %>%
      filter(to %in% SEA_company_nodes$to & from %in% SEA_investor_nodes$from) %>%
      merge(SEA_company_nodes, by="to", all.x=TRUE) %>%
      merge(SEA_investor_nodes, by="from", all.x=TRUE) %>%
      select(to, from, company_origin, company_category_grp, company_funding_to_date,
           investor_type, investor_funding_to_date) %>% as.data.frame
    SEA_igraph_neg_df[[year]]$label <- 0

    #the label is actually the ground truth prediction for the next year, so we update the prev year
    if(year != 1) {
        SEA_igraph_neg_df[[year-1]] <- merge(SEA_igraph_pos_df[[year]] %>% select(to, from, label),
                                             SEA_igraph_neg_df[[year-1]] %>% select(-label),
                                             by = c("to", "from") ,
                                             all.y = TRUE) 
        SEA_igraph_neg_df[[year-1]]$label[is.na(SEA_igraph_neg_df[[year-1]]$label)] <- 0
    }

    ##### for CHN graph and df #####
    CHN_igraph_pos_df[[year]] <- CHN_edges %>%
      filter(funded_at_ts<timestamp_from_2011[year]) %>%
      group_by(from, to, investor_type, company_origin, company_category_grp) %>%
      summarise(company_funding_to_date = max(company_funding_to_date),
                investor_funding_to_date = max(investor_funding_to_date)) %>%
      select(to, from, company_origin, company_category_grp, company_funding_to_date,
             investor_type, investor_funding_to_date) %>% as.data.frame
    CHN_igraph_pos_df[[year]]$label <- 1
    CHN_igraph[[year]] <- graph_from_data_frame(CHN_igraph_pos_df[[year]], directed=FALSE)
    V(CHN_igraph[[year]])$type <- bipartite_mapping(CHN_igraph[[year]])$type

    CHN_company_nodes <- CHN_igraph_pos_df[[year]] %>%
      group_by(to, company_origin, company_category_grp) %>%
      summarise(company_funding_to_date = max(company_funding_to_date)) %>% ungroup
    CHN_investor_nodes <- CHN_igraph_pos_df[[year]] %>%
      group_by(from, investor_type) %>%
      summarise(investor_funding_to_date = max(investor_funding_to_date)) %>% ungroup

    CHN_igraph_neg_df[[year]] <- expand.grid(to = CHN_igraph_pos_df[[year]]$to,
                                             from = CHN_igraph_pos_df[[year]]$from) %>%
      filter(to %in% CHN_company_nodes$to & from %in% CHN_investor_nodes$from) %>%
      merge(CHN_company_nodes, by="to", all.x=TRUE) %>%
      merge(CHN_investor_nodes, by="from", all.x=TRUE) %>%
      select(to, from, company_origin, company_category_grp, company_funding_to_date,
           investor_type, investor_funding_to_date) %>% as.data.frame
    CHN_igraph_neg_df[[year]]$label <- 0

    #the label is actually the ground truth prediction for the next year, so we update the prev year
    if(year != 1) {
      CHN_igraph_neg_df[[year-1]] <- merge(CHN_igraph_pos_df[[year]] %>% select(to, from, label),
                                                 CHN_igraph_neg_df[[year-1]] %>% select(-label),
                                                 by = c("to", "from") ,
                                                 all.y = TRUE)
      CHN_igraph_neg_df[[year-1]]$label[is.na(CHN_igraph_neg_df[[year-1]]$label)] <- 0
    }
}

# create combined list
SEA_igraph_combi_df <- list()
CHN_igraph_combi_df <- list()
for(year in 1:YEARS) { 
    SEA_igraph_combi_df[[year]] <- rbind(SEA_igraph_pos_df[[year]], 
                                         SEA_igraph_neg_df[[year]]) %>% 
                                    mutate(label = factor(label))
    CHN_igraph_combi_df[[year]] <- rbind(CHN_igraph_pos_df[[year]], 
                                         CHN_igraph_neg_df[[year]]) %>% 
                                    mutate(label = factor(label))
}

nrow(SEA_igraph_combi_df[[5]])
nrow(CHN_igraph_combi_df[[5]])

# below are what we are supposed to predict out
SEA_igraph_neg_df[[5]] %>% filter(label==1)
CHN_igraph_neg_df[[5]] %>% filter(label==1)
nrow(SEA_igraph_neg_df[[5]] %>% filter(label==1))
nrow(CHN_igraph_neg_df[[5]] %>% filter(label==1))

# read in files for company network and investor network
# SEA
SEA_company_edges_till_2014 <- read.csv("NodesNEdges/company_SEA_2010-2014_edge_list.csv")
SEA_company_nodes_till_2014 <- read.csv("NodesNEdges/company_SEA_2010-2014_node_list.csv")
SEA_investor_edges_till_2014 <- read.csv("NodesNEdges/investor_SEA_2010-2014_edge_list.csv")
SEA_investor_nodes_till_2014 <- read.csv("NodesNEdges/investor_SEA_2010-2014_node_list.csv")
# CHN
CHN_company_edges_till_2014 <- read.csv("NodesNEdges/company_CHN_2010-2014_edge_list.csv")
CHN_company_nodes_till_2014 <- read.csv("NodesNEdges/company_CHN_2010-2014_node_list.csv")
CHN_investor_edges_till_2014 <- read.csv("NodesNEdges/investor_CHN_2010-2014_edge_list.csv")
CHN_investor_nodes_till_2014 <- read.csv("NodesNEdges/investor_CHN_2010-2014_node_list.csv")

######################################################################################
# Below we obtain some of the metrics using the projection of 
# bipartite graph to company network and investor network
######################################################################################

## Get community membership ID of the nodes
getMembershipID <- function(company_graph, investor_graph, df) {
  company_wc = cluster_walktrap(company_graph)
  company_wc_size <- company_wc %>% sizes() %>% enframe() %>%
    rename(company_comm_id=name, member_size=value) %>% mutate(rank = rank(desc(member_size), ties.method = "first"))
  
  company_comm_id <- membership(company_wc) %>% enframe() %>% rename(company_comm_id=value)
  company_comm_id <- merge(company_comm_id, company_wc_size, by="company_comm_id", all.x=TRUE) %>% 
    mutate(company_comm_id_final = factor(ifelse(rank<=10, company_comm_id, 0))) %>% 
    select(-company_comm_id, -member_size, -rank) %>% rename(company_comm_id=company_comm_id_final)
  
  investor_wc = cluster_walktrap(investor_graph)
  investor_wc_size <- investor_wc %>% sizes() %>% enframe() %>% 
    rename(investor_comm_id=name, member_size=value)%>% mutate(rank = rank(desc(member_size), ties.method = "first"))
  
  investor_comm_id <- membership(investor_wc) %>% enframe() %>% rename(investor_comm_id=value)
  investor_comm_id <- merge(investor_comm_id, investor_wc_size, by="investor_comm_id", all.x=TRUE) %>% 
    mutate(investor_comm_id_final = factor(ifelse(rank<=10, investor_comm_id, 0))
    ) %>% select(-investor_comm_id, -member_size, -rank) %>% rename(investor_comm_id=investor_comm_id_final)
  
  df <- df %>% 
    merge(company_comm_id, by.x="to", by.y="name", all.x=TRUE) %>%
    merge(investor_comm_id, by.x="from", by.y="name", all.x=TRUE)
  return (df)
}

## Get degree of the nodes
getDegree <- function(company_graph, investor_graph, df) {
  company_degree <- degree(company_graph)
  company_degree <- enframe(company_degree)
  investor_degree <- degree(investor_graph)
  investor_degree <- enframe(investor_degree)
  
  df <- df %>% 
    merge(company_degree, by.x="to", by.y="name", all.x=TRUE) %>% rename(company_degree=value) %>%
    merge(investor_degree, by.x="from", by.y="name", all.x=TRUE) %>% rename(investor_degree=value)
  return (df)
}

## Get betweenness of the nodes
getBetweenness <- function(company_graph, investor_graph, df) {
  company_betweenness <- betweenness(company_graph, weights=1/E(company_graph)$weight)
  company_betweenness <- enframe(company_betweenness)
  investor_betweenness <- betweenness(investor_graph, weights=1/E(investor_graph)$weight)
  investor_betweenness <- enframe(investor_betweenness)
  
  df <- df %>% 
    merge(company_betweenness, by.x="to", by.y="name", all.x=TRUE) %>% rename(company_betweenness=value) %>%
    merge(investor_betweenness, by.x="from", by.y="name", all.x=TRUE) %>% rename(investor_betweenness=value)
  return (df)
}

## Get closeness of the nodes
getCloseness <- function(company_graph, investor_graph, df) {
  company_closeness <- closeness(company_graph, weights=1/E(company_graph)$weight)
  company_closeness <- enframe(company_closeness)
  investor_closeness <- closeness(investor_graph, weights=1/E(investor_graph)$weight)
  investor_closeness <- enframe(investor_closeness)
  
  df <- df %>% 
    merge(company_closeness, by.x="to", by.y="name", all.x=TRUE) %>% rename(company_closeness=value) %>%
    merge(investor_closeness, by.x="from", by.y="name", all.x=TRUE) %>% rename(investor_closeness=value)
  return (df)
}

## Get clustering coefficient of the nodes
getClustCoeff <- function(company_graph, investor_graph, df) {
  company_cluster_coef <- transitivity(company_graph, type="local", isolates="zero")
  names(company_cluster_coef) <- as_ids(V(company_graph))
  company_cluster_coef <- enframe(company_cluster_coef)
  investor_cluster_coef <- transitivity(investor_graph, type="local", isolates="zero")
  names(investor_cluster_coef) <- as_ids(V(investor_graph))
  investor_cluster_coef <- enframe(investor_cluster_coef)
  
  df <- df %>% 
    merge(company_cluster_coef, by.x="to", by.y="name", all.x=TRUE) %>% rename(company_cluster_coef=value) %>%
    merge(investor_cluster_coef, by.x="from", by.y="name", all.x=TRUE) %>% rename(investor_cluster_coef=value)
  return (df)
}

## Get page rank of the nodes
getPageRank <- function(company_graph, investor_graph, df) {
  company_page_rank <- page_rank(company_graph)$vector
  company_page_rank <- enframe(company_page_rank)
  investor_page_rank <- page_rank(investor_graph)$vector
  investor_page_rank <- enframe(investor_page_rank)
  
  df <- df %>% 
    merge(company_page_rank, by.x="to", by.y="name", all.x=TRUE) %>% rename(company_page_rank=value) %>%
    merge(investor_page_rank, by.x="from", by.y="name", all.x=TRUE) %>% rename(investor_page_rank=value)
  return (df)
}

## Get eccentricity of the nodes
getEccentricity <- function(company_graph, investor_graph, df) { # no weights parameter
  company_eccentricity <- eccentricity(company_graph)
  company_eccentricity <- enframe(company_eccentricity)
  investor_eccentricity <- eccentricity(investor_graph)
  investor_eccentricity <- enframe(investor_eccentricity)
  
  df <- df %>% 
    merge(company_eccentricity, by.x="to", by.y="name", all.x=TRUE) %>% rename(company_eccentricity=value) %>%
    merge(investor_eccentricity, by.x="from", by.y="name", all.x=TRUE) %>% rename(investor_eccentricity=value)
  return (df)
}

# use up to 2014 to predict 2015
#SEA
SEA_company_community_2014_ig <- graph_from_data_frame(
  SEA_company_edges_till_2014, directed=FALSE, vertices=SEA_company_nodes_till_2014)
SEA_investor_community_2014_ig <- graph_from_data_frame(
  SEA_investor_edges_till_2014, directed=FALSE, vertices=SEA_investor_nodes_till_2014)
SEA_data_df <- getMembershipID(SEA_company_community_2014_ig, SEA_investor_community_2014_ig, SEA_igraph_combi_df[[5]])
SEA_data_df <- getDegree(SEA_company_community_2014_ig, SEA_investor_community_2014_ig, SEA_data_df)
SEA_data_df <- getBetweenness(SEA_company_community_2014_ig, SEA_investor_community_2014_ig, SEA_data_df)
SEA_data_df <- getCloseness(SEA_company_community_2014_ig, SEA_investor_community_2014_ig, SEA_data_df)
SEA_data_df <- getClustCoeff(SEA_company_community_2014_ig, SEA_investor_community_2014_ig, SEA_data_df)
SEA_data_df <- getPageRank(SEA_company_community_2014_ig, SEA_investor_community_2014_ig, SEA_data_df)
SEA_data_df <- getEccentricity(SEA_company_community_2014_ig, SEA_investor_community_2014_ig, SEA_data_df)

# check na records
SEA_data_df %>%
  mutate_all(is.na) %>%
  gather(key = "variable_name", value="is_na_count") %>% 
  group_by(variable_name) %>% 
  summarise(
    is_na_count=sum(is_na_count), 
    percentage = sum(is_na_count) / nrow(SEA_data_df) * 100) %>%
  arrange(desc(is_na_count))

#CHN
CHN_company_community_2014_ig <- graph_from_data_frame(
  CHN_company_edges_till_2014, directed=FALSE, vertices=CHN_company_nodes_till_2014)
CHN_investor_community_2014_ig <- graph_from_data_frame(
  CHN_investor_edges_till_2014, directed=FALSE, vertices=CHN_investor_nodes_till_2014)
CHN_data_df <- getMembershipID(CHN_company_community_2014_ig, CHN_investor_community_2014_ig, CHN_igraph_combi_df[[5]])
CHN_data_df <- getDegree(CHN_company_community_2014_ig, CHN_investor_community_2014_ig, CHN_data_df)
CHN_data_df <- getBetweenness(CHN_company_community_2014_ig, CHN_investor_community_2014_ig, CHN_data_df)
CHN_data_df <- getCloseness(CHN_company_community_2014_ig, CHN_investor_community_2014_ig, CHN_data_df)
CHN_data_df <- getClustCoeff(CHN_company_community_2014_ig, CHN_investor_community_2014_ig, CHN_data_df)
CHN_data_df <- getPageRank(CHN_company_community_2014_ig, CHN_investor_community_2014_ig, CHN_data_df)
CHN_data_df <- getEccentricity(CHN_company_community_2014_ig,CHN_investor_community_2014_ig, CHN_data_df)

# check na records
CHN_data_df %>%
  mutate_all(is.na) %>%
  gather(key = "variable_name", value="is_na_count") %>% 
  group_by(variable_name) %>% 
  summarise(
    is_na_count=sum(is_na_count), 
    percentage = sum(is_na_count) / nrow(CHN_data_df) * 100) %>%
  arrange(desc(is_na_count))

write.csv(SEA_data_df, "SEA_data_df_v4.csv", row.names = F)
write.csv(CHN_data_df, "CHN_data_df_v4.csv", row.names = F)

SEA_data_df <- read.csv("SEA_data_df_v4.csv")
SEA_data_df$label <- factor(SEA_data_df$label)
SEA_data_df$company_comm_id <- factor(SEA_data_df$company_comm_id)
SEA_data_df$investor_comm_id <- factor(SEA_data_df$investor_comm_id)
CHN_data_df <- read.csv("CHN_data_df_v4.csv")
CHN_data_df$label <- factor(CHN_data_df$label)
CHN_data_df$company_comm_id <- factor(CHN_data_df$company_comm_id)
CHN_data_df$investor_comm_id <- factor(CHN_data_df$investor_comm_id)

str(SEA_data_df)
str(CHN_data_df)
head(SEA_data_df)
head(CHN_data_df)

######################################################################################
# Machine Learning
######################################################################################
library(randomForest) # for random forest
library(ROCR) 
library(precrec) 

acc <- function(predicted, actual) {
  return(mean(predicted == actual))
}

sample_size_SEA = nrow(SEA_data_df)
set.seed(1234)
train_ids_SEA <- sample(1:sample_size_SEA, 0.7*sample_size_SEA)

sample_size_CHN = nrow(CHN_data_df)
set.seed(1234)
train_ids_CHN <- sample(1:sample_size_CHN, 0.5*sample_size_CHN)

#50-50 for random forest cos of error "long vectors (argument 26) are not supported in .C"

# removed all network features 
train_data_SEA_base <- subset(SEA_data_df[train_ids_SEA, ], select=-c(from, to, company_comm_id:investor_eccentricity))
test_data_SEA_base <- subset(SEA_data_df[-train_ids_SEA, ], select=-c(from, to, company_comm_id:investor_eccentricity))

# removed all network features 
train_data_CHN_base <- subset(CHN_data_df[train_ids_CHN, ], select=-c(from, to, company_comm_id:investor_eccentricity))
test_data_CHN_base <- subset(CHN_data_df[-train_ids_CHN, ], select=-c(from, to, company_comm_id:investor_eccentricity))

train_data_SEA <- subset(SEA_data_df[train_ids_SEA, ], select=-c(from, to))
test_data_SEA <- subset(SEA_data_df[-train_ids_SEA, ], select=-c(from, to))

train_data_CHN <- subset(CHN_data_df[train_ids_CHN, ], select=-c(from, to))
test_data_CHN <- subset(CHN_data_df[-train_ids_CHN, ], select=-c(from, to))

######################################################################################
# Machine Learning - Random Forest
######################################################################################

############ Baseline without Network Features ############

###############
####SEA...#####
###############

ptm <- proc.time()

set.seed(1234)
model.rf_SEA_base <- randomForest(label ~ ., data=train_data_SEA_base, mtry=3, importance=TRUE)
model.rf_SEA_base
importance(model.rf_SEA_base)
varImpPlot(model.rf_SEA_base)

# accuracy
model.rf.predict_SEA_base <- predict(model.rf_SEA_base, subset(test_data_SEA_base, select=-label), type="prob")[,"1"]
cat('Random Forest Accuracy:', acc(ifelse(model.rf.predict_SEA_base>=0.5,1,0), test_data_SEA_base$label), '\n')

# misclassification error in test data
model.rf.predict_SEA_resp_base <- predict(model.rf_SEA_base, subset(test_data_SEA_base, select=-label), type="response")
table(model.rf.predict_SEA_resp_base, test_data_SEA_base$label)

# Calculate ROC & AUC of ROC curve
model.rf.prediction_SEA_base <- ROCR::prediction(model.rf.predict_SEA_base, test_data_SEA_base$label)
rf.roc_SEA_base <- performance(model.rf.prediction_SEA_base, measure="tpr", x.measure="fpr")
plot(rf.roc_SEA_base)

rf.auc_SEA_base <- performance(model.rf.prediction_SEA_base, measure="auc")
cat("Random Forest AUC: ", rf.auc_SEA_base@y.values[[1]], '\n')

# Calculate PRC & AUPR curve
rf.prc_SEA_base <- performance(model.rf.prediction_SEA_base, measure="prec", x.measure="rec")
plot(rf.prc_SEA_base)

rf.prc_SEA_obj_base <- mmdata(as.numeric(model.rf.predict_SEA_base),test_data_SEA_base$label)
rf_SEA_perf_base <- evalmod(mdat = rf.prc_SEA_obj_base) 
rf_SEA_perf_base
proc.time() - ptm

###############
####CHN...#####
###############

ptm <- proc.time()

set.seed(1234)
model.rf_CHN_base <- randomForest(label ~ ., data=train_data_CHN_base, mtry=3, importance=TRUE)
model.rf_CHN_base
importance(model.rf_CHN_base)
varImpPlot(model.rf_CHN_base)

# accuracy
model.rf.predict_CHN_base <- predict(model.rf_CHN_base, subset(test_data_CHN_base, select=-label), type="prob")[,"1"]
cat('Random Forest Accuracy:', acc(ifelse(model.rf.predict_CHN_base>=0.5,1,0), test_data_CHN_base$label), '\n')

# misclassification error in test data
model.rf.predict_CHN_resp_base <- predict(model.rf_CHN_base, subset(test_data_CHN_base, select=-label), type="response")
table(model.rf.predict_CHN_resp_base, test_data_CHN_base$label)

# Calculate ROC & AUC of ROC curve
model.rf.prediction_CHN_base <- ROCR::prediction(model.rf.predict_CHN_base, test_data_CHN_base$label)
rf.roc_CHN_base <- performance(model.rf.prediction_CHN_base, measure="tpr", x.measure="fpr")
plot(rf.roc_CHN_base)

rf.auc_CHN_base <- performance(model.rf.prediction_CHN_base, measure="auc")
cat("Random Forest AUC: ", rf.auc_CHN_base@y.values[[1]], '\n')

# Calculate PRC & AUPR curve
rf.prc_CHN_base <- performance(model.rf.prediction_CHN_base, measure="prec", x.measure="rec")
plot(rf.prc_CHN_base)

rf.prc_CHN_obj_base <- mmdata(as.numeric(model.rf.predict_CHN_base),test_data_CHN_base$label)
rf_CHN_perf_base <- evalmod(mdat = rf.prc_CHN_obj_base) 
rf_CHN_perf_base
proc.time() - ptm

############ Include Network Features ############

###############
####SEA...#####
###############

ptm <- proc.time()

set.seed(1234)
model.rf_SEA <- randomForest(label ~ ., data=train_data_SEA, mtry=3, importance=TRUE)
model.rf_SEA
importance(model.rf_SEA)
varImpPlot(model.rf_SEA)

# accuracy
model.rf.predict_SEA <- predict(model.rf_SEA, subset(test_data_SEA, select=-label), type="prob")[,"1"]
cat('Random Forest Accuracy:', acc(ifelse(model.rf.predict_SEA>=0.5,1,0), test_data_SEA$label), '\n')

# misclassification error in test data
model.rf.predict_SEA_resp <- predict(model.rf_SEA, subset(test_data_SEA, select=-label), type="response")
table(model.rf.predict_SEA_resp, test_data_SEA$label)

# Calculate ROC & AUC of ROC curve
model.rf.prediction_SEA <- ROCR::prediction(model.rf.predict_SEA, test_data_SEA$label)
rf.roc_SEA <- performance(model.rf.prediction_SEA, measure="tpr", x.measure="fpr")
plot(rf.roc_SEA)

rf.auc_SEA <- performance(model.rf.prediction_SEA, measure="auc")
cat("Random Forest AUC: ", rf.auc_SEA@y.values[[1]], '\n')

# Calculate PRC & AUPR curve
rf.prc_SEA <- performance(model.rf.prediction_SEA, measure="prec", x.measure="rec")
plot(rf.prc_SEA)

rf.prc_SEA_obj <- mmdata(as.numeric(model.rf.predict_SEA),test_data_SEA$label)
rf_SEA_perf <- evalmod(mdat = rf.prc_SEA_obj) 
rf_SEA_perf
proc.time() - ptm

# partial dependency plots
ptm1 <- proc.time()
partialPlot(model.rf_SEA, train_data_SEA, x.var="company_closeness", which.class="1")
proc.time() - ptm1

ptm1 <- proc.time()
partialPlot(model.rf_SEA, train_data_SEA, x.var="company_comm_id", which.class="1")
proc.time() - ptm1

###############
####CHN...#####
###############

ptm <- proc.time()

set.seed(1234)
model.rf_CHN <- randomForest(label ~ ., data=train_data_CHN, mtry=3, importance=TRUE)
model.rf_CHN
importance(model.rf_CHN)
varImpPlot(model.rf_CHN)

# accuracy
model.rf.predict_CHN <- predict(model.rf_CHN, subset(test_data_CHN, select=-label), type="prob")[,"1"]
cat('Random Forest Accuracy:', acc(ifelse(model.rf.predict_CHN>=0.5,1,0), test_data_CHN$label), '\n')

# misclassification error in test data
model.rf.predict_CHN_resp <- predict(model.rf_CHN, subset(test_data_CHN, select=-label), type="response")
table(model.rf.predict_CHN_resp, test_data_CHN$label)

# Calculate ROC & AUC of ROC curve
model.rf.prediction_CHN <- ROCR::prediction(model.rf.predict_CHN, test_data_CHN$label)
rf.roc_CHN <- performance(model.rf.prediction_CHN, measure="tpr", x.measure="fpr")
plot(rf.roc_CHN)

rf.auc_CHN <- performance(model.rf.prediction_CHN, measure="auc")
cat("Random Forest AUC: ", rf.auc_CHN@y.values[[1]], '\n')

# Calculate PRC & AUPR curve
rf.prc_CHN <- performance(model.rf.prediction_CHN, measure="prec", x.measure="rec")
plot(rf.prc_CHN)

rf.prc_CHN_obj <- mmdata(as.numeric(model.rf.predict_CHN),test_data_CHN$label)
rf_CHN_perf <- evalmod(mdat = rf.prc_CHN_obj) 
rf_CHN_perf
proc.time() - ptm

# partial dependency plots
ptm1 <- proc.time()
partialPlot(model.rf_CHN, train_data_CHN, x.var="company_cluster_coef", which.class="1")
proc.time() - ptm1

ptm1 <- proc.time()
partialPlot(model.rf_CHN, train_data_CHN, x.var="company_comm_id", which.class="1")
proc.time() - ptm1

ptm1 <- proc.time()
partialPlot(model.rf_CHN, train_data_CHN, x.var="company_degree", which.class="1")
proc.time() - ptm1

######################################################################################
# Machine Learning - Logistic Regression
######################################################################################

############ Baseline without Network Features ############

##SEA glm baseline
set.seed(123)
model.lr_SEA_base<-glm(label ~ ., data=train_data_SEA_base, family = binomial)
model.lr_SEA_prob_base <- predict(model.lr_SEA_base,test_data_SEA_base, type = "response")
model.lr_SEA_pred_base <- ifelse(model.lr_SEA_prob_base>0.5, "1", "0")
model.lr_SEA_prediction_base <- ROCR::prediction(model.lr_SEA_prob_base, test_data_SEA_base$label)
#Accuracy
model.lr_SEA_accuracy_base <-mean(model.lr_SEA_pred_base == test_data_SEA_base$label)#0.
cat("GLM baseline SEA Accuracy: ", model.lr_SEA_accuracy_base , '\n')
table(model.lr_SEA_pred_base, test_data_SEA_base$label)
#AUC
lr.auc_SEA_base <- ROCR::performance(model.lr_SEA_prediction_base, measure="auc")
cat("GLM with network features SEA AUC: ", lr.auc_SEA_base@y.values[[1]], '\n')
#AUCPR
lr.prc_SEA_obj_base <- precrec::mmdata(as.numeric(model.lr_SEA_prob_base),test_data_SEA_base$label)
lr_SEA_perf_base<-precrec::evalmod(mdat = lr.prc_SEA_obj_base) 
lr_SEA_perf_base
#ROC curve
lr.roc_SEA_base <- ROCR::performance(model.lr_SEA_prediction_base, measure="tpr", x.measure="fpr")
plot(lr.roc_SEA_base)
#AUCPR curve
lr.aucpr_SEA_base <- ROCR::performance(model.lr_SEA_prediction_base, measure="prec", x.measure="rec")
plot(lr.aucpr_SEA_base )

##CHN glm baseline
set.seed(123)
model.lr_CHN_base<-glm(label ~ ., data=train_data_CHN_base, family = binomial)
model.lr_CHN_prob_base <- predict(model.lr_CHN_base,test_data_CHN_base, type = "response")
model.lr_CHN_pred_base <- ifelse(model.lr_CHN_prob_base>0.5, "1", "0")
model.lr_CHN_prediction_base <- ROCR::prediction(model.lr_CHN_prob_base, test_data_CHN_base$label)
#Accuracy
model.lr_CHN_accuracy_base <-mean(model.lr_CHN_pred_base == test_data_CHN_base$label)#0.
cat("GLM baseline CHN Accuracy: ", model.lr_CHN_accuracy_base , '\n')
table(model.lr_CHN_pred_base, test_data_CHN_base$label)
#AUC
lr.auc_CHN_base <- ROCR::performance(model.lr_CHN_prediction_base, measure="auc")
cat("GLM with network features CHN AUC: ", lr.auc_CHN_base@y.values[[1]], '\n')
#AUCPR
lr.prc_CHN_obj_base <- precrec::mmdata(as.numeric(model.lr_CHN_prob_base),test_data_CHN_base$label)
lr_CHN_perf_base<-precrec::evalmod(mdat = lr.prc_CHN_obj_base) 
lr_CHN_perf_base
#ROC curve
lr.roc_CHN_base <- ROCR::performance(model.lr_CHN_prediction_base, measure="tpr", x.measure="fpr")
plot(lr.roc_CHN_base)
#AUCPR curve
lr.aucpr_CHN_base <- ROCR::performance(model.lr_CHN_prediction_base, measure="prec", x.measure="rec")
plot(lr.aucpr_CHN_base )

############ Include Network Features ############

##SEA glm with features
set.seed(123)
model.lr_SEA<-glm(label ~ ., data=train_data_SEA, family = binomial)
model.lr_SEA_prob <- model.lr_SEA %>% predict(test_data_SEA, type = "response")
model.lr_SEA_pred <- ifelse(model.lr_SEA_prob > 0.5, "1", "0")
model.lr_SEA_prediction <- ROCR::prediction(model.lr_SEA_prob, test_data_SEA$label)
#Accuracy
model.lr_SEA_accuracy <-mean(model.lr_SEA_pred == test_data_SEA$label) #0.9791418
cat("GLM with network features SEA Accuracy: ", model.lr_SEA_accuracy , '\n')
table(model.lr_SEA_pred, test_data_SEA$label)
#AUC
lr.auc_SEA <- ROCR::performance(model.lr_SEA_prediction, measure="auc")
cat("GLM with network features SEA AUC: ", lr.auc_SEA@y.values[[1]], '\n')
#AUCPR
lr.prc_SEA_obj <- precrec::mmdata(as.numeric(model.lr_SEA_prob),test_data_SEA$label)
lr_SEA_perf<-precrec::evalmod(mdat = lr.prc_SEA_obj) 
lr_SEA_perf
#ROC curve
lr.roc_SEA <- ROCR::performance(model.lr_SEA_prediction, measure="tpr", x.measure="fpr")
plot(lr.roc_SEA)
#AUCPR curve
lr.aucpr_SEA <- ROCR::performance(model.lr_SEA_prediction, measure="prec", x.measure="rec")
plot(lr.aucpr_SEA )

##CHN glm with features
set.seed(123)
model.lr_CHN<-glm(label ~ ., data=train_data_CHN, family = binomial)
model.lr_CHN_prob <- predict(model.lr_CHN,test_data_CHN, type = "response")
model.lr_CHN_pred <- ifelse(model.lr_CHN_prob>0.5, "1", "0")
model.lr_CHN_prediction <- ROCR::prediction(model.lr_CHN_prob, test_data_CHN$label)
#Accuracy
model.lr_CHN_accuracy <-mean(model.lr_CHN_pred == test_data_CHN$label)#0.9683291
cat("GLM with network features CHN Accuracy: ", model.lr_CHN_accuracy , '\n')
table(model.lr_CHN_pred, test_data_CHN$label)
#AUC
lr.auc_CHN <- ROCR::performance(model.lr_CHN_prediction, measure="auc")
cat("GLM with network features CHN AUC: ", lr.auc_CHN@y.values[[1]], '\n')
#AUCPR
lr.prc_CHN_obj <- precrec::mmdata(as.numeric(model.lr_CHN_prob),test_data_CHN$label)
lr_CHN_perf<-precrec::evalmod(mdat = lr.prc_CHN_obj) 
lr_CHN_perf
#ROC curve
lr.roc_CHN <- ROCR::performance(model.lr_CHN_prediction, measure="tpr", x.measure="fpr")
plot(lr.roc_CHN)
#AUCPR curve
lr.aucpr_CHN <- ROCR::performance(model.lr_CHN_prediction, measure="prec", x.measure="rec")
plot(lr.aucpr_CHN )

