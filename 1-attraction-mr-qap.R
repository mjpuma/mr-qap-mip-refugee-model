warnings()

library(foreign)
library(igraph) ### Remember this conflicts with statnet & sna
library(tidyverse)
library(corrr)
library(ggraph)
library(dplyr)
library(rlang)
library(naniar)
library(statnet)
library(sna)
library(network)
library(cluster)    # clustering algorithms
library(factoextra) # clustering visualization
library(dendextend) # for comparing two dendrograms
library(ggplot2)
library(fmsb)
library(sjmisc)
library(prettyR)

#####################################################
# Matrix Creation
print("Matrix Creation")
#####################################################
edges <- read.csv("mr-qap/training.csv")
edges.test <- read.csv("mr-qap/test.csv")

igraph.edges <- graph_from_data_frame(edges, directed = T)
igraph.edges.test <- graph_from_data_frame(edges.test, directed = T)

######## TRAIN DATA #######
##########################
ref_pct_tot.mat <-
  as.matrix(as_adjacency_matrix(
    igraph.edges,
    type = "both",
    names = T,
    attr = "pct_tot"
  ))

gdp.mat <-
  as.matrix(as_adjacency_matrix(
    igraph.edges,
    type = "both",
    names = T,
    attr = "GDP.norm"
  ))

libdem.mat <-
  as.matrix(
    as_adjacency_matrix(
      igraph.edges,
      type = "both",
      names = T,
      attr = "v2x_libdem"
    )
  )

migrant.mat <-
  as.matrix(
    as_adjacency_matrix(
      igraph.edges,
      type = "both",
      names = T,
      attr = "International.migrant.stock....of.population._norm"
    )
  )

justice.mat <-
  as.matrix(
    as_adjacency_matrix(
      igraph.edges,
      type = "both",
      names = T,
      attr = "v2xeg_eqdr"
    )
  )

trade.mat <-
  as.matrix(
    as_adjacency_matrix(
      igraph.edges,
      type = "both",
      names = T,
      attr = "trade_norm"
    )
  )

pop.mat <-
  as.matrix(
    as_adjacency_matrix(
      igraph.edges,
      type = "both",
      names = T,
      attr = "Population..total_norm"
    )
  )


######## TEST DATA #######
##########################
test.ref_pct_tot.mat <-
  as.matrix(as_adjacency_matrix(
    igraph.edges.test,
    type = "both",
    names = T,
    attr = "pct_tot"
  ))

test.gdp.mat <-
  as.matrix(
    as_adjacency_matrix(
      igraph.edges.test,
      type = "both",
      names = T,
      attr = "GDP.norm"
    )
  )

test.libdem.mat <-
  as.matrix(
    as_adjacency_matrix(
      igraph.edges.test,
      type = "both",
      names = T,
      attr = "v2x_libdem"
    )
  )

test.migrant.mat <-
  as.matrix(
    as_adjacency_matrix(
      igraph.edges.test,
      type = "both",
      names = T,
      attr = "International.migrant.stock....of.population._norm"
    )
  )

test.justice.mat <-
  as.matrix(
    as_adjacency_matrix(
      igraph.edges.test,
      type = "both",
      names = T,
      attr = "v2xeg_eqdr"
    )
  )

test.trade.mat <-
  as.matrix(
    as_adjacency_matrix(
      igraph.edges.test,
      type = "both",
      names = T,
      attr = "trade_norm"
    )
  )

test.pop.mat <-
  as.matrix(
    as_adjacency_matrix(
      igraph.edges.test,
      type = "both",
      names = T,
      attr = "Population..total_norm"
    )
  )

#####################################################
# MRQAPs
print("Fit MRQAPs")
#####################################################

ref_lm <-
  netlm(
    ref_pct_tot.mat,
    # Dependent variable/network
    list(gdp.mat,
         libdem.mat),
#         migrant.mat,
#         justice.mat,
#         trade.mat),
    # List the independent variables/networks
    reps = 1000,
    nullhyp = "qapspp",
    test.statistic = "t-value"
  )


preds <- ref_lm$coefficients[1] +
  ref_lm$coefficients[2] * test.gdp.mat +
  ref_lm$coefficients[3] * test.libdem.mat
#  ref_lm$coefficients[4] * test.migrant.mat +
#  ref_lm$coefficients[5] * test.justice.mat + 
#  ref_lm$coefficients[6] * test.trade.mat

preds <- data.frame(preds[1, 2:8])
colnames(preds) <- c("predicted_shares")
preds <- cbind(country = rownames(preds), preds)
rownames(preds) <- 1:nrow(preds)
preds$predicted_shares <- preds$predicted_shares / sum(preds$predicted_shares)
print(preds)

write.csv(preds, file = 'mr-qap/mr-qap-results.csv', row.names=FALSE)