library(gprofiler2)
library(enrichR)
library(readxl)
library(Seurat)
library(hypeR)
library(tidyverse)
library(HGNChelper)
library(magrittr)
library(limma)
library(msigdb)
library(RCy3)
replace.genes<-function(genes){
  map<-checkGeneSymbols(genes)
  rownames(map)<-map$x
  genes<-sapply(genes,function(gene) (ifelse(is.na(map[gene,"Suggested.Symbol"]),gene,map[gene,"Suggested.Symbol"])))
  return(genes)
}
ebayes2em<-function(subtype="GBM",neg=F,sim=0.3,adj.p=0.05,n.clust=10,all=F,max_words=3,p.val=1,q.val=1,use.cyto=F){
  df<-read_excel("results_disco/wgcna.ebayes.zscore.xlsx", sheet = subtype)
  df$gene<-replace.genes(df$gene) # check this versus gene symbols and FDR threshold was not scanned
  m.factor<-1
  if (neg){
    m.factor<--1
  }
  if (all){
    m.factor<-0
  }
  signature <- df %>% filter(t*(m.factor) >= 0 & adj.P.Val < adj.p) %>% use_series(gene)
  gostres <- gost(query = signature, 
                  organism = "hsapiens", evcodes = TRUE, multi_query = FALSE, sources = c("GO","KEGG","REAC","WP","MIRNA","HPA","CORUM","HP"))#t > 0 &
  gostplot(gostres, capped = TRUE, interactive = TRUE)
  gem <- gostres$result[,c("term_id", "term_name", "p_value", "intersection")]
  colnames(gem) <- c("GO.ID", "Description", "p.Val", "Genes")
  gem$FDR <- gem$p.Val
  gem$Phenotype = "+1"
  gem <- gem[,c("GO.ID", "Description", "p.Val", "FDR", "Phenotype", "Genes")]
  gmt.out<-list(genesets=list(),geneset.names=c(),geneset.descriptions=c())
  for (i in 1:nrow(gem)){
    nm<-gem$GO.ID[i]
    gmt.out$geneset.names<-c(gmt.out$geneset.names,nm)
    gmt.out$geneset.descriptions<-c(gmt.out$geneset.descriptions,gem$Description[i])
    gmt.out$genesets[[nm]]<-unlist(strsplit2(gem$Genes[i],",")[1,])
  }
  write.gmt(gmt.out,'test.gmt')
  if (use.cyto){
  gmt.file<-"test.gmt"
  em_command <- paste('enrichmentmap build analysisType="generic"',
                      "gmtFile=", paste(getwd(), gmt.file, sep="/"),
                      "pvalue=", p.val,
                      "qvalue=", q.val,
                      "similaritycutoff=",sim,
                      "coefficients=","JACCARD")
  commandsGET(em_command)
  aa_command <- paste("autoannotate annotate-clusterBoosted",
                      "clusterAlgorithm=MCL",
                      "labelColumn=EnrichmentMap::GS_DESCR",
                      "maxWords=",max_words)
  commandsGET(aa_command)
  createSubnetwork(c(1:n.clust),"__mclCluster")
  commandsGET(aa_command)
  }
  
  return(gem)
}
