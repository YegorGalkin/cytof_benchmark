#!/usr/bin/env Rscript

if(!require(tidyverse)){
  install.packages("tidyverse")
  library(tidyverse)
}

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("flowCore")
library(flowCore)

if(!require(data.table)){
  install.packages("data.table")
  library(data.table)
}

args<-commandArgs(TRUE)

fcs_folder <- args[1]
output_dir <- args[2]

fcs_filenames <- list.files(fcs_folder)

fcss <- list.files(fcs_folder,full.names = T) %>%
  map(read.FCS)%>%
  map(~as.data.frame(.x@exprs))%>%
  map2_dfr(fcs_filenames,
       ~.x%>%
         mutate(cell_type=str_split_fixed(.y,'_',3)[,2])%>%
         mutate(day=str_remove(str_extract(.y,'[1-7].fcs'),'.fcs')))


pheno_data <- fcss%>%
  mutate(id=row_number())%>%
  select(id,cell_type,day)%>%
  as.data.table()

expression_data <- fcss %>% 
  select(1:41) %>%
  mutate(id=row_number())%>%
  select(id,everything())%>%
  as.data.table()

dir.create(file.path(output_dir,'data'))
dir.create(file.path(output_dir,'data','full'))
dir.create(file.path(output_dir,'data','reduced'))

expression_data %>%
  data.table::fwrite(file.path(output_dir,'data','full','data.csv.gz'))

pheno_data%>%
  data.table::fwrite(file.path(output_dir,'data','full','metadata.csv.gz'))

idx <- sample(1:nrow(expression_data), 10000)

expression_data%>%
  dplyr::filter(row_number() %in% idx)%>%
  data.table::fwrite(file.path(output_dir,'data','reduced','data_reduced.csv.gz'))

pheno_data%>%
  dplyr::filter(row_number() %in% idx)%>%
  data.table::fwrite(file.path(output_dir,'data','reduced','metadata_reduced.csv.gz'))
