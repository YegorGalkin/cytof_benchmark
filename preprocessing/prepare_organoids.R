library(tidyverse)
library(flowCore)
library(data.table)

setwd('D:/Rstuff/organoids')

fcs_folder <- './experiment_83654_20221005115016593_files'

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

dir.create('./results')
dir.create('./results/full/')
dir.create('./results/reduced/')

expression_data %>%
  data.table::fwrite('results/full/data.csv.gz')

pheno_data%>%
  data.table::fwrite('results/full/metadata.csv.gz')

idx <- sample(1:nrow(expression_data), 10000)

expression_data%>%
  dplyr::filter(row_number() %in% idx)%>%
  data.table::fwrite('results/reduced/data_reduced.csv')

pheno_data%>%
  dplyr::filter(row_number() %in% idx)%>%
  data.table::fwrite('results/reduced/metadata_reduced.csv')
