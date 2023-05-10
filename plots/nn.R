library(tidyverse)
library(RANN)
library(data.table)

data_dir <- '/data/PycharmProjects/cytof_benchmark/results/latent_data/dim2/'
latent_files <- list.files(data_dir,pattern = "*test.csv",full.names = TRUE,recursive = TRUE)
output_dir <- '/data/PycharmProjects/cytof_benchmark/results/nn_plots'

models <- str_split(latent_files,'/')%>%
  map_chr(~.x[9])

datasets <- str_split(latent_files,'/')%>%
  map_chr(~.x[10])

latents <- tibble(model=models,
                  dataset=datasets,
                  latent = map(latent_files,read_csv,col_select=c('VAE1','VAE2')))

results = list()

for (dataset in unique(datasets)){
  latents_list = list()
  indices_list = list()
  for (i in 1:length(unique(models))){
    model <- unique(models)[i]
    
    latents_list[[i]] <- latents%>%
      filter(model==!!model,dataset==!!dataset)%>%
      pull(latent)%>%.[[1]]%>%filter(row_number()<=1e5)
    
    indices_list[[i]] <- melt(data.table(RANN::nn2(latents_list[[i]], k = 1001)$nn.idx), 
                              id.vars = 'V1', variable.factor = F)
    
  }
  
  for (i in 1:(length(unique(models))-1)){
    for (j in (i+1):length(unique(models))){
      model1 = unique(models)[i]
      model2 = unique(models)[j]
      latent1 = latents_list[[i]]
      latent2 = latents_list[[j]]
      
      indices_1 = indices_list[[i]] 
      indices_2 = indices_list[[j]] 
        
      mutual_neighbourhood <- indices_1[indices_2, on = .(V1,value), nomatch = NULL]
      
      colnames(mutual_neighbourhood)<-c('cell_id','n1','idx','n2')
      mutual_neighbourhood[,`:=`(n1=as.integer(str_remove(n1,'V'))-1L,
                                 n2=as.integer(str_remove(n2,'V'))-1L,
                                 n_max = pmax(n1,n2))]
      
      intersections <- mutual_neighbourhood[,.(common=.N),by=.(cell_id,n_max)][,.(sum(common)),by=n_max]
      setkey(intersections,n_max)
      
      common_fractions <- cumsum(intersections$V1) / c(1:1000) / nrow(latent1)
      
      result = data.frame(common_neighbours = common_fractions, 
                          n = 1:1000,
                          model1=model1,
                          model2=model2,
                          dataset=dataset)
      
      results[[length(results)+1]] = result
      print(paste(model1,model2,dataset,'done',collapse=' '))
    }
  }
}

results%>%
  bind_rows()%>%
  mutate(models=paste(model1,model2,sep=' vs '))%>%
  ggplot(aes(x=n,y=common_neighbours,color = models))+
  geom_point(alpha = 1/2,size = 0.25)+
  geom_line()+
  scale_x_log10()+
  labs(x = "Nearest neighbourhood size", 
       y = "Nearest neighbourhood intersection", 
       title = "Correspondence of nearest neighbourhoods between different models in all datasets")+
  facet_wrap(~dataset,ncol=3)+
  theme(legend.position="bottom")

ggsave(file.path(output_dir,paste0('nn.png')),
       width=12, height=6, dpi=100)