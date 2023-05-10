library(tidyverse)
library(ggpubr)
library(viridis)

data_dir = '/data/PycharmProjects/cytof_benchmark/results/mse_data'
pca_data_dir = '/data/PycharmProjects/cytof_benchmark/results/pca_data/biomarker_mse.csv'
mse_files = list.files(data_dir,pattern = "*.csv",recursive=TRUE,full.names = TRUE)

output_dir = '/data/PycharmProjects/cytof_benchmark/results/mse_plots'

splits <- str_split(mse_files,'/')%>%
  map_chr(~.x[10])%>%
  str_remove('.csv')

datasets <- str_split(mse_files,'/')%>%
  map_chr(~.x[9])

models <- str_split(mse_files,'/')%>%
  map_chr(~.x[8])

dims <- str_split(mse_files,'/')%>%
  map_chr(~.x[7])

mse_vae_data<-
  pmap_dfr(list(file = mse_files, 
                model=models, 
                dataset=datasets,
                dim = dims,
                split = splits),
     function(file,model,dataset,dim,split){
       read_csv(file)%>%
         select(-1)%>%
         summarise_all(mean)%>%
         pivot_longer(names_to='biomarker',values_to ='mse_vae',cols=everything())%>%
         mutate(model=model,dataset=dataset,dim=dim,split=split)
       }
     )

mse_vae_data%>%
  mutate(dim = as.integer(str_remove(dim,'dim')))%>%
  group_by(model,dataset,dim,split)%>%
  filter(split=='test')%>%
  summarise(mse_vae=mean(mse_vae))%>%
  ungroup()%>%
  ggplot(aes(x=dim,y=mse_vae,color = model))+
  geom_point()+
  geom_line()+
  scale_x_continuous(breaks=c(2,3,5))+
  facet_wrap(~dataset,scales = 'free_y',ncol=3)+
  labs(x='Latent layer dimensionality',y='MSE',color='Model')+
  theme(legend.position="bottom")

ggsave(file.path(output_dir,paste0('mse_per_dim.png')),
       width=10, height=5, dpi=100)

mse_vae_data%>%
  group_by(model,dataset,dim,split)%>%
  summarise(mse_vae=mean(mse_vae))%>%
  pivot_wider(names_from=split,values_from = mse_vae)%>%
  mutate(percent_gain_train = ((train-val)/val)*100,
         percent_gain_test = ((test-val)/val)*100)%>%
  ungroup()%>%
  summarise(median_train = median(percent_gain_train),
            min_train = min(percent_gain_train),
            max_train = max(percent_gain_train),
            median_test = median(percent_gain_test),
            min_test = min(percent_gain_test),
            max_test = max(percent_gain_test),
            )

mse_pca_data<-read_csv(pca_data_dir)%>%
  left_join(mse_vae_data)%>%
  arrange(dataset,biomarker,model)

pc_equiv_data<-mse_pca_data%>%
  group_by(biomarker,dataset,model)%>%
  mutate(prev_pc_mask = mse == min(mse[mse>=mse_vae]),
         next_pc_mask = mse == max(mse[mse<=mse_vae]))%>%
  filter(prev_pc_mask | next_pc_mask)%>%
  mutate(prev_pc = pc[prev_pc_mask],next_pc = pc[next_pc_mask],
         prev_mse = mse[prev_pc_mask],next_mse = mse[next_pc_mask])%>%
  mutate(approx_pc_eqiv = prev_pc+(prev_mse-mse_vae)/(prev_mse-next_mse))%>%
  select(biomarker,mse_vae,dataset,model,approx_pc_eqiv)%>%
  distinct()%>%
  ungroup()

pc_equiv_data%>%
  ggplot(aes(y=biomarker,x=approx_pc_eqiv,fill = model))+
  scale_fill_viridis(discrete = T)+
  geom_bar(position="dodge", stat="identity")+
  facet_wrap(~dataset,scales = 'free_y')

ggsave(file.path(output_dir,paste0('mse_pca_equiv.png')),
       width=12, height=8, dpi=100)

pc_equiv_data%>%
  ggplot(aes(y=biomarker,x=mse_vae,fill = model))+
  scale_fill_viridis(discrete = T)+
  geom_bar(position="dodge", stat="identity")+
  facet_wrap(~dataset,scales = 'free')

ggsave(file.path(output_dir,paste0('mse.png')),
       width=12, height=8, dpi=100)