library(tidyverse)
library(ggpubr)
library(viridis)

data_dir = '/data/PycharmProjects/cytof_benchmark/results/mse_data'
pca_data_dir = '/data/PycharmProjects/cytof_benchmark/results/pca_data/biomarker_mse.csv'
mse_files = list.files(data_dir,pattern = "*mse.csv",full.names = TRUE)

output_dir = '/data/PycharmProjects/cytof_benchmark/results/mse'

models <- str_split(mse_files,'/')%>%
  map_chr(~tail(.x,1))%>%
  str_remove('.+?Dataset_')%>%
  str_remove('_mse.csv')

datasets <- str_split(mse_files,'/')%>%
  map_chr(~tail(.x,1))%>%
  str_extract('.+?Dataset')

mse_vae_data<-
  pmap_dfr(list(file = mse_files, model=models, dataset=datasets),
     function(file,model,dataset){
       read_csv(file)%>%
         select(-1)%>%
         summarise_all(mean)%>%
         pivot_longer(names_to='biomarker',values_to ='mse_vae',cols=everything())%>%
         mutate(model=model,dataset=dataset)
       }
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