library(tidyverse)
library(ggpubr)
library(viridis)

data_dir = '/data/PycharmProjects/cytof_benchmark/results/mse_data'
mse_files = list.files(data_dir,pattern = "*mse.csv",full.names = TRUE)


organoid_files = mse_files[str_detect(mse_files,'OrganoidDataset')]
models = str_split(organoid_files,'/')%>%
  map_chr(~tail(.x,1))%>%
  str_remove('OrganoidDataset_')%>%
  str_remove('_mse.csv')

organoid_mse_data <- 
  map2_dfr(organoid_files, models, 
           ~.x%>%read_csv()%>%mutate(model=.y)%>%rename(id=1))

organoid_mse_data%>%
  select(-id)%>%
  group_by(model)%>%
  summarise_all(mean)%>%
  pivot_longer(names_to='AB',values_to ='MSE',-model)%>%
  ggplot(aes(y=AB,x=MSE,fill = model))+
  scale_fill_viridis(discrete = T)+
  geom_bar(position="dodge", stat="identity")

result_plot_dir = '/data/PycharmProjects/cytof_benchmark/results/mse'

ggsave(file.path(result_plot_dir,paste0('organoid_AB.png')),
       width=8, height=8, dpi=300)

mses <- organoid_mse_data%>%
  select(-id)%>%
  nest_by(model)%>%
  mutate(mse = list(rowMeans(as.matrix(data))))

corr_data = data.frame(model1=c(),model2=c(),spearman=c())
for (model1 in mses$model){
  for (model2 in mses$model){
    corr_data<-corr_data%>%rbind(
      data.frame(model1=model1,model2=model2,
              spearman=cor(
                mses%>%filter(model==model1)%>%pull(mse)%>%.[[1]],
                mses%>%filter(model==model2)%>%pull(mse)%>%.[[1]])
      ))
  }
}

corr_data%>%
  filter(model1!=model2)%>%
  write_csv(file.path(result_plot_dir,'organoid_cell_correlation.csv'))

### CAF dataset
caf_files = mse_files[str_detect(mse_files,'CafDataset')]
models = str_split(caf_files,'/')%>%
  map_chr(~tail(.x,1))%>%
  str_remove('CafDataset_')%>%
  str_remove('_mse.csv')

caf_mse_data <- 
  map2_dfr(caf_files, models, 
           ~.x%>%read_csv()%>%mutate(model=.y)%>%rename(id=1))

caf_mse_data%>%
  select(-id)%>%
  group_by(model)%>%
  summarise_all(mean)%>%
  pivot_longer(names_to='AB',values_to ='MSE',-model)%>%
  ggplot(aes(y=AB,x=MSE,fill = model))+
  scale_fill_viridis(discrete = T)+
  geom_bar(position="dodge", stat="identity")

result_plot_dir = '/data/PycharmProjects/cytof_benchmark/results/mse'

ggsave(file.path(result_plot_dir,paste0('caf_AB.png')),
       width=8, height=8, dpi=300)

mses <- caf_mse_data%>%
  select(-id)%>%
  nest_by(model)%>%
  mutate(mse = list(rowMeans(as.matrix(data))))

corr_data = data.frame(model1=c(),model2=c(),spearman=c())
for (model1 in mses$model){
  for (model2 in mses$model){
    corr_data<-corr_data%>%rbind(
      data.frame(model1=model1,model2=model2,
                 spearman=cor(
                   mses%>%filter(model==model1)%>%pull(mse)%>%.[[1]],
                   mses%>%filter(model==model2)%>%pull(mse)%>%.[[1]])
      ))
  }
}

corr_data%>%
  filter(model1!=model2)%>%
  write_csv(file.path(result_plot_dir,'caf_cell_correlation.csv'))


### Breast Cancer Challenge Dataset
challenge_files = mse_files[str_detect(mse_files,'ChallengeDataset')]
models = str_split(challenge_files,'/')%>%
  map_chr(~tail(.x,1))%>%
  str_remove('ChallengeDataset_')%>%
  str_remove('_mse.csv')

challenge_mse_data <- 
  map2_dfr(challenge_files, models, 
           ~.x%>%read_csv()%>%mutate(model=.y)%>%rename(id=1))

challenge_mse_data%>%
  select(-id)%>%
  group_by(model)%>%
  summarise_all(mean)%>%
  pivot_longer(names_to='AB',values_to ='MSE',-model)%>%
  ggplot(aes(y=AB,x=MSE,fill = model))+
  scale_fill_viridis(discrete = T)+
  geom_bar(position="dodge", stat="identity")

result_plot_dir = '/data/PycharmProjects/cytof_benchmark/results/mse'

ggsave(file.path(result_plot_dir,paste0('challenge_AB.png')),
       width=8, height=8, dpi=300)

mses <- challenge_mse_data%>%
  select(-id)%>%
  nest_by(model)%>%
  mutate(mse = list(rowMeans(as.matrix(data))))

corr_data = data.frame(model1=c(),model2=c(),spearman=c())
for (model1 in mses$model){
  for (model2 in mses$model){
    corr_data<-corr_data%>%rbind(
      data.frame(model1=model1,model2=model2,
                 spearman=cor(
                   mses%>%filter(model==model1)%>%pull(mse)%>%.[[1]],
                   mses%>%filter(model==model2)%>%pull(mse)%>%.[[1]])
      ))
  }
}

corr_data%>%
  filter(model1!=model2)%>%
  write_csv(file.path(result_plot_dir,'challenge_cell_correlation.csv'))