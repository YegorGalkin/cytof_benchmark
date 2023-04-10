library(tidyverse)
library(ggpubr)
library(viridis)

baseline_file <- '/data/PycharmProjects/cytof_benchmark/results/classifier_data/lgbm_baseline.csv'
latent_file <- '/data/PycharmProjects/cytof_benchmark/results/classifier_data/lgbm_latent.csv'
pca_file <- '/data/PycharmProjects/cytof_benchmark/results/classifier_data/lgbm_pca.csv'

output_dir = '/data/PycharmProjects/cytof_benchmark/results/classifier_plots'

baselines<- read_csv(baseline_file)%>%
  select(-1)%>%
  filter(variable!='Cell_type')

latents<- read_csv(latent_file)%>%
  select(-1)%>%
  mutate(dim = as.integer(str_remove(dim,'dim')))%>%
  filter(variable!='Cell_type')

pcas<- read_csv(pca_file)%>%
  select(-1)%>%
  filter(variable!='Cell_type')

ggplot(latents,aes(x = dim, y = test_acc))+
  geom_point(aes(color=model))+
  geom_line(aes(color=model))+
  geom_hline(data=baselines,aes(yintercept = maj_class_acc,color = 'Majority class'))+
  geom_hline(data=baselines,aes(yintercept = lgbm_all_vars_test_acc,color = 'All variables'))+
  geom_point(data=pcas,aes(x = dim, y = test_acc, color='PCA'))+
  geom_line(data=pcas,aes(x = dim, y = test_acc, color='PCA'))+
  facet_wrap(dataset~variable, scale = 'free_y',nrow =2)+
  scale_color_discrete(breaks=c('BetaVAE', 'DBetaVAE', 'WAE_MMD', 'HyperSphericalVAE',
                                'Majority class','PCA','All variables'))+
  scale_x_continuous(breaks=c(2,3,5))+
  labs(x='Latent space dimensionality',y='Test set accuracy (LightGBM)',color='Model',
       title = 'Clasiffier accuracy for predicting metadata using latent space embeddings across datasets.')

ggsave(file.path(output_dir,paste0('classifier_acc.png')),
       width=9, height=6, dpi=100)