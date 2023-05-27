library(tidyverse)
library(ggpubr)
pbt_data_file = '/data/PycharmProjects/cytof_benchmark/results/deprecated/benchmark_loss_dynamics.csv'
exp_5_data_file = '/home/egor/Desktop/logs/BetaVAE/exp_5/run_5/loss_curve.csv'
result_plot_dir = '/data/PycharmProjects/cytof_benchmark/results/summary'
training_metrics <- read_csv(pbt_data_file)%>%
  rename(training_run = 1, index = 2)%>%
  arrange(training_run,index)%>%
  group_by(dataset,model)%>%
  mutate(training_run = as.numeric(as.factor(training_run)))%>%
  group_by(model,dataset)%>%
  nest()

exp_5_metrics = read_csv(exp_5_data_file)

pbt_metrics <-training_metrics%>%filter(dataset=='OrganoidDataset',model=='BetaVAE')%>%pull(data)%>%.[[1]]

p1<-pbt_metrics%>%
  ggplot(aes(x=step,y=MSE,color=as.factor(training_run)))+
  geom_line()+
  coord_cartesian(ylim = c(0.29,0.45))+
  labs(x = 'Epoch',title='Population based training')+ 
  guides(color='none')

p2<-exp_5_metrics%>%
  select(epoch,train_MSE,val_MSE)%>%
  pivot_longer(-epoch,names_to='stage',values_to = 'MSE')%>%
  mutate(stage=str_remove(stage,'_MSE'))%>%
  mutate(stage=str_replace(stage,'val','Validation'))%>%
  mutate(stage=str_replace(stage,'train','Training'))%>%
  ggplot(aes(x=epoch,y=MSE,color=stage))+
  geom_line()+
  coord_cartesian(ylim = c(0.29,0.45))+
  labs(x = 'Epoch',y='MSE',color='Stage')+
  theme(legend.position = c(.8, .8))+
  scale_color_hue(c=100, l=60)+
  labs(title='Experiment 5')

ggpubr::ggarrange(p1,p2)

ggsave(file.path(result_plot_dir,paste0('summary_training.png')),
       width=8, height=4, dpi=100)
