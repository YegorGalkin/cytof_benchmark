library(tidyverse)
library(ggpubr)
data_file = '/data/PycharmProjects/cytof_benchmark/results/benchmark_loss_dynamics.csv'

result_plot_dir = '/data/PycharmProjects/cytof_benchmark/results/training_dynamics'

training_metrics <- read_csv(data_file)%>%
  rename(training_run = 1, index = 2)%>%
  arrange(training_run,index)%>%
  group_by(dataset,model)%>%
  mutate(training_run = as.numeric(as.factor(training_run)))%>%
  group_by(model,dataset)%>%
  nest()

## Organoid dataset plots
organoid_metrics <-training_metrics[training_metrics$dataset=='OrganoidDataset',]

for (model in organoid_metrics$model){
  data = organoid_metrics[organoid_metrics$model==model,]%>%pull(data)%>%.[[1]]
  
  p1<-
    data%>%
    ggplot(aes(x=checkpoint_time,y=loss,color=as.factor(training_run)))+
    geom_line()+
    coord_cartesian(ylim = c(0.31,0.45))+
    labs(x = 'Training time (s)')+ 
    guides(color=guide_legend(title="Run"))
  
  p2<-
    data%>%
    ggplot(aes(x=checkpoint_time,y=MSE,color=as.factor(training_run)))+
    geom_line()+
    coord_cartesian(ylim = c(0.29,0.45))+
    labs(x = 'Training time (s)')+ 
    guides(color=guide_legend(title="Run"))
  
  p3<-data%>%
    ggplot(aes(x=checkpoint_time,y=lr,color=as.factor(training_run)))+
    geom_line()+
    scale_y_log10()+
    labs(x = 'Training time (s)')+ 
    guides(color=guide_legend(title="Run"))
  
  ggarrange(p1, p2, p3, ncol=2, nrow=2, common.legend = TRUE, legend="right")
  
  ggsave(file.path(result_plot_dir,paste0('OrganoidDataset','_',model,'.png')),
         width=8, height=8, dpi=300)
  
}

## Challenge dataset plots
challenge_metrics <-training_metrics[training_metrics$dataset=='ChallengeDataset',]

for (model in challenge_metrics$model){
  data = challenge_metrics[challenge_metrics$model==model,]%>%pull(data)%>%.[[1]]
  
  p1<-
    data%>%
    ggplot(aes(x=checkpoint_time,y=loss,color=as.factor(training_run)))+
    geom_line()+
    coord_cartesian(ylim = c(0.38,0.5))+
    labs(x = 'Training time (s)')+ 
    guides(color=guide_legend(title="Run"))
  
  p2<-
    data%>%
    ggplot(aes(x=checkpoint_time,y=MSE,color=as.factor(training_run)))+
    geom_line()+
    coord_cartesian(ylim = c(0.37,0.45))+
    labs(x = 'Training time (s)')+ 
    guides(color=guide_legend(title="Run"))
  
  p3<-data%>%
    ggplot(aes(x=checkpoint_time,y=lr,color=as.factor(training_run)))+
    geom_line()+
    scale_y_log10()+
    labs(x = 'Training time (s)')+ 
    guides(color=guide_legend(title="Run"))
  
  ggarrange(p1, p2, p3, ncol=2, nrow=2, common.legend = TRUE, legend="right")
  
  ggsave(file.path(result_plot_dir,paste0('ChallengeDataset','_',model,'.png')),
         width=8, height=8, dpi=300)
  
}

## CAF dataset plots
caf_metrics <-training_metrics[training_metrics$dataset=='CafDataset',]

for (model in caf_metrics$model){
  data = caf_metrics[caf_metrics$model==model,]%>%pull(data)%>%.[[1]]
  
  p1<-
    data%>%
    ggplot(aes(x=checkpoint_time,y=loss,color=as.factor(training_run)))+
    geom_line()+
    coord_cartesian(ylim = c(0.5,0.7))+
    labs(x = 'Training time (s)')+ 
    guides(color=guide_legend(title="Run"))
  
  p2<-
    data%>%
    ggplot(aes(x=checkpoint_time,y=MSE,color=as.factor(training_run)))+
    geom_line()+
    coord_cartesian(ylim = c(0.5,0.65))+
    labs(x = 'Training time (s)')+ 
    guides(color=guide_legend(title="Run"))
  
  p3<-data%>%
    ggplot(aes(x=checkpoint_time,y=lr,color=as.factor(training_run)))+
    geom_line()+
    scale_y_log10()+
    labs(x = 'Training time (s)')+ 
    guides(color=guide_legend(title="Run"))
  
  ggarrange(p1, p2, p3, ncol=2, nrow=2, common.legend = TRUE, legend="right")
  
  ggsave(file.path(result_plot_dir,paste0('CAFDataset','_',model,'.png')),
         width=8, height=8, dpi=300)
  
}

best_metrics = training_metrics%>%
  unnest(data)%>%
  group_by(dataset,model)%>%
  filter(loss ==min(loss))%>%
  filter(index==max(index))

best_metrics%>%
  ggplot(aes(x=model,y=MSE,fill = model))+
  geom_col()+
  facet_wrap(~dataset)+
  
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
ggsave(file.path(result_plot_dir,paste0('metrics_dim2.png')),
       width=8, height=8, dpi=300)

best_metrics%>%
  arrange(dataset,MSE)%>%
  write_csv(file.path(result_plot_dir,paste0('metrics_dim2.csv')))