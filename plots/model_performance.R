library(tidyverse)
library(ggpubr)

experiment_dir = '/home/egor/Desktop/logs/BetaVAE'

loss_curves_file <- list.files(experiment_dir,pattern = '*loss_curve.csv',recursive = TRUE,full.names = TRUE)
config_files <- list.files(experiment_dir,pattern = '*config.txt',recursive = TRUE,full.names = TRUE)

result_plot_dir = '/data/PycharmProjects/cytof_benchmark/results/summary'

configs <-config_files%>%
  map(read_lines)%>%
  map(str_subset,':')%>%
  map2_dfr(config_files, ~data.frame(key   = str_split_fixed(.x,': ',2)[,1],
                                value = str_split_fixed(.x,': ',2)[,2],
                                file  = .y))

configs_layer <- config_files%>%
  map(read_lines)%>%
  map(str_subset,'-')%>%
  map2_dfr(config_files, ~data.frame(depth = length(.x),
                                width = as.integer(str_remove(.x[1],'- ')),
                                file = .y))%>%
  pivot_longer(cols = depth:width, names_to = 'key',values_to = 'value')

config_params<- rbind(configs,configs_layer)%>%
  mutate(experiment = map_chr(file,~str_split_fixed(.x,'/',9)[,7]),
         run = map_chr(file,~str_split_fixed(.x,'/',9)[,8]))%>%
  select(-file)

loss_curves<-
loss_curves_file%>%
  map_dfr(~read_csv(.x,show_col_types = FALSE)%>%mutate(file = .x))%>%
  mutate(experiment = map_chr(file,~str_split_fixed(.x,'/',9)[,7]),
         run = map_chr(file,~str_split_fixed(.x,'/',9)[,8]))%>%
  select(-file)


mse_summary<-loss_curves%>%
  group_by(experiment,run)%>%
  filter(epoch==max(epoch))%>%
  group_by(experiment)%>%
  filter(val_MSE==min(val_MSE))%>%
  ungroup()%>%
  left_join(config_params%>%
              pivot_wider(names_from = key,values_from = value))%>%
  select(experiment,run,val_MSE,depth,width,epochs)%>%
  mutate(depth=as.integer(depth),width=as.integer(width),epochs=as.integer(epochs))%>%
  add_row(experiment='pbt',val_MSE=0.2974060810448831,width=256,epochs=5600,depth=5)

ggplot(mse_summary,aes(x=experiment,y=val_MSE-0.29,fill=experiment))+
  geom_bar(position="dodge", stat="identity")+
  scale_y_continuous(limits = c(0,0.03), breaks=seq(0,0.03,0.005),
                     labels=seq(0.29,0.32,0.005))+
  scale_fill_discrete(labels = c('20 total, (3k,256,5)',
                                 '18 total, (3k,128,5)',
                                 '8 total, (3k,256,5)',
                                 '5 total, (20k,256,5)',
                                 '5 total, (3k,1024,5)',
                                 '4 total, (3k,512,5)',
                                 '4 total, (3k,256,5)',
                                 '16 total, (5.6k,256,5)'))+
  labs(x='Experiment',y='Validation MSE',
       fill = 'Number of experiments,\nbest hyperparameters\n(epochs,width,depth)',
       title='Summary of hyperparameter tuning',
       subtitle = 'with BetaVAE on Organoid dataset')

ggsave(file.path(result_plot_dir,paste0('summary_mse.png')),
       width=6, height=6, dpi=100)