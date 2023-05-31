library(tidyverse)
library(scatterpie)

experiment_dir = '/data/PycharmProjects/cytof_benchmark/logs/VQVAE/exp_8'

loss_curves_file <- list.files(experiment_dir,pattern = '*loss_curve.csv',recursive = TRUE,full.names = TRUE)
config_files <- list.files(experiment_dir,pattern = '*config.txt',recursive = TRUE,full.names = TRUE)

result_plot_dir = '/data/PycharmProjects/cytof_benchmark/results/summary/vqvae'

configs <-config_files%>%
  map(read_lines)%>%
  map(str_subset,':')%>%
  map2_dfr(config_files, ~data.frame(key   = str_split_fixed(.x,': ',2)[,1],
                                     value = str_split_fixed(.x,': ',2)[,2],
                                     file  = .y))%>%
  pivot_wider(id_cols=file,names_from = key,values_from = value)%>%
  select(file,embed_channels,embed_dim,embed_entries,hidden_features,n_layers)%>%
  mutate(run = map_chr(file,~str_split_fixed(.x,'/',9)[,8]))%>%
  select(-file)%>%
  select(run,everything())
  

loss_curves<-
  loss_curves_file%>%
  map_dfr(~read_csv(.x,show_col_types = FALSE)%>%mutate(file = .x))%>%
  mutate(run = map_chr(file,~str_split_fixed(.x,'/',9)[,8]))%>%
  select(-file)

loss_curves%>%
  left_join(configs)%>%
  mutate(embed_config=paste0('c',embed_channels,' dim',embed_dim,' ent',embed_entries))%>%
  mutate(params = paste0('hidden',hidden_features,' layers',n_layers))%>%
  ggplot(aes(x=epoch,y=val_MSE,color=params))+
  geom_line()+
  coord_cartesian(ylim = c(0.43,0.75))+
  facet_wrap(~embed_config)

ggsave(file.path(result_plot_dir,'summary_vqvae.png'),
       width=12, height=8, dpi=100)

loss_curves%>%
  left_join(configs)%>%
  filter(epoch==max(epoch))%>%
  arrange(val_MSE)%>%
  write_csv(file.path(result_plot_dir,'summary_vqvae.csv'))

loss_curves%>%
  left_join(configs)%>%
  mutate(embed_config=paste0('c',embed_channels,' dim',embed_dim,' ent',embed_entries))%>%
  mutate(params = paste0('hidden',hidden_features,' layers',n_layers))%>%
  group_by(run)%>%
  mutate(min_mse = min(val_MSE))%>%
  ungroup()%>%
  group_by(embed_config)%>%
  filter(min_mse==min(min_mse))%>%
  ggplot(aes(x=epoch,y=val_MSE,color=embed_config))+
  geom_line()+
  coord_cartesian(ylim = c(0.43,0.75))+
  labs(x='Epoch',y='MSE loss (val)',color='Embedding')

ggsave(file.path(result_plot_dir,'vqvae_training.png'),
       width=6, height=4, dpi=100)

vqvae_2dim_file = '/data/PycharmProjects/cytof_benchmark/results/summary/vqvae/latent_8bit_coords.csv'

vqvae_2d = read_csv(vqvae_2dim_file)

vq_vae_2d_data<-vqvae_2d%>%
  group_by(VQVAE1,VQVAE2)%>%
  mutate(cluster = cur_group_id())%>%
  ungroup()%>%
  group_by(cluster)%>%
  mutate(size = n())%>%
  group_by(cluster,size,cell_type,VQVAE1,VQVAE2)%>%
  summarise(n_cell_type = n())%>%
  pivot_wider(names_from = cell_type,values_from = n_cell_type,values_fill = 0)%>%
  mutate(r = log10(size)/50)

  ggplot()+
  geom_scatterpie(data=vq_vae_2d_data, 
                  aes(x=VQVAE1,y=VQVAE2,group=cluster,r=r),
                  cols=unique(vqvae_2d$cell_type),
                  legend_name = 'cell_type',
                  color=NA)+
    coord_fixed(ratio = 1)+
    labs(color='Cell Type')
  
ggsave(file.path(result_plot_dir,'vqvae_2d.png'),
         width=6, height=4, dpi=100)

vqvae_8dim_file = '/data/PycharmProjects/cytof_benchmark/results/summary/vqvae/latent_8bit_binary.csv'

vqvae_8d = read_csv(vqvae_8dim_file)

vq_vae_8d_data<-vqvae_8d%>%
  group_by(VQ_1, VQ_2, VQ_3, VQ_4, VQ_5, VQ_6, VQ_7, VQ_8)%>%
  mutate(cluster = cur_group_id())%>%
  mutate(size = n())%>%
  group_by(cluster,size,cell_type,VQ_1, VQ_2, VQ_3, VQ_4, VQ_5, VQ_6, VQ_7, VQ_8)%>%
  summarise(n_cell_type = n())%>%
  pivot_wider(names_from = cell_type,values_from = n_cell_type,values_fill = 0)%>%
  mutate(r = log10(size)/7)%>%
  mutate(x=VQ_1+2*VQ_2+4*VQ_3+8*VQ_4,
         y=VQ_5+2*VQ_6+4*VQ_7+8*VQ_8)

labels_x = c(16+0:15)%>%map(intToBitVect)%>%map_chr(paste0,collapse="")%>%map_chr(str_remove,"^1")%>%map_chr(paste0,'xxxx')
labels_y = c(16+0:15)%>%map(intToBitVect)%>%map_chr(paste0,collapse="")%>%map_chr(str_remove,"^1")%>%map_chr(~paste0('xxxx',.x))

ggplot()+
  geom_scatterpie(data=vq_vae_8d_data, 
                  aes(x=x,y=y,group=cluster,r=r),
                  cols=unique(vqvae_2d$cell_type),
                  legend_name = 'cell_type',
                  color=NA)+
  coord_fixed(ratio = 1)+
  labs(color='Cell Type')+
  scale_x_continuous(breaks = c(0:15),labels = labels_x)+
  scale_y_continuous(breaks = c(0:15),labels = labels_y)+ 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  labs(color='Cell Type',x='Latent code',y='Latent code')

ggsave(file.path(result_plot_dir,'vqvae_8d.png'),
       width=8, height=6, dpi=100)