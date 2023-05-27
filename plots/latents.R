library(tidyverse)
library(ggpubr)
library(viridis)

data_dir = '/data/PycharmProjects/cytof_benchmark/results/latent_plots'
latent_files = list.files(data_dir,pattern = "*.csv",recursive=TRUE,full.names = TRUE)

output_dir = '/data/PycharmProjects/cytof_benchmark/results/summary'

datasets <- str_split(latent_files,'/')%>%
  map_chr(~.x[9])

models <- str_split(latent_files,'/')%>%
  map_chr(~.x[8])

dims <- str_split(latent_files,'/')%>%
  map_chr(~.x[7])

variables <- str_split(latent_files,'/')%>%
  map_chr(~.x[10])%>%
  str_remove('_plot.csv')

latent_data <-
  pmap_dfr(list(file = latent_files, 
                model=models, 
                dataset=datasets,
                dim = dims,
                variable = variables),
           function(file,model,dataset,dim,variable){
             read_csv(file,show_col_types = FALSE)%>%
               mutate(model=model,dataset=dataset,dim=dim,variable=variable)
           }
  )%>%
  nest_by(model,dataset,dim,variable)

plot_data<-latent_data%>%
  filter(dim=='dim2',dataset=='OrganoidDataset',variable=='cell_type')%>%
  ungroup()%>%
  select(model,data)%>%
  unnest(data)%>%
  select_if(~!all(is.na(.x)))%>%
  select(-...1)%>%
  mutate(coords_x = coalesce(VAE1,x),
         coords_y = coalesce(VAE2,y))%>%
  select(model,coords_x,coords_y,cell_type)

FacetEqualWrap <- ggproto(
  "FacetEqualWrap", FacetWrap,
  
  train_scales = function(self, x_scales, y_scales, layout, data, params) {
    
    # doesn't make sense if there is not an x *and* y scale
    if (is.null(x_scales) || is.null(x_scales)) {
      stop("X and Y scales required for facet_equal_wrap")
    }
    
    # regular training of scales
    ggproto_parent(FacetWrap, self)$train_scales(x_scales, y_scales, layout, data, params)
    
    # switched training of scales (x and y and y on x)
    for (layer_data in data) {
      match_id <- match(layer_data$PANEL, layout$PANEL)
      
      x_vars <- intersect(x_scales[[1]]$aesthetics, names(layer_data))
      y_vars <- intersect(y_scales[[1]]$aesthetics, names(layer_data))
      
      SCALE_X <- layout$SCALE_X[match_id]
      ggplot2:::scale_apply(layer_data, y_vars, "train", SCALE_X, x_scales)
      
      SCALE_Y <- layout$SCALE_Y[match_id]
      ggplot2:::scale_apply(layer_data, x_vars, "train", SCALE_Y, y_scales)
    }
    
  }
)

facet_wrap_equal <- function(...) {
  # take advantage of the sanitizing that happens in facet_wrap
  facet_super <- facet_wrap(...)
  
  ggproto(NULL, FacetEqualWrap,
          shrink = facet_super$shrink,
          params = facet_super$params
  )
}

ggplot(plot_data,aes(x=coords_x,y=coords_y,color=cell_type))+
  geom_point(size=0.05)+
  facet_wrap_equal(~model,ncol=4,scales='free')+
  labs(x='VAE1',y='VAE2',color='Cell type')+
  theme(legend.position="bottom")+ 
  guides(color = guide_legend(override.aes = list(size = 5), nrow=1))

ggsave(file.path(output_dir,paste0('latent2d.png')),
       width=12, height=3.8, dpi=100)

plot_data_3d <-latent_data%>%
  filter(dim=='dim3',dataset=='CafDataset',variable=='Patient')%>%
  ungroup()%>%
  select(model,data)%>%
  unnest(data)%>%
  select_if(~!all(is.na(.x)))

ggplot(plot_data_3d,aes(x=UMAP1,y=UMAP2,color=as.factor(Patient)))+
  geom_point(size=0.05)+
  facet_wrap(~model,ncol=4,scales='free')+
  labs(x='UMAP1',y='UMAP2',color='Patient ID')+
  theme(legend.position="bottom")+ 
  guides(color = guide_legend(override.aes = list(size = 5), nrow=1))

ggsave(file.path(output_dir,paste0('latent3d.png')),
       width=12, height=3.8, dpi=100)

plot_data_5d <-latent_data%>%
  filter(dim=='dim5',dataset=='ChallengeDataset',variable=='cell_line')%>%
  ungroup()%>%
  select(model,data)%>%
  unnest(data)%>%
  select_if(~!all(is.na(.x)))

ggplot(plot_data_5d,aes(x=UMAP1,y=UMAP2,color=cell_line))+
  geom_point(size=0.05)+
  facet_wrap(~model,ncol=4,scales='free')+
  labs(x='UMAP1',y='UMAP2',color='Cell line')+
  theme(legend.position="bottom")+ 
  guides(color = guide_legend(override.aes = list(size = 5), nrow=3))

ggsave(file.path(output_dir,paste0('latent5d.png')),
       width=12, height=4.2, dpi=100)

plot_data_supp <- latent_data%>%
  filter(dataset=='OrganoidDataset',variable=='cell_type',model=='BetaVAE')%>%
  ungroup()%>%
  select(dim,data)%>%
  unnest(data)%>%
  select_if(~!all(is.na(.x)))%>%
  mutate(x=if_else(is.na(VAE1),UMAP1,VAE1),
         y=if_else(is.na(VAE2),UMAP2,VAE2)
         )%>%
  select(dim,cell_type,x,y)%>%
  mutate(dim=paste0(str_remove(dim,'dim'),'d'))

ggplot(plot_data_supp,aes(x=x,y=y,color=cell_type))+
  geom_point(size=0.05)+
  facet_wrap_equal(~dim,ncol=3,scales='free')+
  labs(x='Coordinate 1',y='Coordinate 2',color='Cell type')+
  theme(legend.position="bottom")+ 
  guides(color = guide_legend(override.aes = list(size = 5), nrow=1))

ggsave(file.path(output_dir,paste0('latent_supp.png')),
       width=9, height=4, dpi=100)