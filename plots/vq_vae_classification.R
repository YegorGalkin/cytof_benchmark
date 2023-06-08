library(tidyverse)

vqvae_8dim_file = '/data/PycharmProjects/cytof_benchmark/results/summary/vqvae/latent_8bit_binary_val.csv'

vqvae_8d = read_csv(vqvae_8dim_file)

prediction_cell_type<-vqvae_8d%>%
  mutate(code = 128*VQ_1+64*VQ_2+32*VQ_3+16*VQ_4+8*VQ_5+4*VQ_6+2*VQ_7+VQ_8)%>%
  select(code,cell_type)%>%
  group_by(code,cell_type)%>%
  summarise(total = n())%>%
  ungroup()%>%
  group_by(code)%>%
  mutate(total_cells=sum(total),max_cells=max(total))%>%
  summarise(predicted_cell_type = cell_type[total==max_cells])%>%
  arrange(code,predicted_cell_type)%>%
  distinct(code,.keep_all = T)

prediction_day<-vqvae_8d%>%
  mutate(code = 128*VQ_1+64*VQ_2+32*VQ_3+16*VQ_4+8*VQ_5+4*VQ_6+2*VQ_7+VQ_8)%>%
  select(code,day)%>%
  group_by(code,day)%>%
  summarise(total = n())%>%
  ungroup()%>%
  group_by(code)%>%
  mutate(total_cells=sum(total),max_cells=max(total))%>%
  summarise(predicted_day = day[total==max_cells])%>%
  arrange(code,predicted_day)%>%
  distinct(code,.keep_all = T)

vqvae_8d_test = read_csv('/data/PycharmProjects/cytof_benchmark/results/summary/vqvae/latent_8bit_binary_test.csv')

vqvae_8d_test%>%
  mutate(code = 128*VQ_1+64*VQ_2+32*VQ_3+16*VQ_4+8*VQ_5+4*VQ_6+2*VQ_7+VQ_8)%>%
  select(cell_type,day,code)%>%
  left_join(prediction_cell_type)%>%
  left_join(prediction_day)%>%
  summarise(acc_cell_type = sum(predicted_cell_type==cell_type,na.rm = T)/n(),
            acc_day = sum(predicted_day==day,na.rm = T)/n())