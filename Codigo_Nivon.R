library(tidyverse)
library(keras)

# Declaramos workspace
setwd("~/Documents/maestría/Proyecto_aprendizaje")
rm(list=ls())


carpeta<-paste0("Bases/",Sys.Date())
#dir.create(carpeta)

# Preprocesamiento de imagenes #

# Normalizamos pixeles a [0,1] (los pixeles originalmente vienen entre 0 y 255)
# Preprocesamiento de imagenes. Genera minilotes de datos de imagenes.
entrena <- image_data_generator(rescale = 1/255)
valida <- image_data_generator(rescale = 1/255)
prueba <- image_data_generator(rescale = 1/255)

# Escalamos el tamaño de las imágenes
ancho <- 50
alto <- 50
tamaño <- c(ancho, alto)

# Generamos minilotes de entrenamiento y validación en A COLOR
gen_minilote_entrena <- 
  flow_images_from_directory('train/', 
                             entrena, 
                             target_size = tamaño,
                             batch_size = 32,
                             class_mode = "binary",
                             classes = c("perro", "gato"),
                             shuffle = T,
                             seed = 42711)
gen_valida <- 
  flow_images_from_directory('validation/', 
                             valida,
                             target_size = tamaño,
                             batch_size = 500,
                             class_mode = "binary",
                             shuffle = FALSE,
                             classes = c("perro", "gato"))

# Creamos tablas de los objetos por carpeta
# 0 corresponde a perro y 1 corresponde a gato
table(gen_minilote_entrena$classes)
table(gen_valida$classes)

# Número de observaciones por muestras
n_entrena <- gen_minilote_entrena$n
n_valida <- gen_valida$n

# Índices de cada objeto
indices <- gen_minilote_entrena$class_indices

# Prueba de modelo # 
#Regularización
lambda <- 0.01
# Red simple
modelo <- keras_model_sequential()


modelo%>%
  layer_conv_2d(filters = 8, kernel_size = c(5,5), 
                activation = 'relu',
                input_shape = c(16,16,1), 
                padding ='same',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_1') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_conv_2d(filters = 12, kernel_size = c(3,3), 
                activation = 'relu',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_2') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 50, activation = 'relu',
              kernel_regularizer = regularizer_l2(lambda)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

modelo %>% compile(
  loss = "binary_crossentropy",
  #puedes probar con otros optimizadores (adam, por ejemplo), 
  #recuerda ajustar lr y momento:
  optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9), 
  metrics = "accuracy"
)

early_stop <- callback_early_stopping(monitor = "val_loss",min_delta=0.005,patience = 5,verbose=1)

ajuste <- modelo %>% fit_generator(
  gen_minilote_entrena,
  validation_data = gen_valida,
  validation_steps = n_valida / 500,
  steps_per_epoch = n_entrena / 32, # entre tamaño de minibatch 
  workers = 4,
  epochs = 40 , # Incrementamos épocas
  verbose = 1,
  callbacks = list(early_stop))

# Exporta métricas
write.csv(as.data.frame(ajuste$metrics),paste0(carpeta,"/metricas_2-2.csv"),row.names=F)

# Predecimos
prueba <- image_data_generator(rescale = 1/255)
gen_prueba <- 
  flow_images_from_directory('test/', 
                             prueba,
                             target_size = tamaño,
                             batch_size = 1,
                             shuffle = F)

prediccion<-data.frame(prediccion1=modelo %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1))
for(i in 1:3){
  gen_prueba <- 
    flow_images_from_directory('test/', 
                               prueba,
                               target_size = tamaño,
                               batch_size = 1,
                               shuffle = F)
  aux<-modelo %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1)
  prediccion<-cbind(prediccion,aux)
  names(prediccion)[i]<-paste0("prediccion",i)
}
write.csv(prediccion,paste0(carpeta,"/prediccion_2-2.csv"),row.names=F)

# Guardamos gráfica de épocas
ggsave(paste0(carpeta,"/plot_2-2.jpeg"),plot(ajuste),scale=1.5)


# Modelo 2
#modelo%>%
#  layer_conv_2d(filters = 8, kernel_size = c(5,5), 
#                activation = 'relu',
#                input_shape = c(16,16,1), 
#                padding ='same',
#                kernel_regularizer = regularizer_l2(lambda),
#                name = 'conv_1') %>%
#  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
#  layer_dropout(rate = 0.25) %>% 
#  layer_conv_2d(filters = 12, kernel_size = c(3,3), 
#                activation = 'relu',
#                kernel_regularizer = regularizer_l2(lambda),
#                name = 'conv_2') %>% 
#  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
#  layer_dropout(rate = 0.25) %>% 
#  layer_flatten() %>% 
#  layer_dense(units = 50, activation = 'relu',
#              kernel_regularizer = regularizer_l2(lambda)) %>%
#  layer_dropout(rate = 0.5) %>%
#  layer_dense(units = 1, activation = "sigmoid")