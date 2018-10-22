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
modelo_1 <- keras_model_sequential()
modelo_2 <- keras_model_sequential()
modelo_3 <- keras_model_sequential()
modelo_4 <- keras_model_sequential()
modelo_5 <- keras_model_sequential()
modelo_6 <- keras_model_sequential()
modelo_7 <- keras_model_sequential()
modelo_8 <- keras_model_sequential()
modelo_9 <- keras_model_sequential()
modelo_10 <- keras_model_sequential()


modelo_1%>%
  layer_conv_2d(filters = 8, kernel_size = c(5,5), 
                activation = 'relu',
                input_shape = c(ancho, alto, 3), 
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

modelo_1 %>% compile(
  loss = "binary_crossentropy",
  #puedes probar con otros optimizadores (adam, por ejemplo), 
  #recuerda ajustar lr y momento:
  optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9), 
  metrics = "accuracy"
)

early_stop <- callback_early_stopping(monitor = "val_loss",min_delta=0.005,patience = 5,verbose=1)

ajuste <- modelo_1 %>% fit_generator(
  gen_minilote_entrena,
  validation_data = gen_valida,
  validation_steps = n_valida / 500,
  steps_per_epoch = n_entrena / 32, # entre tamaño de minibatch 
  workers = 4,
  epochs = 40 , # Incrementamos épocas
  verbose = 1,
  callbacks = list(early_stop))

# Exporta métricas
write.csv(as.data.frame(ajuste$metrics),paste0(carpeta,"/metricas_1-1.csv"),row.names=F)

# Predecimos
prueba <- image_data_generator(rescale = 1/255)
gen_prueba <- 
  flow_images_from_directory('test/', 
                             prueba,
                             target_size = tamaño,
                             batch_size = 1,
                             shuffle = F)

prediccion<-data.frame(prediccion1=modelo_1 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1))
for(i in 1:3){
  gen_prueba <- 
    flow_images_from_directory('test/', 
                               prueba,
                               target_size = tamaño,
                               batch_size = 1,
                               shuffle = F)
  aux<-modelo_1 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1)
  prediccion<-cbind(prediccion,aux)
  names(prediccion)[i]<-paste0("prediccion",i)
}
write.csv(prediccion,paste0(carpeta,"/prediccion_1-1.csv"),row.names=F)

# Guardamos gráfica de épocas
ggsave(paste0(carpeta,"/plot_1-1.jpeg"),plot(ajuste),scale=1.5)






modelo_2%>%
  layer_conv_2d(filters = 8, kernel_size = c(5,5), 
                activation = 'relu',
                input_shape = c(ancho, alto, 3), 
                padding ='same',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_1') %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_conv_2d(filters = 12, kernel_size = c(3,3), 
                activation = 'sigmoid',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_2') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 50, activation = 'selu',
              kernel_regularizer = regularizer_l2(lambda)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

modelo_2 %>% compile(
  loss = "binary_crossentropy",
  #puedes probar con otros optimizadores (adam, por ejemplo), 
  #recuerda ajustar lr y momento:
  optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9), 
  metrics = "accuracy"
)

early_stop <- callback_early_stopping(monitor = "val_loss",min_delta=0.005,patience = 5,verbose=1)

ajuste <- modelo_2 %>% fit_generator(
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

prediccion<-data.frame(prediccion1=modelo_2 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1))
for(i in 1:3){
  gen_prueba <- 
    flow_images_from_directory('test/', 
                               prueba,
                               target_size = tamaño,
                               batch_size = 1,
                               shuffle = F)
  aux <- modelo_2 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1)
  prediccion<-cbind(prediccion,aux)
  names(prediccion)[i]<-paste0("prediccion",i)
}
write.csv(prediccion,paste0(carpeta,"/prediccion_2-2.csv"),row.names=F)

# Guardamos gráfica de épocas
ggsave(paste0(carpeta,"/plot_2-2.jpeg"),plot(ajuste),scale=1.5)






modelo_3%>%
  layer_conv_2d(filters = 8, kernel_size = c(5,5), 
                activation = 'relu',
                input_shape = c(ancho, alto, 3), 
                padding ='same',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_1') %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_conv_2d(filters = 12, kernel_size = c(3,3), 
                activation = 'sigmoid',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_2') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 50, activation = 'selu',
              kernel_regularizer = regularizer_l2(lambda)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

modelo_3 %>% compile(
  loss = "binary_crossentropy",
  #puedes probar con otros optimizadores (adam, por ejemplo), 
  #recuerda ajustar lr y momento:
  optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9), 
  metrics = "accuracy"
)

early_stop <- callback_early_stopping(monitor = "val_loss",min_delta=0.005,patience = 5,verbose=1)

ajuste <- modelo_3 %>% fit_generator(
  gen_minilote_entrena,
  validation_data = gen_valida,
  validation_steps = n_valida / 500,
  steps_per_epoch = n_entrena / 32, # entre tamaño de minibatch 
  workers = 4,
  epochs = 40 , # Incrementamos épocas
  verbose = 1,
  callbacks = list(early_stop))

# Exporta métricas
write.csv(as.data.frame(ajuste$metrics),paste0(carpeta,"/metricas_3-3.csv"),row.names=F)

# Predecimos
prueba <- image_data_generator(rescale = 1/255)
gen_prueba <- 
  flow_images_from_directory('test/', 
                             prueba,
                             target_size = tamaño,
                             batch_size = 1,
                             shuffle = F)

prediccion<-data.frame(prediccion1=modelo_3 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1))
for(i in 1:3){
  gen_prueba <- 
    flow_images_from_directory('test/', 
                               prueba,
                               target_size = tamaño,
                               batch_size = 1,
                               shuffle = F)
  aux<-modelo_3 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1)
  prediccion<-cbind(prediccion,aux)
  names(prediccion)[i]<-paste0("prediccion",i)
}
write.csv(prediccion,paste0(carpeta,"/prediccion_3-3.csv"),row.names=F)

# Guardamos gráfica de épocas
ggsave(paste0(carpeta,"/plot_3-3.jpeg"),plot(ajuste),scale=1.5)





modelo_4%>%
  layer_conv_2d(filters = 8, kernel_size = c(5,5), 
                activation = 'relu',
                input_shape = c(ancho, alto, 3), 
                padding ='same',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_1') %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_dense(units = 1, activation = "relu")

modelo_4 %>% compile(
  loss = "binary_crossentropy",
  #puedes probar con otros optimizadores (adam, por ejemplo), 
  #recuerda ajustar lr y momento:
  optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9), 
  metrics = "accuracy"
)

early_stop <- callback_early_stopping(monitor = "val_loss",min_delta=0.005,patience = 5,verbose=1)

ajuste <- modelo_4 %>% fit_generator(
  gen_minilote_entrena,
  validation_data = gen_valida,
  validation_steps = n_valida / 500,
  steps_per_epoch = n_entrena / 32, # entre tamaño de minibatch 
  workers = 4,
  epochs = 40 , # Incrementamos épocas
  verbose = 1,
  callbacks = list(early_stop))

# Exporta métricas
write.csv(as.data.frame(ajuste$metrics),paste0(carpeta,"/metricas_4-4.csv"),row.names=F)

# Predecimos
prueba <- image_data_generator(rescale = 1/255)
gen_prueba <- 
  flow_images_from_directory('test/', 
                             prueba,
                             target_size = tamaño,
                             batch_size = 1,
                             shuffle = F)

prediccion<-data.frame(prediccion1=modelo_4 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1))
for(i in 1:3){
  gen_prueba <- 
    flow_images_from_directory('test/', 
                               prueba,
                               target_size = tamaño,
                               batch_size = 1,
                               shuffle = F)
  aux<-modelo_4%>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1)
  prediccion<-cbind(prediccion,aux)
  names(prediccion)[i]<-paste0("prediccion",i)
}
write.csv(prediccion,paste0(carpeta,"/prediccion_4-4.csv"),row.names=F)

# Guardamos gráfica de épocas
ggsave(paste0(carpeta,"/plot_4-4.jpeg"),plot(ajuste),scale=1.5)




modelo_5%>%
  layer_conv_2d(filters = 8, kernel_size = c(5,5), 
                activation = 'relu',
                input_shape = c(ancho, alto, 3), 
                padding ='same',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_1') %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_conv_2d(filters = 12, kernel_size = c(3,3), 
                activation = 'sigmoid',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_2') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 50, activation = 'selu',
              kernel_regularizer = regularizer_l2(lambda)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

modelo_3 %>% compile(
  loss = "binary_crossentropy",
  #puedes probar con otros optimizadores (adam, por ejemplo), 
  #recuerda ajustar lr y momento:
  optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9), 
  metrics = "accuracy"
)

early_stop <- callback_early_stopping(monitor = "val_loss",min_delta=0.005,patience = 5,verbose=1)

ajuste <- modelo_3 %>% fit_generator(
  gen_minilote_entrena,
  validation_data = gen_valida,
  validation_steps = n_valida / 500,
  steps_per_epoch = n_entrena / 32, # entre tamaño de minibatch 
  workers = 4,
  epochs = 40 , # Incrementamos épocas
  verbose = 1,
  callbacks = list(early_stop))

# Exporta métricas
write.csv(as.data.frame(ajuste$metrics),paste0(carpeta,"/metricas_1-1.csv"),row.names=F)

# Predecimos
prueba <- image_data_generator(rescale = 1/255)
gen_prueba <- 
  flow_images_from_directory('test/', 
                             prueba,
                             target_size = tamaño,
                             batch_size = 1,
                             shuffle = F)

prediccion<-data.frame(prediccion1=modelo_3 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1))
for(i in 1:3){
  gen_prueba <- 
    flow_images_from_directory('test/', 
                               prueba,
                               target_size = tamaño,
                               batch_size = 1,
                               shuffle = F)
  aux<-modelo_3 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1)
  prediccion<-cbind(prediccion,aux)
  names(prediccion)[i]<-paste0("prediccion",i)
}
write.csv(prediccion,paste0(carpeta,"/prediccion_3-3.csv"),row.names=F)

# Guardamos gráfica de épocas
ggsave(paste0(carpeta,"/plot_3-3.jpeg"),plot(ajuste),scale=1.5)






modelo_6%>%
  layer_conv_2d(filters = 8, kernel_size = c(5,5), 
                activation = 'relu',
                input_shape = c(ancho, alto, 3), 
                padding ='same',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_1') %>%
  layer_max_pooling_2d(pool_size = c(5, 5)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_conv_2d(filters = 12, kernel_size = c(5,5), 
                activation = 'sigmoid',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_2') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 50, activation = 'selu',
              kernel_regularizer = regularizer_l2(lambda)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 40, activation = 'sigmoid',
              kernel_regularizer = regularizer_l2(lambda*2)) %>%
  layer_dense(units = 40, activation = 'sigmoid',
              kernel_regularizer = regularizer_l2(lambda*2)) %>%
  layer_dense(units = 1, activation = "sigmoid")

modelo_6 %>% compile(
  loss = "binary_crossentropy",
  #puedes probar con otros optimizadores (adam, por ejemplo), 
  #recuerda ajustar lr y momento:
  optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9), 
  metrics = "accuracy"
)

early_stop <- callback_early_stopping(monitor = "val_loss",min_delta=0.005,patience = 5,verbose=1)

ajuste <- modelo_6 %>% fit_generator(
  gen_minilote_entrena,
  validation_data = gen_valida,
  validation_steps = n_valida / 500,
  steps_per_epoch = n_entrena / 32, # entre tamaño de minibatch 
  workers = 4,
  epochs = 40 , # Incrementamos épocas
  verbose = 1,
  callbacks = list(early_stop))

# Exporta métricas
write.csv(as.data.frame(ajuste$metrics),paste0(carpeta,"/metricas_6-6.csv"),row.names=F)

# Predecimos
prueba <- image_data_generator(rescale = 1/255)
gen_prueba <- 
  flow_images_from_directory('test/', 
                             prueba,
                             target_size = tamaño,
                             batch_size = 1,
                             shuffle = F)

prediccion<-data.frame(prediccion1=modelo_6 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1))
for(i in 1:3){
  gen_prueba <- 
    flow_images_from_directory('test/', 
                               prueba,
                               target_size = tamaño,
                               batch_size = 1,
                               shuffle = F)
  aux<-modelo_6 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1)
  prediccion<-cbind(prediccion,aux)
  names(prediccion)[i]<-paste0("prediccion",i)
}
write.csv(prediccion,paste0(carpeta,"/prediccion_6-6.csv"),row.names=F)

# Guardamos gráfica de épocas
ggsave(paste0(carpeta,"/plot_6-6.jpeg"),plot(ajuste),scale=1.5)




modelo_7%>%
  layer_conv_2d(filters = 8, kernel_size = c(5,5), 
                activation = 'relu',
                input_shape = c(ancho, alto, 3), 
                padding ='same',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_1') %>%
  layer_max_pooling_2d(pool_size = c(5, 5)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_conv_2d(filters = 12, kernel_size = c(5,5), 
                activation = 'sigmoid',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_2') %>% 
  layer_flatten() %>% 
  layer_dense(units = 50, activation = 'selu',
              kernel_regularizer = regularizer_l2(lambda)) %>%
  layer_dropout(rate = 0.5) %>%
    layer_dense(units = 40, activation = 'relu',
              kernel_regularizer = regularizer_l2(lambda*2)) %>%
        layer_dense(units = 40, activation = 'sigmoid',
              kernel_regularizer = regularizer_l2(lambda*2)) %>%
                layer_dense(units = 40, activation = 'sigmoid',
              kernel_regularizer = regularizer_l2(lambda*2)) %>%
                        layer_dense(units = 20, activation = 'sigmoid',
              kernel_regularizer = regularizer_l2(lambda*2)) %>%
  layer_dense(units = 1, activation = "sigmoid")

modelo_7 %>% compile(
  loss = "binary_crossentropy",
  #puedes probar con otros optimizadores (adam, por ejemplo), 
  #recuerda ajustar lr y momento:
  optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9), 
  metrics = "accuracy"
)

early_stop <- callback_early_stopping(monitor = "val_loss",min_delta=0.005,patience = 5,verbose=1)

ajuste <- modelo_7 %>% fit_generator(
  gen_minilote_entrena,
  validation_data = gen_valida,
  validation_steps = n_valida / 500,
  steps_per_epoch = n_entrena / 32, # entre tamaño de minibatch 
  workers = 4,
  epochs = 40 , # Incrementamos épocas
  verbose = 1,
  callbacks = list(early_stop))

# Exporta métricas
write.csv(as.data.frame(ajuste$metrics),paste0(carpeta,"/metricas_7-7.csv"),row.names=F)

# Predecimos
prueba <- image_data_generator(rescale = 1/255)
gen_prueba <- 
  flow_images_from_directory('test/', 
                             prueba,
                             target_size = tamaño,
                             batch_size = 1,
                             shuffle = F)

prediccion<-data.frame(prediccion1=modelo_7 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1))
for(i in 1:3){
  gen_prueba <- 
    flow_images_from_directory('test/', 
                               prueba,
                               target_size = tamaño,
                               batch_size = 1,
                               shuffle = F)
  aux<-modelo_7 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1)
  prediccion<-cbind(prediccion,aux)
  names(prediccion)[i]<-paste0("prediccion",i)
}
write.csv(prediccion,paste0(carpeta,"/prediccion_7-7.csv"),row.names=F)

# Guardamos gráfica de épocas
ggsave(paste0(carpeta,"/plot_7-7.jpeg"),plot(ajuste),scale=1.5)






modelo_8%>%
  layer_conv_2d(filters = 8, kernel_size = c(3,3), 
                activation = 'relu',
                input_shape = c(ancho, alto, 3), 
                padding ='same',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_1') %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_conv_2d(filters = 12, kernel_size = c(3,3), 
                activation = 'sigmoid',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_2') %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>%
    layer_dense(units = 40, activation = 'relu',
              kernel_regularizer = regularizer_l2(lambda*2)) %>%
    layer_dense(units = 40, activation = 'sigmoid',
              kernel_regularizer = regularizer_l2(lambda*2)) %>%
    layer_dense(units = 40, activation = 'sigmoid',
              kernel_regularizer = regularizer_l2(lambda*2)) %>%
    layer_dense(units = 20, activation = 'sigmoid',
              kernel_regularizer = regularizer_l2(lambda*2)) %>%
  layer_dense(units = 1, activation = "sigmoid")

modelo_8 %>% compile(
  loss = "binary_crossentropy",
  #puedes probar con otros optimizadores (adam, por ejemplo), 
  #recuerda ajustar lr y momento:
  optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9), 
  metrics = "accuracy"
)

early_stop <- callback_early_stopping(monitor = "val_loss",min_delta=0.005,patience = 5,verbose=1)

ajuste <- modelo_8 %>% fit_generator(
  gen_minilote_entrena,
  validation_data = gen_valida,
  validation_steps = n_valida / 500,
  steps_per_epoch = n_entrena / 32, # entre tamaño de minibatch 
  workers = 4,
  epochs = 40 , # Incrementamos épocas
  verbose = 1,
  callbacks = list(early_stop))

# Exporta métricas
write.csv(as.data.frame(ajuste$metrics),paste0(carpeta,"/metricas_8-8.csv"),row.names=F)

# Predecimos
prueba <- image_data_generator(rescale = 1/255)
gen_prueba <- 
  flow_images_from_directory('test/', 
                             prueba,
                             target_size = tamaño,
                             batch_size = 1,
                             shuffle = F)

prediccion<-data.frame(prediccion1=modelo_8 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1))
for(i in 1:3){
  gen_prueba <- 
    flow_images_from_directory('test/', 
                               prueba,
                               target_size = tamaño,
                               batch_size = 1,
                               shuffle = F)
  aux<-modelo_8 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1)
  prediccion<-cbind(prediccion,aux)
  names(prediccion)[i]<-paste0("prediccion",i)
}
write.csv(prediccion,paste0(carpeta,"/prediccion_8-8.csv"),row.names=F)

# Guardamos gráfica de épocas
ggsave(paste0(carpeta,"/plot_8-8.jpeg"),plot(ajuste),scale=1.5)







modelo_9%>%
  layer_conv_2d(filters = 8, kernel_size = c(5,5), 
                activation = 'relu',
                input_shape = c(ancho, alto, 3), 
                padding ='same',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_1') %>%
  layer_max_pooling_2d(filters = 12, kernel_size = c(5,5), 
                activation = 'sigmoid',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_2') %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_conv_2d(filters = 12, kernel_size = c(5,5), 
                activation = 'selu',
                kernel_regularizer = regularizer_l2(2*lambda),
                name = 'conv_3') %>% 
  layer_max_pooling_2d(filters = 12, kernel_size = c(5,5), 
                activation = 'sigmoid',
                kernel_regularizer = regularizer_l2(2*lambda),
                name = 'conv_4') %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 40, activation = 'sigmoid',
              kernel_regularizer = regularizer_l2(lambda*2)) %>%
  layer_dense(units = 1, activation = "sigmoid")

modelo_9 %>% compile(
  loss = "binary_crossentropy",
  #puedes probar con otros optimizadores (adam, por ejemplo), 
  #recuerda ajustar lr y momento:
  optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9), 
  metrics = "accuracy"
)

early_stop <- callback_early_stopping(monitor = "val_loss",min_delta=0.005,patience = 5,verbose=1)

ajuste <- modelo_9 %>% fit_generator(
  gen_minilote_entrena,
  validation_data = gen_valida,
  validation_steps = n_valida / 500,
  steps_per_epoch = n_entrena / 32, # entre tamaño de minibatch 
  workers = 4,
  epochs = 40 , # Incrementamos épocas
  verbose = 1,
  callbacks = list(early_stop))

# Exporta métricas
write.csv(as.data.frame(ajuste$metrics),paste0(carpeta,"/metricas_9-9.csv"),row.names=F)

# Predecimos
prueba <- image_data_generator(rescale = 1/255)
gen_prueba <- 
  flow_images_from_directory('test/', 
                             prueba,
                             target_size = tamaño,
                             batch_size = 1,
                             shuffle = F)

prediccion<-data.frame(prediccion1=modelo_9 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1))
for(i in 1:3){
  gen_prueba <- 
    flow_images_from_directory('test/', 
                               prueba,
                               target_size = tamaño,
                               batch_size = 1,
                               shuffle = F)
  aux<-modelo_9 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1)
  prediccion<-cbind(prediccion,aux)
  names(prediccion)[i]<-paste0("prediccion",i)
}
write.csv(prediccion,paste0(carpeta,"/prediccion_9-9.csv"),row.names=F)

# Guardamos gráfica de épocas
ggsave(paste0(carpeta,"/plot_9-9.jpeg"),plot(ajuste),scale=1.5)




modelo_10%>%
  layer_conv_2d(filters = 8, kernel_size = c(5,5), 
                activation = 'relu',
                input_shape = c(ancho, alto, 3), 
                padding ='same',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_1') %>%
  layer_max_pooling_2d(filters = 8, kernel_size = c(5,5), 
                activation = 'sigmoid',
                kernel_regularizer = regularizer_l2(lambda),
                name = 'conv_2') %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_conv_2d(filters = 8, kernel_size = c(3,3), 
                activation = 'selu',
                kernel_regularizer = regularizer_l2(2*lambda),
                name = 'conv_3') %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 40, activation = 'sigmoid',
              kernel_regularizer = regularizer_l2(lambda*2)) %>%
  layer_dense(units = 20, activation = 'relu',
              kernel_regularizer = regularizer_l2(lambda*2)) %>%
  layer_dense(units = 20, activation = 'selu',
              kernel_regularizer = regularizer_l2(lambda*2)) %>%
  layer_dense(units = 1, activation = "sigmoid")

modelo_10 %>% compile(
  loss = "binary_crossentropy",
  #puedes probar con otros optimizadores (adam, por ejemplo), 
  #recuerda ajustar lr y momento:
  optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9), 
  metrics = "accuracy"
)

early_stop <- callback_early_stopping(monitor = "val_loss",min_delta=0.005,patience = 5,verbose=1)

ajuste <- modelo_10 %>% fit_generator(
  gen_minilote_entrena,
  validation_data = gen_valida,
  validation_steps = n_valida / 500,
  steps_per_epoch = n_entrena / 32, # entre tamaño de minibatch 
  workers = 4,
  epochs = 40 , # Incrementamos épocas
  verbose = 1,
  callbacks = list(early_stop))

# Exporta métricas
write.csv(as.data.frame(ajuste$metrics),paste0(carpeta,"/metricas_10-10.csv"),row.names=F)

# Predecimos
prueba <- image_data_generator(rescale = 1/255)
gen_prueba <- 
  flow_images_from_directory('test/', 
                             prueba,
                             target_size = tamaño,
                             batch_size = 1,
                             shuffle = F)

prediccion<-data.frame(prediccion1=modelo_10 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1))
for(i in 1:3){
  gen_prueba <- 
    flow_images_from_directory('test/', 
                               prueba,
                               target_size = tamaño,
                               batch_size = 1,
                               shuffle = F)
  aux<-modelo_10 %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1)
  prediccion<-cbind(prediccion,aux)
  names(prediccion)[i]<-paste0("prediccion",i)
}
write.csv(prediccion,paste0(carpeta,"/prediccion_10-10.csv"),row.names=F)

# Guardamos gráfica de épocas
ggsave(paste0(carpeta,"/plot_10-10.jpeg"),plot(ajuste),scale=1.5)