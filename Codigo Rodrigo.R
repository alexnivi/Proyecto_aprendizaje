# Carga de paquetes
library(tidyverse)
library(keras)

# Declaramos workspace
setwd("~/Dropbox/Maestria Ciencia de Datos 2018/Aprendizaje de maquina/Examen1/Proyecto_Aprendizaje")
rm(list=ls())

############## Muestras (Opcional) ############## 

# # Creamos muestra de entrenamiento, muestra de validación y muestra de prueba
# # /train0 son las imágenes originales
# # Observamos los archivos en cada carpeta
# totales<-c(paste0("train0/gato/",list.files("train0/gato")),paste0("train0/perro/",list.files("train0/perro")))
# set.seed(1567) # Se puede modificar semilla
# entrenamiento<-sample(totales,0.75*length(totales))
# validacion<-setdiff(totales,entrenamiento)
# rm(totales)
# 
# # Pordemos eliminar si ya hay directorios existentes y para probar con una muestra diferente
# unlink(c("train","validation"),recursive=T)
# 
# # Generamos carpetas
# carpetas<-c("train","train/gato","train/perro","validation","validation/gato","validation/perro")
# sapply(carpetas, dir.create)
# 
# # Copiamos archivos a carpetas
# file.copy(entrenamiento[grep("cat",entrenamiento)],"train/gato")
# file.copy(entrenamiento[grep("dog",entrenamiento)],"train/perro")
# file.copy(validacion[grep("cat",validacion)],"validation/gato")
# file.copy(validacion[grep("dog",validacion)],"validation/perro")
# rm(entrenamiento,validacion,carpetas)

############## Primer intento, 16-Oct ############## 
#   # Preprocesamiento de imagenes #
# 
# # Normalizamos pixeles a [0,1] (los pixeles originalmente vienen entre 0 y 255)
# # Preprocesamiento de imagenes. Genera minilotes de datos de imagenes.
# entrena <- image_data_generator(rescale = 1/255)
# valida <- image_data_generator(rescale = 1/255)
# prueba <- image_data_generator(rescale = 1/255)
# 
# # Escalamos el tamaño de las imágenes
# ancho <- 50
# alto <- 50
# tamaño <- c(ancho, alto)
# 
# # Generamos minilotes de entrenamiento y validación en ESCALA DE GRISES
# gen_minilote_entrena <- 
#   flow_images_from_directory('train/', 
#                              entrena, 
#                              target_size = tamaño,
#                              color_mode="grayscale", # Cambia a blanco y negro
#                              batch_size = 32,
#                              class_mode = "binary",
#                              classes = c("perro", "gato"),
#                              shuffle = TRUE,
#                              seed = 42711)
# gen_valida <- 
#   flow_images_from_directory('validation/', 
#                              valida,
#                              color_mode="grayscale",
#                              target_size = tamaño,
#                              batch_size = 500,
#                              class_mode = "binary",
#                              shuffle = FALSE,
#                              classes = c("perro", "gato"))
# 
# # Creamos tablas de los objetos por carpeta
# # 0 corresponde a perro y 1 corresponde a gato
# table(gen_minilote_entrena$classes)
# table(gen_valida$classes)
# 
# # Número de observaciones por muestras
# n_entrena <- gen_minilote_entrena$n
# n_valida <- gen_valida$n
# 
# # Índices de cada objeto
# indices <- gen_minilote_entrena$class_indices
# 
#   # Prueba de modelo # 
# # Red simple
# modelo <- keras_model_sequential()
# modelo %>% 
#   layer_conv_2d(filter = 16, kernel_size = c(3,3), 
#                 input_shape = c(ancho, alto, 1),
#                 activation = "relu") %>%
#   layer_max_pooling_2d(pool_size = c(2,2)) %>%
#   layer_dropout(0.2) %>% 
#   layer_flatten() %>%
#   layer_dense(units = 50, activation="relu") %>% 
#   layer_dropout(0.2) %>% 
#   layer_dense(units = 1, activation = "sigmoid") 
# 
# modelo %>% compile(
#   loss = "binary_crossentropy",
#   #puedes probar con otros optimizadores (adam, por ejemplo), 
#   #recuerda ajustar lr y momento:
#   optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9), 
#   metrics = "accuracy"
# )
# 
# ajuste <- modelo %>% fit_generator(
#   gen_minilote_entrena,
#   validation_data = gen_valida,
#   validation_steps = n_valida / 500,
#   steps_per_epoch = n_entrena / 32, # entre tamaño de minibatch 
#   workers = 4,
#   epochs = 20,
#   verbose = 1)
# 
# # Predecimos
# prueba <- image_data_generator(rescale = 1/255)
# gen_prueba <- 
#   flow_images_from_directory('test/', 
#                              prueba,
#                              color_mode="grayscale",
#                              target_size = tamaño,
#                              batch_size = 1,
#                              class_mode = "binary",
#                              shuffle = F)
# 
# prediccion <- modelo %>% predict_generator(gen_prueba, step = 6887, verbose = 1)
# prediccion<-as.tibble(prediccion) %>%
#   rename(probabilidad = V1) %>%
#   mutate(indice = 1:nrow(prediccion))
# prediccion<-prediccion[c(2,1)]
# write.csv(prediccion,"2018-10-16_prediccion_1.csv",row.names=F)

############## Segundo intento, 16-Oct ############## 

# # Preprocesamiento de imagenes #
# 
# # Normalizamos pixeles a [0,1] (los pixeles originalmente vienen entre 0 y 255)
# # Preprocesamiento de imagenes. Genera minilotes de datos de imagenes.
# entrena <- image_data_generator(rescale = 1/255)
# valida <- image_data_generator(rescale = 1/255)
# prueba <- image_data_generator(rescale = 1/255)
# 
# # Escalamos el tamaño de las imágenes
# ancho <- 50
# alto <- 50
# tamaño <- c(ancho, alto)
# 
# # Generamos minilotes de entrenamiento y validación en A COLOR
# gen_minilote_entrena <- 
#   flow_images_from_directory('train/', 
#                              entrena, 
#                              target_size = tamaño,
#                              batch_size = 32,
#                              class_mode = "binary",
#                              classes = c("perro", "gato"),
#                              shuffle = TRUE,
#                              seed = 42711)
# gen_valida <- 
#   flow_images_from_directory('validation/', 
#                              valida,
#                              target_size = tamaño,
#                              batch_size = 500,
#                              class_mode = "binary",
#                              shuffle = FALSE,
#                              classes = c("perro", "gato"))
# 
# # Creamos tablas de los objetos por carpeta
# # 0 corresponde a perro y 1 corresponde a gato
# table(gen_minilote_entrena$classes)
# table(gen_valida$classes)
# 
# # Número de observaciones por muestras
# n_entrena <- gen_minilote_entrena$n
# n_valida <- gen_valida$n
# 
# # Índices de cada objeto
# indices <- gen_minilote_entrena$class_indices
# 
# # Prueba de modelo # 
# # Red simple
# modelo <- keras_model_sequential()
# modelo %>% 
#   layer_conv_2d(filter = 16, kernel_size = c(3,3), 
#                 input_shape = c(ancho, alto, 3), # Cambiamos 1 por 3 porque son 3 colores
#                 activation = "relu") %>%
#   layer_max_pooling_2d(pool_size = c(2,2)) %>%
#   layer_dropout(0.2) %>% 
#   layer_flatten() %>%
#   layer_dense(units = 50, activation="relu") %>% 
#   layer_dropout(0.2) %>% 
#   layer_dense(units = 1, activation = "sigmoid") 
# 
# modelo %>% compile(
#   loss = "binary_crossentropy",
#   #puedes probar con otros optimizadores (adam, por ejemplo), 
#   #recuerda ajustar lr y momento:
#   optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9), 
#   metrics = "accuracy"
# )
# 
# ajuste <- modelo %>% fit_generator(
#   gen_minilote_entrena,
#   validation_data = gen_valida,
#   validation_steps = n_valida / 500,
#   steps_per_epoch = n_entrena / 32, # entre tamaño de minibatch 
#   workers = 4,
#   epochs = 20, 
#   verbose = 1)
# 
# # Predecimos
# prueba <- image_data_generator(rescale = 1/255)
# gen_prueba <- 
#   flow_images_from_directory('test/', 
#                              prueba,
#                              target_size = tamaño,
#                              batch_size = 1,
#                              class_mode = "binary",
#                              shuffle = F)
# 
# prediccion <- modelo %>% predict_generator(gen_prueba, step = 6887, verbose = 1)
# prediccion<-as.tibble(prediccion) %>%
#   rename(probabilidad = V1) %>%
#   mutate(indice = 1:nrow(prediccion))
# prediccion<-prediccion[c(2,1)]
# write.csv(prediccion,"2018-10-16_prediccion_2.csv",row.names=F)

############## Primer intento, 17-Oct ############## 

carpeta<-paste0("Bases/",Sys.Date())
dir.create(carpeta)

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
# Red simple
modelo <- keras_model_sequential()
modelo %>% 
  layer_conv_2d(filter = 32, kernel_size = c(3,3),  # Aumentamos el numero de filtros en segunda capa
                input_shape = c(ancho, alto, 3), # Cambiamos 1 por 3 porque son 3 colores
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.2) %>% 
  layer_flatten() %>%
  layer_dense(units = 60, activation="relu") %>% # Aumentamos el número de variables en la cuarta capa
  layer_dropout(0.2) %>% 
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
  epochs = 40, # Incrementamos épocas
  verbose = 1,
  callbacks = list(early_stop))

# Exporta métricas
write.csv(as.data.frame(ajuste$metrics),paste0(carpeta,"/metricas_1.csv"),row.names=F)

# Predecimos
prueba <- image_data_generator(rescale = 1/255)
gen_prueba <- 
  flow_images_from_directory('test/', 
                             prueba,
                             target_size = tamaño,
                             batch_size = 1,
                             shuffle = F)

prediccion<-data.frame(prediccion1=modelo %>% predict_generator(gen_prueba, step = 6887, verbose = 1,workers=1))
for(i in 1:5){
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
write.csv(prediccion,paste0(carpeta,"/prediccion_2.csv"),row.names=F)

# Guardamos gráfica de épocas
ggsave(paste0(carpeta,"/plot_1.jpeg"),plot(ajuste),scale=1.5)
