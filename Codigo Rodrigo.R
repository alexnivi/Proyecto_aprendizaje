# Carga de paquetes
library(tidyverse)
library(keras)

# Declaramos workspace
#setwd("~/Dropbox/Maestria Ciencia de Datos 2018/Aprendizaje de maquina/Examen1/Proyecto_Aprendizaje")
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

# ############## Preprocesamiento de imagenes ############## 

# Normalizamos pixeles a [0,1] (los pixeles originalmente vienen entre 0 y 255)
# Preprocesamiento de imagenes. Genera minilotes de datos de imagenes.
entrena <- image_data_generator(rescale = 1/255)
valida <- image_data_generator(rescale = 1/255)

# Escalamos el tamaño de las imágenes
ancho <- 50
alto <- 50
tamaño <- c(ancho, alto)

# Generamos minilotes de entrenamiento y validación en escala de grises
gen_minilote_entrena <- 
  flow_images_from_directory('train/', 
                             entrena, 
                             target_size = tamaño,
                             color_mode="grayscale", # Cambia a blanco y negro
                             batch_size = 32,
                             class_mode = "binary",
                             classes = c("perro", "gato"),
                             shuffle = TRUE,
                             seed = 42711)
gen_valida <- 
  flow_images_from_directory('validation/', 
                             valida,
                             color_mode="grayscale",
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

# Generamos minilotes de entrenamiento y validación a color
gen_minilote_entrena_col <- 
  flow_images_from_directory('train/', 
                             entrena, 
                             target_size = tamaño,
                             batch_size = 32,
                             class_mode = "binary",
                             classes = c("perro", "gato"),
                             shuffle = TRUE,
                             seed = 42711)
gen_valida_col <- 
  flow_images_from_directory('validation/', 
                             valida,
                             color_mode="grayscale",
                             target_size = tamaño,
                             batch_size = 500,
                             class_mode = "binary",
                             shuffle = FALSE,
                             classes = c("perro", "gato"))

# Creamos tablas de los objetos por carpeta
# 0 corresponde a perro y 1 corresponde a gato
table(gen_minilote_entrena_col$classes)
table(gen_valida_col$classes)

# Número de observaciones por muestras
n_entrena_col <- gen_minilote_entrena_col$n
n_valida_col <- gen_valida_col$n

# Índices de cada objeto
indices_col <- gen_minilote_entrena_col$class_indices