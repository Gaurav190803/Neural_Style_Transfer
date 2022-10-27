import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def load_img(img_path,img_rows):
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    (height, width, channels) = img.shape
    img_cols = img_rows*(width/height)
    img = tf.image.resize(img,size=(img_rows,int(img_cols)))
    return img

def preprocess_img(img):
    img = tf.expand_dims(img,axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def show_mul(*images,figsize=None):
    plots = len(images)
    iter = 0
    fig, objects = plt.subplots(1,plots,figsize=figsize)
    for i in tuple(objects):
        i.imshow(images[iter])
        i.axis(False)
        iter+=1
    plt.show()
    
def content_fn(base_img,combination_img):
    return tf.reduce_sum(tf.square(base_img-combination_img))

def gram_matrix(img):
    img = tf.transpose(img,(2,0,1))
    img = tf.reshape(img, shape=(tf.shape(img)[0],-1))
    matrix = tf.matmul(img,img,transpose_b=True)
    return matrix
    
def style_fn(style_img,combination_img):
    style_gram = gram_matrix(style_img)
    combination_gram = gram_matrix(combination_img)
    return tf.reduce_sum(tf.square(style_gram-combination_gram)) / (4*9*(400**4))
    
def deprocess_image(img):
    img = tf.squeeze(img).numpy()
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img

def show(image ,figsize = (10,10),deprocess = True):
    plt.figure(figsize = figsize)
    plt.axis(False)
    if(deprocess):
        plt.imshow(deprocess_image(image.numpy()))
    else:
        plt.imshow(image.numpy())
