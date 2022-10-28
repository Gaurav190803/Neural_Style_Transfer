import tensorflow as tf
from helper_fuctions import *
from tqdm import tqdm
import os
class VGG(tf.keras.Model):
    def __init__(self):
        vgg = tf.keras.applications.vgg19.VGG19(weights='imagenet',include_top=False)
        vgg.trainable = False
        super(VGG,self).__init__()
        self.choose = "style"
        self.chosen_style = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
        self.chosen_content = 'block5_conv2'
        self.model = vgg
    def call(self,x):
        features = []
        if self.choose=='style':
            for l in self.model.layers:
                x = l(x)
                if l.name in self.chosen_style:
                    features.append(x)
        elif self.choose=='content':
            for l in self.model.layers:
                x = l(x)
                if l.name is self.chosen_content:
                    features = x
        return features

    def transform(self,
    content_image,
    style_image,
    optimizer = tf.keras.optimizers.Adam(),
    epochs = 400,
    content_weight = 2.5e-8,
    style_weight = 1e-6/5,
    decay_rate = 0.96,
    decay_step = 100,
    save_best_pic = False,
    save_best_path = None,
    experiment_name = None):
        combination_img = tf.Variable(content_image)
        pbar = tqdm(range(1,epochs+1),desc = "NST",unit = " epochs",ncols=150)
        best_loss = 100000
        for epoch in pbar:
            optimizer.learning_rate = 10*(tf.pow(decay_rate,(epoch/decay_step)))
            with tf.GradientTape() as tape:
                tape.watch(combination_img)
                self.choose = 'style'
                style_features = self(style_image)
                generated_style_features = self(combination_img)
                self.choose = 'content'
                content_features = self(content_image)
                generated_content_features = self(combination_img)
                # Content loss...
                content_loss = content_fn(content_features,generated_content_features,content_weight)
                # Style loss...
                style_loss = style_fn(style_features,generated_style_features,style_weight)
                total_loss = style_loss + content_loss
            
            if(save_best_pic):
                if(total_loss < best_loss):
                    best_loss = total_loss
                    try:
                        os.remove(save_best_path + experiment_name + ".jpg")
                    except:
                        pass
                    tf.keras.utils.save_img(
                        path = save_best_path + experiment_name + ".jpg",
                        x = deprocess_image(combination_img.numpy()),
                        data_format = "channels_last",
                    )


            pbar.set_postfix_str(f"content loss: {content_loss:.4f} | style loss: {style_loss:.4f} | total loss: {total_loss:.4f}")
            grads = tape.gradient(total_loss,[combination_img])
            optimizer.apply_gradients(zip(grads,[combination_img]))
        return combination_img