import numpy as np
import tensorflow as tf
import cv2
import os

tf.compat.v1.disable_eager_execution()
#export PYTHONPATH="${PYTHONPATH}:/workspace/SegGradCAM/SegGradCAM"

from seggradcam.seggradcam import SegGradCAM, PixelRoI

#Effective Receptive Field calculation
prop_to_layers = ["conv2d_1", "conv2d_3", "conv2d_5", "conv2d_7", "conv2d_15", "conv2d_17", "conv2d_19", "conv2d_21", "conv2d_22"]
img_names = os.listdir("/workspace/Vitis-AI/tutorials/SegGradCAM/Cube_224_416_TC_PN_Npy/")
img_names.sort()

#Parámetros comunes
clip_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
clip_max = [0.111, 0.119, 0.133, 0.125, 0.092, 0.127, 0.149, 0.143, 0.119, 0.082, 0.116, 0.135, 0.118, 0.101, 0.083, 0.101, 0.105, 0.114, 0.102, 0.09, 0.118, 0.13, 0.132, 0.125, 0.088]
norm_min = clip_min
norm_max = np.array(clip_max) / 0.9921875
cls = 2
n_classes = 6
roi_value = (175, 209)
normalize = False
abs_w = False
posit_w = False

float_model_name = "/workspace/Vitis-AI/tutorials/SegGradCAM/float_model_todo_224_416_TC_PN_5_3_32_fold4_init1_explicit_different_norm_wd_wd_tf2_train.h5"
trained_model = tf.keras.models.load_model(float_model_name)

gt =  np.array(cv2.imread('./' + 'nf1111_576_gt.png', 1)[:, : , 0])
vis =  np.array(cv2.imread('./' + 'nf1111_576_vis.png', 1)[:, : , 0])

outfolder = "./output/HSI-Drive/HSI"

for img_name in img_names[0:1]: #11 #35
       img_name = img_name[-30:-10]
       for prop_to_layer in prop_to_layers[0:-1]:

              #HYPERSPECTRAL
              #Interesa ver cómo cambia la salida
              prop_from_layer = trained_model.layers[-1].name #conv2d_22
              #En función de lo que cambia la activación de esta capa.

              #Hay que elegir una imagen (cubo y gt)
              img = np.load('./Cube_224_416_TC_PN_Npy/' + img_name + '_TC_PN.npy')
              img = np.clip(img, clip_min, clip_max)

              for i in range(25):
                     img[:, :, i] = 2*( (img[:, :, i] - norm_min[i]) / (norm_max[i] - norm_min[i]) ) - 1


              # PIXEL
              # create a SegGradCAM object
              roi=PixelRoI(*roi_value,img)

              pixsgc = SegGradCAM(trained_model, img, cls,  prop_to_layer, prop_from_layer, image_id = img_name[7:] , roi=roi, normalize=normalize, abs_w=abs_w, posit_w=posit_w, erf_output_irt_input=False, erf_conv_irt_input=True)
              pixsgc.SGC()

