import numpy as np
import tensorflow as tf
import cv2


tf.compat.v1.disable_eager_execution()
#export PYTHONPATH="${PYTHONPATH}:/workspace/SegGradCAM/SegGradCAM"

from seggradcam.seggradcam import SegGradCAM, ClassRoI, PixelRoI
from seggradcam.visualize_sgc import SegGradCAMplot

#Parámetros comunes
prop_to_layer = "conv2d_1"
img_name = "nf1111_576"
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
erf_output_irt_input=False
erf_conv_irt_input=False

#HYPERSPECTRAL
#Hay que elegir una imagen (cubo y gt)
img = np.load('./Cube_224_416_TC_PN_Npy/' + img_name + '_TC_PN.npy')
img = np.clip(img, clip_min, clip_max)

for i in range(25):
       img[:, :, i] = 2*( (img[:, :, i] - norm_min[i]) / (norm_max[i] - norm_min[i]) ) - 1

gt =  np.array(cv2.imread('./' + img_name + '_gt.png', 1)[:, : , 0])
vis =  np.array(cv2.imread('./' + img_name + '_vis.png', 1)[:, : , 0])

float_model_name = "/workspace/Vitis-AI/tutorials/SegGradCAM/float_model_todo_224_416_TC_PN_5_3_32_fold4_init1_explicit_different_norm_wd_wd_tf2_train.h5"
trained_model = tf.keras.models.load_model(float_model_name)
#Interesa ver cómo cambia la salida
prop_from_layer = trained_model.layers[-1].name #conv2d_22
#En función de lo que cambia la activación de esta capa.

outfolder = "./output/HSI-Drive/HSI"

# PIXEL
# create a SegGradCAM object
roi=PixelRoI(*roi_value,img)

pixsgc = SegGradCAM(trained_model, img, cls,  prop_to_layer, prop_from_layer, image_id = img_name[7:], roi=roi, normalize=normalize, abs_w=abs_w, posit_w=posit_w, erf_output_irt_input=erf_output_irt_input, erf_conv_irt_input=erf_conv_irt_input)
pixsgc.SGC()

#create an object with plotting functionality
plotter = SegGradCAMplot(pixsgc, model=trained_model,n_classes=n_classes,outfolder=outfolder, gt = gt, vis=vis)

#plot explanations on 1 picture
plotter.explainPixel()

## CLASE
clsroi = ClassRoI(model=trained_model,image=img,cls=cls)
newsgc = SegGradCAM(trained_model, img, cls, prop_to_layer, prop_from_layer, image_id = img_name[7:], roi=clsroi, normalize=normalize, abs_w=abs_w, posit_w=posit_w, erf_output_irt_input=False, erf_conv_irt_input=False)
newsgc.SGC()

# create an object with plotting functionality
plotter = SegGradCAMplot(newsgc,model=trained_model,n_classes=n_classes,outfolder=outfolder, gt = gt, vis=vis)
# plot explanations on 1 picture
plotter.explainClass()


# #MONOCROMO
# #Hay que elegir una imagen (cubo y gt)
# img = np.load('./' + img_name + '_TC_PN_Channel_12.npy')
# clip_min = clip_min[12]
# clip_max = clip_max[12]
# img = np.clip(img, clip_min, clip_max)
# norm_min = clip_min
# norm_max = np.array(clip_max) / 0.9921875

# img[:, :] = 2*( (img[:, :] - norm_min) / (norm_max - norm_min) ) - 1

# gt =  np.array(cv2.imread('./' + img_name + '_gt.png', 1)[:, : , 0])
# vis =  np.array(cv2.imread('./' + img_name + '_vis.png', 1)[:, : , 0])

# float_model_name = "/workspace/tutorials/SegGradCAM/float_model_todo_224_416_TC_PN_channel_12_5_3_32_fold4_init1_explicit_norm_tf2_train.h5"
# trained_model = tf.keras.models.load_model(float_model_name, compile=False)
# #Interesa ver cómo cambia la salida
# prop_from_layer = trained_model.layers[-1].name #conv2d_22
# #En función de lo que cambia la activación de esta capa.

# outfolder = "./output/HSI-Drive/Mono"

# ## PIXEL
# #create a SegGradCAM object
# roi=PixelRoI(*roi_value,img)

# pixsgc = SegGradCAM(trained_model, img, cls,  prop_to_layer, prop_from_layer, image_id = img_name[7:], roi=roi, normalize=normalize, abs_w=abs_w, posit_w=posit_w, erf_output_irt_input=erf_output_irt_input, erf_conv_irt_input=erf_conv_irt_input)
# pixsgc.SGC()

# # create an object with plotting functionality
# plotter = SegGradCAMplot(pixsgc, model=trained_model,n_classes=n_classes,outfolder=outfolder, gt = gt, vis=vis)

# # plot explanations on 1 picture
# plotter.explainPixel()

# ## CLASE
# clsroi = ClassRoI(model=trained_model,image=img,cls=cls)
# newsgc = SegGradCAM(trained_model, img, cls, prop_to_layer, prop_from_layer, image_id = img_name[7:], roi=clsroi, normalize=normalize, abs_w=abs_w, posit_w=posit_w, erf_output_irt_input=False, erf_conv_irt_input=False)
# newsgc.SGC()

# # create an object with plotting functionality
# plotter = SegGradCAMplot(newsgc,model=trained_model,n_classes=n_classes,outfolder=outfolder, gt = gt, vis=vis)
# # plot explanations on 1 picture
# plotter.explainClass()