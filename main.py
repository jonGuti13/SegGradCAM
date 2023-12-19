import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import unet_train_tf2
import cv2

tf.compat.v1.disable_eager_execution()
#export PYTHONPATH="${PYTHONPATH}:/workspace/SegGradCAM/SegGradCAM"

from seggradcam.seggradcam import SegGradCAM, SuperRoI, ClassRoI, PixelRoI, BiasRoI
from seggradcam.visualize_sgc import SegGradCAMplot

float_model_name = "/workspace/tutorials/SegGradCAM/float_model_todo_224_416_TC_PN_5_3_32_fold4_init1_explicit_different_norm_wd_wd_tf2_train.h5"
trained_model = tf.keras.models.load_model(float_model_name)
#print(trained_model.summary())

#Interesa ver cómo cambia la salida
prop_from_layer = trained_model.layers[-1].name #conv2d_22
#En función de lo que cambia la activación de esta capa.
prop_to_layer = 'conv2d_5'

#Hay que elegir una imagen (cubo y gt)
img = np.load('./nf1111_576_TC_PN.npy')
clip_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
clip_max = [0.111, 0.119, 0.133, 0.125, 0.092, 0.127, 0.149, 0.143, 0.119, 0.082, 0.116, 0.135, 0.118, 0.101, 0.083, 0.101, 0.105, 0.114, 0.102, 0.09, 0.118, 0.13, 0.132, 0.125, 0.088]
img = np.clip(img, clip_min, clip_max)
norm_min = clip_min
norm_max = np.array(clip_max) / 0.9921875

for i in range(25):
       img[:, :, i] = 2*( (img[:, :, i] - norm_min[i]) / (norm_max[i] - norm_min[i]) ) - 1

gt =  np.array(cv2.imread('./nf1111_576_gt.png', 1)[:, : , 0])
vis =  np.array(cv2.imread('./nf1111_576_vis.png', 1)[:, : , 0])

cls = 2

## PIXEL
#create a SegGradCAM object
roi=PixelRoI(175,209,img)

pixsgc = SegGradCAM(trained_model, img, cls,  prop_to_layer,prop_from_layer, roi=roi, normalize=False, abs_w=False, posit_w=False)
pixsgc.SGC()

# create an object with plotting functionality
plotter = SegGradCAMplot(pixsgc, model=trained_model,n_classes=6,outfolder="./output/HSI-Drive", gt = gt, vis=vis)

# plot explanations on 1 picture
plotter.explainPixel()


## CLASE
clsroi = ClassRoI(model=trained_model,image=img,cls=cls)
newsgc = SegGradCAM(trained_model, img, cls, prop_to_layer,  prop_from_layer, roi=clsroi,
                 normalize=False, abs_w=False, posit_w=False)
newsgc.SGC()

# create an object with plotting functionality
plotter = SegGradCAMplot(newsgc,model=trained_model,n_classes=6,outfolder="./", gt = gt, vis=vis)
# plot explanations on 1 picture
plotter.explainClass()