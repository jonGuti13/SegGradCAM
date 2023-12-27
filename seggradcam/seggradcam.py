import numpy as np
import cv2
from keras import backend as K
from skimage import measure
import matplotlib.pyplot as plt
from operator import sub
import math

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

def encontrar_contorno(matriz):
    # Encuentra las coordenadas (índices) de los 1s en la matriz
    coordenadas_1 = [(i, j) for i, fila in enumerate(matriz) for j, valor in enumerate(fila) if valor == 1]

    if not coordenadas_1:
        # Si no hay ningún 1, devuelve None
        return None

    # Obtiene las coordenadas mínimas y máximas para formar el contorno del cuadrado
    min_fila, min_columna = min(coordenadas_1, key=lambda x: x[0])[0], min(coordenadas_1, key=lambda x: x[1])[1]
    max_fila, max_columna = max(coordenadas_1, key=lambda x: x[0])[0], max(coordenadas_1, key=lambda x: x[1])[1]

    # Calcula el ancho y alto del cuadrado
    ancho = max_columna - min_columna + 1
    alto = max_fila - min_fila + 1

    # Devuelve las variables necesarias para patches.Rectangle
    return min_columna - 0.5, min_fila - 0.5, ancho, alto

def encontrar_contorno_circulo(matriz):
    # Encuentra las coordenadas (índices) de los 1s en la matriz
    coordenadas_1 = [(i, j) for i, fila in enumerate(matriz) for j, valor in enumerate(fila) if valor == 1]

    if not coordenadas_1:
        # Si no hay ningún 1, devuelve None
        return None

    # Obtiene las coordenadas mínimas y máximas para formar el contorno del círculo
    min_fila, min_columna = min(coordenadas_1, key=lambda x: x[0])[0], min(coordenadas_1, key=lambda x: x[1])[1]
    max_fila, max_columna = max(coordenadas_1, key=lambda x: x[0])[0], max(coordenadas_1, key=lambda x: x[1])[1]

    # Calcula el centro y el radio del círculo
    centro_x = (min_columna + max_columna) / 2
    centro_y = (min_fila + max_fila) / 2
    radio = max(max_columna - min_columna, max_fila - min_fila) / 2

    # Devuelve las variables necesarias para patches.Circle
    return centro_x, centro_y, radio

def make_grayscale_colormap():
    # Define the colors for the negative, zero, and positive values
    c_neg = 'black'
    c_pos = 'white'

    # Create a list of tuples containing the position and color for each segment
    colors = [(0, c_neg), (0.5, 'gray'), (1, c_pos)]

    # Create the colormap using LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('grayscale_cmap', colors)

    return cmap

class SuperRoI:  # or rename it to ClassRoI
    def __init__(self, image =None):
        self.image = image
        self.roi = 1
        self.fullroi = None
        self.i = None
        self.j = None

    def setRoIij(self):
        #print("Shape of RoI: ", self.roi.shape)
        self.i = np.where(self.roi == 1)[0]
        self.j = np.where(self.roi == 1)[1]
        #print("Lengths of i and j index lists:", len(self.i), len(self.j))

    def meshgrid(self):
        # mesh for contour
        ylist = np.linspace(0, self.image.shape[0], self.image.shape[0])
        xlist = np.linspace(0, self.image.shape[1], self.image.shape[1])
        return np.meshgrid(xlist, ylist) #returns X,Y


class ClassRoI(SuperRoI):
    def __init__(self, model, image, cls):
        preds = model.predict(np.expand_dims(image, 0))[0]
        max_preds = preds.argmax(axis=-1)
        self.image = image
        #self.roi = np.round(preds[..., cls] * (max_preds == cls)).reshape(image.shape[-3], image.shape[-2])
        self.roi = max_preds == cls
        self.fullroi = self.roi
        self.setRoIij()

    def connectedComponents(self):
        all_labels = measure.label(self.fullroi, background=0)
        (values, counts) = np.unique(all_labels * (all_labels != 0), return_counts=True)
        print("connectedComponents values, counts: ", values, counts)
        return all_labels, values, counts

    def largestComponent(self):
        all_labels, values, counts = self.connectedComponents()
        # find the largest component
        ind = np.argmax(counts[values != 0]) + 1  # +1 because indexing starts from 0 for the background
        print("argmax: ", ind)
        # define RoI
        self.roi = (all_labels == ind).astype(int)
        self.setRoIij()

    def smallestComponent(self):
        all_labels, values, counts = self.connectedComponents()
        ind = np.argmin(counts[values != 0]) + 1
        print("argmin: ", ind)  #
        self.roi = (all_labels == ind).astype(int)
        self.setRoIij()


class PixelRoI(SuperRoI):
    def __init__(self, i, j, image):
        self.image = image
        self.roi = np.zeros((image.shape[-3], image.shape[-2]))
        self.roi[i, j] = 1
        self.i = i
        self.j = j


class BiasRoI(SuperRoI):
    def __init__(self, next_batch, image_id):
        self.id = image_id
        self.image = next_batch[0][image_id][..., 0]
        self.gt_mask = next_batch[1][image_id]  # shape: (64,64,11)
        # self.tile_dict = next_batch[2][image_id]#[...,0]
        self.biased_tile = next_batch[2][image_id]['biased_tile'][..., 0]
        self.is_biased = next_batch[2][image_id]['is_biased']  # True or False
        self.background = next_batch[2][image_id]['background'][..., 0]
        self.digit_with_infill = next_batch[2][image_id]['digit_with_infill'][..., 0]

        self.biased_mask = self.biased_tile * self.background

    def biasedMask(self):
        plt.title('Biased mask for image ' + str(self.id))
        plt.imshow(self.biased_mask)
        plt.colorbar()
        return self.biased_mask
        # save?

    def unbiasedMask(self):

        c = sub(self.background, self.biased_tile)
        print(c.shape)
        c = np.ones(c.shape) * [c > 0]  # np.max(c,0)
        B = c[0]
        plt.title('Unbiased mask for image ' + str(self.id))
        plt.imshow(B)
        plt.colorbar()
        return B

    def biasedTextureContour(self):
        # TODO: draw the contour around the image border where the biased mask is

        # mesh for contour
        X, Y = self.meshgrid()
        plt.figure()
        plt.imshow(self.image, cmap='gray')
        plt.contour(X, Y, self.biased_mask)  # colors=c)

        plt.title('Contour for the biased mask')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


class SegGradCAM:
    """Seg-Grad-CAM method for explanations of predicted segmentation masks.
    Seg-Grad-CAM is applied locally to produce heatmaps showing the relevance of a set of pixels
    or an individual pixel for semantic segmentation.
    """

    def __init__(self, input_model, image, cls=-1, prop_to_layer='activation_9', prop_from_layer='last', image_id = None,
                 roi=SuperRoI(),  # 1, #default: explain all the pixels that belong to cls
                 normalize=True, abs_w=False, posit_w=False, erf_output_irt_input=False, erf_conv_irt_input=False):

        self.input_model = input_model
        self.image = image
        #if cls == None:
        # TODO: add option cls=-1 (predicted class) and cls=None (gt class)
        # TODO print model's confidence (probability) in prediction
        self.cls = cls  # class
        # prop_from_layer is the layer with logits prior to the last activation function
        if prop_from_layer == 'last':
            self.prop_from_layer = self.input_model.layers[-1].name
        else:
            self.prop_from_layer = prop_from_layer
        self.prop_to_layer = prop_to_layer  # an intermediate layer, typically of the bottleneck layers
        self.image_id = image_id

        self.roi = roi  # M, a set of pixel indices of interest in the output mask.
        self.normalize = normalize  # [True, False] normalize the saliency map L_c
        self.abs_w = abs_w  # if True, absolute function is applied to alpha_c
        self.posit_w = posit_w  # if True, ReLU is applied to alpha_c

        self.erf_output_irt_input = erf_output_irt_input #Cálculo del erf de la salida con respecto de la entrada
        self.erf_conv_irt_input = erf_conv_irt_input #Cálculo del erf de una convolución respecto de la entrada

        self.alpha_c = None  # alpha_c, weights for importance of feature maps
        self.A = None  # A, feature maps from the intermediate prop_to_layer
        self.grads_val = None  # gradients of the logits y with respect to all pixels of each feature map 𝐴^𝑘
        self.cam = None  # activation map L_c

        self.cam_max = None

    def featureMapsGradients(self):

        """ This method corresponds to the formula:
        Sum [(d Sum y^c_ij) / (d A^k_uv)] , where
        y^c_ij are logits for every pixel 𝑥_𝑖𝑗 and class c. Pixels x_ij are defined by the region of interest M.
        A^k is a feature map number k. u,v - indexes of pixels of 𝐴^𝑘.

        Return: A, gradients of the logits y with respect to all pixels of each feature map 𝐴^𝑘
        """
        preprocessed_input = np.expand_dims(self.image, 0)
        y_c = self.input_model.get_layer(self.prop_from_layer).output[
                  ..., self.cls] * self.roi.roi  # Mask the region of interest
        #print("y_c: ", type(y_c), np.array(y_c))
        conv_output = self.input_model.get_layer(self.prop_to_layer).output
        #print("conv_output: ", type(conv_output), np.array(conv_output))
        input_output = self.input_model.get_layer("input_1").output

        #Se pueden calcular tres gradientes distintos:
        #1.- Gradiente de la salida con respecto a una capa de convolución concreta (grads_irt_conv_output): esto es lo que se usa en el SegGradCAM
        #2.- Gradiente de la salida con respecto de la entrada (grads_irt_input_output): esto es por definición el effective receptive field.
        #3.- Gradiente de una capa de convolución con respecto a la entrada (): esto nos permite ver cómo evoluciona el erf a lo largo de la red neuronal.
        #IMPORTANTE: Para los casos 2 y 3 lo recomendable es fijarse solo en un píxel de la salida y enmascarar el resto. Como estos cálculos no tienen nada
        #que ver con el del SegGradCAM sería mejor moverlos a otra parte.


        # Normalize if necessary
        # grads = normalize(grads)
        # Gradiente 1: salida del modelo respecto de capa de convolución concreta.
        grads_irt_conv_output = K.gradients(y_c, conv_output)[0]
        gradient_function_grads = K.function([self.input_model.input], [grads_irt_conv_output])
        gradient_function_conv = K.function([self.input_model.input], [conv_output])
        grads_val = gradient_function_grads([preprocessed_input])
        output = gradient_function_conv([preprocessed_input])

        self.A = output[0][0]
        self.grads_val =  grads_val[0][0]


        if self.erf_output_irt_input:
            # Gradiente 2: salida del modelo respecto de la entrada.
            grads_irt_input_output = K.gradients(y_c, input_output)[0]
            gradient_function_grads_2 = K.function([self.input_model.input], [grads_irt_input_output])
            grads_val_2 = gradient_function_grads_2([preprocessed_input])
            erf2 = np.sum(np.array(grads_val_2)[0, 0, :, :, :], axis=-1)

            fig, ax = plt.subplots(figsize=(10 * 2, 5 * 2))
            plt.imshow(erf2, cmap=make_grayscale_colormap(), alpha = 1)
            plt.colorbar()
            plt.savefig("/workspace/Vitis-AI/tutorials/SegGradCAM/erf" + "_img_" + self.image_id + ".png")

        if self.erf_conv_irt_input:
            # Gradiente 3: salida de una capa de convolución respecto de la entrada.
            # Dependiendo de la capa de convolución en la que me encuentre, el tamaño del feature map será uno u otro por lo
            # que voy a elegir siempre la misma posición en la salida el centro del feature map.
            # Por definición de la U-Net todos los feature maps (salvo, como mucho, los de la base) son de dimensiones pares
            # por lo que no hay un centro estrictamente hablando sino 4 píxeles. Con la siguiente fórmula, hacemos que para
            # la base (impar) se escoja el centro y para los pares el píxel inferior derecho del cuadrado de 4 píxeles centrales.
            # (7, 13) -> mask_x = 3, mask_y = 6 como en Python se empieza en 0 conv_output[3, 6] es el centro.
            # (14, 26) --> mask_x = 7, mask_y = 13. Si el píxel superior izquierdo es el [0, 0] y el inferior derecho es el [13, 25],
            # el [7, 13] es el inferior derecho de los 4 que conforman el centro ([6, 12], [6, 13], [7, 12], [7, 13]).

            mask_x = math.floor(conv_output.shape[1]/2)
            mask_y = math.floor(conv_output.shape[2]/2)

            forma = (1, conv_output.shape[1], conv_output.shape[2], conv_output.shape[3])
            mask = np.zeros(forma)
            mask[:, mask_x, mask_y, :] = 1
            conv_output_mask = conv_output * mask

            grads_conv_output_irt_input_output = K.gradients(conv_output_mask, input_output)[0]
            gradient_function_grads_3 = K.function([self.input_model.input], [grads_conv_output_irt_input_output])
            grads_val_3 = gradient_function_grads_3([preprocessed_input])
            erf3 = np.sum(np.array(grads_val_3)[0, 0, :, :, :], axis=-1)

            #Quiero calcular el effective receptive field que se define como la región de la entrada que condiciona el valor de salida de un píxel.
            #Para ello, lo que hago es convertir todos los valores no nulos del erf3 en 1s. De esa manera tengo una matriz de 0s y 1s y lo que
            #hago es hallar el rectángulo que contenga a esos 1s (evidentemente el que mejor se ajusta).

            # Define the radius of the circle
            binarized_erf3 = erf3.copy()
            binarized_erf3[binarized_erf3 != 0] = 1
            variables_circulo = encontrar_contorno_circulo(binarized_erf3)
            variables_rectangulo = encontrar_contorno(binarized_erf3)

            info = np.zeros((2, 3))
            info[0, 0] = variables_circulo[0]
            info[0, 1] = variables_circulo[1]
            info[0, 2] = variables_circulo[2]

            #Como el rectángulo no cambia de imagen a imagen defino el inner dynamic receptive field como el círculo que incluye los puntos que aportan un 25% del máximo del gradiente.
            #No he puesto la restricción de que el círculo tenga que estar centrado en el cuadrado anterior.

            selective_erf3 = erf3.copy()
            selective_erf3 = abs(erf3)
            selective_erf3 = selective_erf3 / np.max(selective_erf3)
            selective_erf3[selective_erf3 <= 0.25] = 0
            selective_erf3[selective_erf3 > 0.25] = 1
            selective_variables = encontrar_contorno_circulo(abs(selective_erf3))

            info[1, 0] = selective_variables[0] #Si centrado --> variables_circulo[0]
            info[1, 1] = selective_variables[1] #Si centrado --> variables_circulo[1]
            info[1, 2] = selective_variables[2] #Si centrado --> selective_variables[2] + np.ceil(np.sqrt( (variables_circulo[0] - selective_variables[0])**2 + (variables_circulo[1] - selective_variables[1])**2 ))

            circle = patches.Circle((selective_variables[0], selective_variables[1]), selective_variables[2], linewidth=1, edgecolor='blue', facecolor='none')
            circle2 = patches.Circle((selective_variables[0], selective_variables[1]), selective_variables[2], linewidth=1, edgecolor='blue', facecolor='none')
            rectangle = patches.Rectangle((variables_rectangulo[0], variables_rectangulo[1]), variables_rectangulo[2], variables_rectangulo[3], linewidth=1, edgecolor='green', facecolor='none')
            rectangle2 = patches.Rectangle((variables_rectangulo[0], variables_rectangulo[1]), variables_rectangulo[2], variables_rectangulo[3], linewidth=1, edgecolor='green', facecolor='none')

            fig, ax = plt.subplots(figsize=(10 * 2, 5 * 2))
            plt.imshow(erf3 / np.max(abs(erf3)), cmap=make_grayscale_colormap(), alpha = 1)
            plt.colorbar()
            plt.gca().add_patch(circle)
            plt.gca().add_patch(rectangle)
            plt.show()
            plt.clim([-1, 1])
            plt.savefig("/workspace/Vitis-AI/tutorials/SegGradCAM/" + self.prop_to_layer + "_erf_" + str(mask_x).zfill(2) + "_" + str(mask_y).zfill(2) + "_img_" + self.image_id + ".png")
            np.save("/workspace/Vitis-AI/tutorials/SegGradCAM/" + self.prop_to_layer + "_erf_" + str(mask_x).zfill(2) + "_" + str(mask_y).zfill(2) + "_img_" + self.image_id + ".npy", erf3)
            plt.close(fig)

            #En esta segunda me quedo con el valor absoluto por si eso hiciera que la Gaussiana cobrase algo más de valor.
            fig, ax = plt.subplots(figsize=(10 * 2, 5 * 2))
            plt.imshow(abs(erf3) / np.max(abs(erf3)), cmap=make_grayscale_colormap(), alpha = 1)
            plt.colorbar()
            plt.gca().add_patch(circle2)
            plt.gca().add_patch(rectangle2)
            plt.show()
            plt.clim([0, 1])
            plt.savefig("/workspace/Vitis-AI/tutorials/SegGradCAM/" + self.prop_to_layer + "_abs_erf_" + str(mask_x).zfill(2) + "_" + str(mask_y).zfill(2) + "_img_" + self.image_id + ".png")
            np.save("/workspace/Vitis-AI/tutorials/SegGradCAM/" + self.prop_to_layer + "_abs_erf_" + str(mask_x).zfill(2) + "_" + str(mask_y).zfill(2) + "_img_" + self.image_id + ".npy", abs(erf3))

            # #Analizando las imágenes para el encoder de la U-Net (a menor profundidad de la capa más evidente es el efecto) se ve que el círculo se ajusta bastante bien a las zonas que abarcan el
            #mayor cambio (no es ni de lejos una Gaussiana pero las zonas centrales tienden a aportar más). Sin embargo, a medida que profundizamos en la red y, sobre todo, en la zona del decoder
            #el radio deja de ajustarse a la zona de mayor cambio. Por tanto, hemos hallado otro radio para un erf selectivo que se define como la zona en la cual está incluida la aportación
            #de 1/4 del máximo del valor absoluto (pudiera darse el caso de que algún píxel con mayor aportación se encontrase fuera de ese círculo).


            np.save("/workspace/Vitis-AI/tutorials/SegGradCAM/" + self.prop_to_layer + "_erf_info" + "_img_" + self.image_id + ".npy", info)

        return self.A, self.grads_val

    def gradientWeights(self):
        """Defines a matrix of alpha(i,j,k)_c. Each alpha(i,j,k)_c denotes importance of a pixel of a certain
        feature map A(i,j,k) for class c. The possibility of applying abs() (abs_w=True) or
        ReLU() (posit_w=True) to the values of the matrix is now covered in activationMap()function."""
        self.alpha_c = self.grads_val

        #Por si quiero anotar estadísticas sobre cuántos píxeles contribuyen positivamente, cuántos negativamente
        #y cuántos no contribuyen.

        #self.alpha_neg = self.alpha_c[:, :, :] < 0
        #print("Número de píxeles que contribuyen negativamente:", np.sum(self.alpha_neg))
        #self.alpha_pos = self.alpha_c[:, :, :] > 0
        #print("Número de píxeles que contribuyen positivamente:", np.sum(self.alpha_pos))
        #self.alpha_zero = self.alpha_c[:, :, :] == 0
        #print("Número de píxeles que no contribuyen:", np.sum(self.alpha_zero))

        return self.alpha_c

    def activationMap(self):
        """The last step to get the activation map. Should be called after outputGradients and gradientWeights.
        If abs_w=True, absolute values of the matrix are processed and returned as weights.
        If posit_w=True, ReLU is applied to the matrix."""
        # weighted sum of feature maps: sum of alpha(i,j,k)_c * A(i,j,k)

        print("\nINFO:")

        if self.abs_w:
            print("Has cambiado el signo de las derivadas negativas")
            self.alpha_c = abs(self.alpha_c)
        if self.posit_w:
            print("Solo incluyes derivadas no negativas")
            self.alpha_c = np.maximum(self.alpha_c, 0)

        cam = np.multiply(self.A, self.alpha_c)
        cam = np.sum(cam, axis=-1)

        #np.multiply(self.A, self.alpha_c) / np.sum(np.multiply(self.A, self.alpha_c), axis=-1)

        img_dim = self.image.shape[:2]
        #Este es un paso muy importante y bastante controvertido. Consiste en determinar cómo
        #pasar de la resolución que tiene el cam a la resolución que tiene la imagen de entrada.
        #Por defecto, se hace uso de una interpolación bilineal espacial aunque sabemos que eso
        #no es estrictamente correcto, ya que el receptive field (efectivo o no) no coincide con
        #la disminución del tamaño a consecuencia de las operaciones de pooling. Algunas personas
        #han propuesto utilizar un upsampling Gaussiano aunque yo tampoco veo que el effective
        #receptive field tenga esa forma siempre.
        cam = cv2.resize(cam, img_dim[::-1], cv2.INTER_LINEAR)

        #Esto es una manera de mejorar la visualización de los datos sin alterar artificialmente el
        #resultados de los mismos. Me deshago de los valores más pequeños y más grandes (outliers)
        #para que el contraste sea mayor.
        cam = np.minimum(cam, np.percentile(cam, 99.9))
        cam = np.maximum(cam, np.percentile(cam, 0.1))

        # normalize non-negative weighted sum
        self.cam_max = np.max(np.abs(cam))
        if self.cam_max != 0 and self.normalize:
            print("Has dividido el CAM por el máximo del valor absoluto. No has realizado una normalización sino un cambio de escala. El 0 anterior sigue siendo un 0 ahora.")
            cam = cam / self.cam_max
        self.cam = cam

        return self.cam

    def SGC(self):
        """Get the activation map"""
        _, _ = self.featureMapsGradients()
        _ = self.gradientWeights()

        return self.activationMap()

    def __sub__(self, otherSGC):
        """Subtraction experiment"""
        pass

    def average(self, otherSGCs):
        """average several seg-grad-cams"""
        new_sgc = self.copy()
        cam = self.SGC()
        cams = [cam]
        if otherSGCs is list:
            for other in otherSGCs:
                cams.append(other.SGC())
        else:
            cams.append(otherSGCs)

        aver = None
        for cc in cams:
            aver += cc
            print("aver shape: ", aver.shape)

        new_sgc.cam = aver/len(cams)
        return new_sgc

    def sortbyMax(self):
        """sort a list of seg-grad-cams by their maximum in activation map before normalization
        for f in sorted(listofSGCs, key = lambda x: x.sortbyMax()):
        print(f.image, f.cls, f.prop_to_layer, f.roi, f.cam_max)
        """
        return self.cam_max
