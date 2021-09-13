import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from skimage.transform import resize
from PIL import Image
import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras

class Visualization:
    def __init__(self, dataframe):
        """
        dataframe = type:pd.DataFrame, full dataframe that contains information.
        """
        self.dataframe = dataframe

    def counts(self, name=str):
        return self.dataframe[name].value_counts()

    def cutName(self, data, threshold=int):
        self.data = data
        dataframe = pd.DataFrame(data).transpose()
        others = 0
        for i in dataframe.columns.tolist():
            if int(dataframe[i]) < threshold:
                others += int(dataframe[i])
                dataframe = dataframe.drop(i, axis=1)
        dataframe['기타'] = others
        return dataframe.transpose()

    def pieplot(self, dataframe):
        self.dataframe = dataframe;
        explode = [0.10] * len(dataframe.index.tolist());

        plt.figure(figsize=(15, 15));
        plt.pie(dataframe, labels=dataframe.index.tolist(), autopct='%.1f%%', startangle=260, counterclock=False,
                explode=explode);
        plt.show();

    def plot(self, name=str, threshold=int):
        self.name = name
        self.threshold = threshold

        colDataFrame = self.counts(name)
        cuttedDataFrame = self.cutName(colDataFrame, threshold)
        self.pieplot(cuttedDataFrame)


class Heatmap:
    def __init__(self, model, pre_trained, last_conv_layer_name, classifier_layer_names, model_classes,
                 input_shape=(240, 240, 3)):
        """
        model = model layers
        pre_trained = pre-trained model. *.h5
        last_conv_layer_name = last convolution layer name
        classifier_layer_name = last layers name that classify image
        model_classes: model classes from dataset
        """
        self.model = model
        self.pre_trained = pre_trained
        self.last_conv_layer_name = last_conv_layer_name
        self.classifier_layer_name = classifier_layer_names
        self.model_classes = model_classes
        self.input_shape = input_shape

    def make_gradcam_heatmap(self, img_array, model, pre_trained, last_conv_layer_name, classifier_layer_names):

        # First, we create a model that maps the input image to the activations
        # of the last conv layer
        last_conv_layer = model.get_layer(pre_trained).get_layer(last_conv_layer_name)
        conv_model = keras.Model(model.get_layer(pre_trained).inputs, last_conv_layer.output)
        # Second, we create a model that maps the activations of the last conv
        # layer to the final class predictions
        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input

        for layer_name in classifier_layer_names:
            x = model.get_layer(layer_name)(x)

        classifier_model = keras.Model(classifier_input, x)
        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            last_conv_layer_output = conv_model(img_array)
            tape.watch(last_conv_layer_output)

            # Compute class predictions
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output.numpy()[0]

        # is our saliency heatmap of class activation
        saliency = np.mean(last_conv_layer_output, axis=-1)
        saliency = np.maximum(saliency, 0) / np.max(saliency)

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        pooled_grads = pooled_grads.numpy()

        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # The channel-wise mean of the resulting feature map
        # is our grad_cam heatmap of class activation
        grad_cam = np.mean(last_conv_layer_output, axis=-1)
        grad_cam = np.maximum(grad_cam, 0) / np.max(grad_cam)

        return grad_cam, saliency

    def merge_with_heatmap(self, original_img, heatmap):
        original_img = np.array(original_img)
        resized_heatmap = resize(heatmap, (240, 240))
        resized_heatmap = np.uint8(255 * resized_heatmap)
        resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
        resized_heatmap = cv2.cvtColor(resized_heatmap, cv2.COLOR_RGB2BGR)
        return cv2.addWeighted(resized_heatmap, 0.7, original_img, 0.5, 6)

    def show_hotmap(self, img, heatmap, title='Heatmap', alpha=0.6, cmap='jet', axisOnOff='off'):
        '''
        img     :    Image
        heatmap :    2d narray
        '''
        resized_heatmap = resize(heatmap, img.size)

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.imshow(resized_heatmap, alpha=alpha, cmap=cmap)
        plt.axis(axisOnOff)
        plt.title(title)
        plt.show()

    def prepare_single_input(self, img_path, target_size=(224, 224)):
        img = image.load_img(img_path, target_size=target_size)
        img = image.img_to_array(img)
        img /= 255.
        img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
        return img

    def predict_image(self, Mymodel, img_path, top_k_num=1):
        image = self.prepare_single_input(img_path, target_size=(240, 240))
        result = Mymodel.predict([image])[0]

        result = list(result)

        classname_list = []
        pred_value_list = []
        for _ in range(top_k_num):
            index = result.index(max(result))
            classname_list.append(self.model_classes[index])
            pred_value_list.append(max(result))
            result[index] = 0.

        return classname_list, pred_value_list

    def show(self, img_path):
        img = Image.open(img_path).resize(size=self.input_shape[:2])
        img_array = self.prepare_single_input(img_path, target_size=(240, 240))

        grad_cam, saliency = self.make_gradcam_heatmap(img_array, self.model, self.pre_train, self.last_conv_layer_name,
                                                  self.classifier_layer_names)

        grad_cam_merge = self.merge_with_heatmap(img, grad_cam)
        saliency_merge = self.merge_with_heatmap(img, saliency)

        plt.subplot(221)
        plt.imshow(grad_cam, 'jet')
        plt.title('GradCam')
        plt.subplot(222)
        plt.imshow(saliency, 'jet')
        plt.title('Saliencia')
        plt.subplot(223)
        plt.imshow(grad_cam_merge)
        plt.title('GradCam-Merge')
        plt.subplot(224)
        plt.imshow(saliency_merge)
        plt.title('Saliencia-Merge')
        plt.show()
