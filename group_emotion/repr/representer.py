#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def display_image(image, title='', figsize=None):
    '''
    Displays a cv image.

    Parameters
    ----------
    image : numpy.ndarray
        image in a numpy array.
    title : string, optional
        title of the image. The default is ''.

    Returns
    -------
    None.

    '''
    if figsize != '':
        plt.figure(figsize=figsize, dpi= 100, facecolor='w', edgecolor='k')
    image = image.astype('float32') * 255
    if image.ndim == 3:    # color image
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGRA2RGB).astype('uint8'), cmap='cubehelix')        
    elif image.ndim == 2:    # grayscale image
        plt.imshow(image, cmap='gray')
    # plt.colorbar()
    plt.title(title)
    plt.show()



class emotion_representer:

    def __init__(self, canvas_width, canvas_height):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.initialize_happiness_images()
        # self.initialize_engagement_images()
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.org = (50, 50)
        self.font_scale = 2
        self.color = (0, 255, 0)
        self.thickness = 5
        self.imgtext = 'text'

    def initialize_happiness_images(self):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        happy_image_file = os.path.join(package_directory, 'images/happy.jpg')
        sad_image_file = os.path.join(package_directory,'images/sad.jpg')
        self.happy_image = cv.imread(happy_image_file)
        self.sad_image = cv.imread(sad_image_file)

    def initialize_engagement_images(self):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        engaged_image_file = os.path.join(package_directory, 'images/engaged.jpg')
        disengaged_image_file = os.path.join(package_directory, 'images/disengaged.jpg')
        self.engaged_image = cv.imread(engaged_image_file)
        self.disengaged_image = cv.imread(disengaged_image_file)

    """
        Reduces the size(width and height) of the given image by 
        the given percentage.
    """
    def reduce_size_by_percentage(self, image, percentage):
        width = int((self.canvas_width * percentage) / 100.0)
        height = int((self.canvas_height * percentage) / 100.0)
        resized_image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
        return resized_image

    """
        Concatenates two images horizontally
    """
    def concat_images(self, imga, imgb):
        ha, wa = imga.shape[:2]
        hb, wb = imgb.shape[:2]
        new_img = np.zeros(shape=(np.max([ha, hb]), wa + wb, 3), dtype=np.uint8)
        new_img[new_img == 0] = 255
        new_img[:ha, :wa] = imga
        new_img[:hb, wa:wa + wb] = imgb
        return new_img

    """
        concatenates n images from a list of images.
    """
    def concat_n_images(self, images):
        output = None
        for i, img in enumerate(images):
            if i == 0:
                output = img
            else:
                output = self.concat_images(output, img)
        return output

    """
        Calculates the unknown variable that is multiplied by the two values 
        so that their sum adds up to 100 to keep the ratio of the
        two values. 
    """
    def calculate_new_ratio(self, prob_a, prob_b):
        # (prob_a * x + prob_b * x) = 100
        diff = 100 / (prob_a + prob_b)
        value1 = prob_a * diff
        value2 = prob_b * diff
        return (value1, value2)

    """
        Returns the resized happiness images based on the emotion percentage 
    """
    def get_resized_happiness_images(self, prob_a, prob_b):
        ratio_a, ratio_b = self.calculate_new_ratio(prob_a, prob_b)
        img1 = self.reduce_size_by_percentage(self.happy_image, ratio_a)
        img2 = self.reduce_size_by_percentage(self.sad_image, ratio_b)
        return [img1, img2]

    """
        Returns the resized engagedness images based on the emotion percentage 
    """
    def get_resized_engagedness_images(self, prob_a, prob_b):
        ratio_a, ratio_b = self.calculate_new_ratio(prob_a, prob_b)
        img1 = self.reduce_size_by_percentage(self.engaged_image, ratio_a)
        img2 = self.reduce_size_by_percentage(self.disengaged_image, ratio_b)
        return [img1, img2]

    """
        Changes the Opacity/Alpha Value of an given image
    """
    def getRGBA(self, opacity, src):
        img = np.array(src, dtype=float)
        img /= 255.0
        a_channel = np.ones(img.shape, dtype=float) - (opacity / 100)
        return img * a_channel

    def produce_happiness_repr(self, happy_prob, sad_prob, unknown_prob):
        size_modified = self.concat_n_images(self.get_resized_happiness_images(happy_prob, sad_prob))
        opacity_modified = self.getRGBA(unknown_prob, size_modified)
        return opacity_modified

    def produce_engagedness_repr(self, engaged_prob, disengaged_prob, unknown_prob):
        size_modified = self.concat_n_images(self.get_resized_engagedness_images(engaged_prob, disengaged_prob))
        opacity_modified = self.getRGBA(unknown_prob, size_modified)
        return opacity_modified
































