#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division

import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass


@dataclass
class emoji_collection:
    """
    A class to hold emoji images.

    Attributes
    ----------
    positive_image : numpy.ndarray
        emoji image of positive emotion.
    negative_image : numpy.ndarray
        emoji image of negative emotion.
    """
    positive_image: np.ndarray
    negative_image: np.ndarray


def generate_emoji_collection(file_pos, file_neg):
    """
    Creates an instance of emoji_collection.

    Parameters
    ----------
    file_pos: str
        filename of positive image, relative to 'images' folder
    file_neg: str
        filename of negative image, relative to 'images' folder

    Returns
    -------
    emoji_collection
        An instance of emoji_collection.
    """
    package_directory = os.path.dirname(os.path.abspath(__file__))
    pos_file_full_path = os.path.join(package_directory, '..', '..', 'images', file_pos)
    neg_file_full_path = os.path.join(package_directory, '..', '..', 'images', file_neg)

    pos_img = cv.imread(pos_file_full_path)
    neg_img = cv.imread(neg_file_full_path)
    if pos_img is None:
        raise ValueError(f'{file_pos} is an invalid filename.')
    elif neg_img is None:
        raise ValueError(f'{file_neg} is an invalid filename.')

    emoji_col = emoji_collection(pos_img, neg_img)
    return emoji_col


def display_image(image, title='', figsize=None):
    """
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

    """
    if figsize != '':
        plt.figure(figsize=figsize, dpi=100, facecolor='w', edgecolor='k')
    image = image.astype('float32') * 255
    if image.ndim == 3:  # color image
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGRA2RGB).astype('uint8'), cmap='cubehelix')
    elif image.ndim == 2:  # grayscale image
        plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()


class emotion_representer:
    def __init__(self, canvas_width, canvas_height, pos_img_filename='happy.jpg', neg_image_filename='sad.jpg'):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.emoji_col = generate_emoji_collection(pos_img_filename, neg_image_filename)

    def reduce_size_by_percentage(self, image, percentage):
        """
        Reduces the size(width and height) of the given image by
        the given percentage.

        Parameters
        ----------
        image : numpy.ndarray
            input image.
        percentage : float
            percentage of the returned image w.r.to. input image.
        Returns
        -------
        numpy.ndarray
            reduced image.
        """
        eps = 0.00001
        width = int((self.canvas_width * percentage) / 100.0)
        height = int((self.canvas_height * percentage) / 100.0)
        resized_image = None
        if abs(width - 0.0) < eps or abs(height - 0.0) < eps:
            resized_image = np.zeros((0, 0, 3))
        else:
            resized_image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
        return resized_image

    def concat_images(self, imga, imgb):
        """
        Concatenates two images horizontally.

        Parameters
        ----------
        imga : numpy.ndarray
            first input image.
        imgb : numpy.ndarray
            second input image.

        Returns
        -------
        numpy.ndarray
            horizontally concatenated image.
        """
        ha, wa = imga.shape[:2]
        hb, wb = imgb.shape[:2]
        # new_img = np.zeros(shape=(np.max([ha, hb]), wa + wb, 3), dtype=np.uint8)
        new_img = np.zeros(shape=(ha + hb, wa + wb, 3), dtype=np.uint8)
        new_img[new_img == 0] = 255
        new_img[:ha, :wa] = imga
        new_img[:hb, wa:wa + wb] = imgb
        return new_img

    def concat_n_images(self, images):
        """
        Concatenates n images from a list of images.

        Parameters
        ----------

        Returns
        -------

        """
        output = None
        for i, img in enumerate(images):
            if i == 0:
                output = img
            else:
                output = self.concat_images(output, img)
        return output

    def calculate_new_ratio(self, prob_a, prob_b):
        """
        Calculates the unknown variable that is multiplied by the two values
        so that their sum adds up to 100 to keep the ratio of the
        two values.

        Parameters
        ----------
        prob_a : float
            probability of positive emotion detection.
        prob_b : float
            probability of positive emotion detection.
        Returns
        -------
        tuple of float
            in the returned tuple summation of first value and sencond value is 100.
        """
        # (prob_a * x + prob_b * x) = 100
        eps = 0.00001
        diff = 0
        if abs(prob_a - 0.0) < eps and abs(prob_b - 0.0) < eps:
            diff = 0
        else:
            diff = 100 / (prob_a + prob_b)
        value1 = prob_a * diff
        value2 = prob_b * diff
        return (value1, value2)

    def get_resized_images(self, prob_a, prob_b):
        """
        Returns the resized images based on the emotion percentage.

        Parameters
        ----------
        prob_a : float
            probability of positive emotion detection.
        prob_b : float
            probability of positive emotion detection.

        Returns
        -------
        a list of numpy.ndarray
            Resized images in a list. Ratio of positive emotion image size to the second
            one is same as prob_b to prob_b.
        """
        ratio_a, ratio_b = self.calculate_new_ratio(prob_a, prob_b)
        img1 = self.reduce_size_by_percentage(self.emoji_col.positive_image, ratio_a)
        img2 = self.reduce_size_by_percentage(self.emoji_col.negative_image, ratio_b)
        return [img1, img2]

    def getRGBA(self, image, opacity):
        """
        Changes the Opacity/Alpha Value of an given image

        Parameters
        ----------
        image : numpy.ndarray
            input image.
        opacity : float
            desired opacity.
        Returns
        -------
        numpy.ndarray
            opacity adjusted image.
        """
        overlay = image.copy()
        output = image.copy()
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), (255, 255, 255), -1)
        output = cv2.addWeighted(overlay, opacity, output, 1 - opacity, 0)
        return output

    def get_repr(self, pos_emotion_prob, neg_emotion_prob, unknown_emotion_prob):
        """
        Concatenates positive emotion emoji with negative emotion emoji by maintaining
        their size ratio as same as their detection probability. This method also changes
        the opacity of the output image depending on the probability of unknown emotion detection.

        Parameters
        ----------
        pos_emotion_prob : float
            probability of positive emotion detection.
        neg_emotion_prob : float
            probability of negative emotion detection.
        unknown_emotion_prob : float
            probability of unknown emotion detection.

        Returns
        -------
        numpy.ndarray
            result image.
        """
        # check if non-negative probabilities are provided
        assert pos_emotion_prob >= 0
        assert neg_emotion_prob >= 0
        assert unknown_emotion_prob >= 0

        # if sum of probabilities are not 1; make their sum to 1
        sum_of_probabilities = pos_emotion_prob + neg_emotion_prob + unknown_emotion_prob
        if abs(sum_of_probabilities - 1) > 0.0001:
            pos_emotion_prob = pos_emotion_prob / sum_of_probabilities
            neg_emotion_prob = neg_emotion_prob / sum_of_probabilities
            unknown_emotion_prob = unknown_emotion_prob / sum_of_probabilities

        size_modified = self.concat_n_images(self.get_resized_images(pos_emotion_prob, neg_emotion_prob))

        # in case the emotion probabilities are [0, 0, 100]
        # print(f'size_modified: {size_modified}, type: {type(size_modified)}, shape: {size_modified.shape}')
        if size_modified.shape == (0, 0, 3):
            size_modified = np.ones((self.canvas_width, self.canvas_height, 3)) * 255

        # size_modified = np.array(size_modified * 255, dtype=np.uint8) # revert back to 255 value
        opacity_modified = self.getRGBA(size_modified, unknown_emotion_prob)

        return opacity_modified / 255
