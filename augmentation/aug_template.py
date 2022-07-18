from matplotlib import pyplot as plt
import numpy as np
import re
from PIL import ImageOps, ImageEnhance, ImageFilter, Image, ImageDraw
import random
from dataclasses import dataclass
from typing import Union

import torch
import torchvision.transforms.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


@dataclass
class MinMax:
    min: Union[float, int]
    max: Union[float, int]


@dataclass
class MinMaxVals:
    shear: MinMax = MinMax(.0, .3)
    translate: MinMax = MinMax(0, 10)  # different from uniaug: MinMax(0,14.4)
    rotate: MinMax = MinMax(0, 30)
    solarize: MinMax = MinMax(0, 256)
    posterize: MinMax = MinMax(0, 4)  # different from uniaug: MinMax(4,8)
    enhancer: MinMax = MinMax(.1, 1.9)
    cutout: MinMax = MinMax(.0, .2)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    return float(level) * maxval / PARAMETER_MAX

# 将maxval缩放到原来的level/PARAMETER_MAX
def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
    return int(level * maxval / PARAMETER_MAX)


class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, pil_img):
        return self.f(pil_img)


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def __repr__(self):
        return '<' + self.name + '>'

    def pil_transformer(self, probability, level):
        def return_function(im):
            if random.random() < probability:
                im = self.xform(im, level)
            return im

        name = self.name + '({:.1f},{})'.format(probability, level)
        print('pil_transformer_name')
        print(name)
        return TransformFunction(return_function, name)


################## Transform Functions ##################
identity = TransformT('identity', lambda pil_img, level: pil_img)

flip_lr = TransformT(
    'FlipLR',
    lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))
flip_ud = TransformT(
    'FlipUD',
    lambda pil_img, level: pil_img.transpose(Image.FLIP_TOP_BOTTOM))

auto_contrast = TransformT(
    'AutoContrast',
    lambda pil_img, level: ImageOps.autocontrast(
        pil_img))

equalize = TransformT(
    'Equalize',
    lambda pil_img, level: ImageOps.equalize(
        pil_img))

def _rotate_impl(pil_img, level):
    """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
    degrees = int_parameter(level, min_max_vals.rotate.max)
    if random.random() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees)


rotate = TransformT('Rotate', _rotate_impl)

def _posterize_impl(pil_img, level):
    """Applies PIL Posterize to `pil_img`."""
    level = int_parameter(level, min_max_vals.posterize.max - min_max_vals.posterize.min)
    return ImageOps.posterize(pil_img, min_max_vals.posterize.max - level)


posterize = TransformT('Posterize', _posterize_impl)

def _shear_x_impl(pil_img, level):
    """Applies PIL ShearX to `pil_img`.
  The ShearX operation shears the image along the horizontal axis with `level`
  magnitude.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had ShearX applied to it.
  """
    level = float_parameter(level, min_max_vals.shear.max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))


shear_x = TransformT('ShearX', _shear_x_impl)

def _shear_y_impl(pil_img, level):
    """Applies PIL ShearY to `pil_img`.
  The ShearY operation shears the image along the vertical axis with `level`
  magnitude.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had ShearX applied to it.
  """
    level = float_parameter(level, min_max_vals.shear.max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))


shear_y = TransformT('ShearY', _shear_y_impl)

def _translate_x_impl(pil_img, level):
    """Applies PIL TranslateX to `pil_img`.
  Translate the image in the horizontal direction by `level`
  number of pixels.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had TranslateX applied to it.
  """
    level = int_parameter(level, min_max_vals.translate.max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))


translate_x = TransformT('TranslateX', _translate_x_impl)

def _translate_y_impl(pil_img, level):
    """Applies PIL TranslateY to `pil_img`.
  Translate the image in the vertical direction by `level`
  number of pixels.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had TranslateY applied to it.
  """
    level = int_parameter(level, min_max_vals.translate.max)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))


translate_y = TransformT('TranslateY', _translate_y_impl)


def _crop_impl(pil_img, level, interpolation=Image.BILINEAR):
    """Applies a crop to `pil_img` with the size depending on the `level`."""
    level = int_parameter(level, 10)
    w = pil_img.width
    h = pil_img.height
    cropped = pil_img.crop((level, level, w - level, h - level))
    resized = cropped.resize((w, h), interpolation)
    return resized


crop_bilinear = TransformT('CropBilinear', _crop_impl)

def _solarize_impl(pil_img, level):
    """Applies PIL Solarize to `pil_img`.
  Translate the image in the vertical direction by `level`
  number of pixels.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had Solarize applied to it.
  """
    level = int_parameter(level, min_max_vals.solarize.max)
    return ImageOps.solarize(pil_img, 256 - level)


solarize = TransformT('Solarize', _solarize_impl)


def _enhancer_impl(enhancer, minimum=None, maximum=None):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""

    def impl(pil_img, level):
        mini = min_max_vals.enhancer.min if minimum is None else minimum
        maxi = min_max_vals.enhancer.max if maximum is None else maximum
        v = float_parameter(level, maxi - mini) + mini  # going to 0 just destroys it
        return enhancer(pil_img).enhance(v)

    return impl


contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
brightness = TransformT('Brightness', _enhancer_impl(
    ImageEnhance.Brightness))
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))


def set_augmentation_space(num_strengths, custom_augmentation_space_augs=None):
    global ALL_TRANSFORMS, min_max_vals, PARAMETER_MAX
    assert num_strengths > 0
    PARAMETER_MAX = num_strengths - 1
    
    min_max_vals = MinMaxVals(
        posterize=MinMax(4, 8)
    )
    assert custom_augmentation_space_augs is not None

    custom_augmentation_space_augs_mapping = {
        'identity': identity,
        'auto_contrast': auto_contrast,
        'equalize': equalize,
        'rotate': rotate,
        'solarize': solarize,
        'posterize': posterize,
        'contrast': contrast,
        'brightness': brightness,
        'sharpness': sharpness,
        'shear_x': shear_x,
        'shear_y': shear_y,
        'translate_x': translate_x,
        'translate_y': translate_y,
        'flip_lr': flip_lr,
        'flip_ud': flip_ud,
        'crop_bilinear': crop_bilinear,
    }
    ALL_TRANSFORMS = []
    ALL_TRANSFORMS += [
        custom_augmentation_space_augs_mapping[aug] for aug in custom_augmentation_space_augs
    ]
    print("CUSTOM Augs set to:", ALL_TRANSFORMS)


class TrivialAugment:
    def __init__(self, num_strengths = 31, custom_augmentation_space_augs = ['identity']):
        set_augmentation_space(num_strengths, custom_augmentation_space_augs)

    def __call__(self, img):
        op = random.choices(ALL_TRANSFORMS, k=1)[0]
        # op = crop_bilinear
        print('op')
        print(op)
        level = random.randint(0, PARAMETER_MAX)
        # level = 0
        print('level')
        print(level)
        img = op.pil_transformer(1., level)(img)
        return img


if __name__ == '__main__':
    print('main')
    img = Image.open(r'C:\Users\t-jiahuihe\OneDrive\learn\pytorch\chapter_computer-vision\dog.jpg')
    augs = ['identity', 'auto_contrast', 'equalize', 'rotate', 'solarize',
             'posterize','contrast', 'brightness', 'sharpness', 'shear_x', 'shear_y',
             'translate_x', 'translate_y', 'flip_lr', 'flip_ud', 'crop_bilinear']
    augmenter = TrivialAugment(31, augs)
    aug_img = augmenter(img)
    print(aug_img.size)# 原来(600, 744)(宽，长)
    print(F.to_tensor(img).size())# F.to_tensor(img).size() torch.Size([3, 744, 600])
    print(F.to_tensor(aug_img).size())
    plt.figure('image')
    plt.imshow(aug_img)
    plt.show()
    # plt.imshow(img)
    # plt.show()
    