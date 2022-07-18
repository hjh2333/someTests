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

# mentor建议旋转平移缩放之类的
# 疑问？比如向右平移图片，最右一截原图就没了？最左什么填平呢？超出部分没了，待填充的用0，RGB的0，0，0是黑色
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
# 左右翻转
flip_lr = TransformT(
    'FlipLR',
    lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))
# 上下翻转
flip_ud = TransformT(
    'FlipUD',
    lambda pil_img, level: pil_img.transpose(Image.FLIP_TOP_BOTTOM))
# pylint:disable=g-long-lambda
# 最大图像对比度。这个函数计算一个输入图像的直方图，从这个直方图中去除最亮和最暗的百分之cutoff，然后重新映射图像，以便保留的最暗像素变为黑色，即0，最亮的变为白色，即255。
# https://blog.csdn.net/u010417185/article/details/72604344
auto_contrast = TransformT(
    'AutoContrast',
    lambda pil_img, level: ImageOps.autocontrast(
        pil_img))
# 均衡图像的直方图。该函数使用一个非线性映射到输入图像，为了产生灰色值均匀分布的输出图像。
equalize = TransformT(
    'Equalize',
    lambda pil_img, level: ImageOps.equalize(
        pil_img))
# 黑白反转？
invert = TransformT(
    'Invert',
    lambda pil_img, level: ImageOps.invert(
        pil_img))
# pylint:enable=g-long-lambda
blur = TransformT(
    'Blur', lambda pil_img, level: pil_img.filter(ImageFilter.BLUR))
smooth = TransformT(
    'Smooth',
    lambda pil_img, level: pil_img.filter(ImageFilter.SMOOTH))

# 顺时针旋转多少度。注意expand 参数设置为True将不规则图片铺满,https://blog.csdn.net/tornado_liao/article/details/109508848
def _rotate_impl(pil_img, level):
    """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
    degrees = int_parameter(level, min_max_vals.rotate.max)
    if random.random() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees)


rotate = TransformT('Rotate', _rotate_impl)

# 多色调分色印刷（或显示）将每个颜色通道上变量bits对应的低(8-bits)个bit置0。变量bits的取值范围为[0,8]
# 没懂，图片全1（全1是空白图）变成全0.9961
def _posterize_impl(pil_img, level):
    """Applies PIL Posterize to `pil_img`."""
    level = int_parameter(level, min_max_vals.posterize.max - min_max_vals.posterize.min)
    return ImageOps.posterize(pil_img, min_max_vals.posterize.max - level)


posterize = TransformT('Posterize', _posterize_impl)

# shear剪切变换。
# 横向剪切_shear_x_impl (x,y)->(x+ky,y)
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

# 纵向剪切_shear_y_impl (x,y)->(x,kx+y)
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

# 横向平移_translate_x_impl (x,y)->(x+k,y)
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

# 纵向平移 _translate_y_impl (x,y)->(x,y+k)
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

# 过度曝光ImageOps.solarize(pil_img, threshold)将高于threshold的像素反转（按位取反）.这个对应参数的效果有点显著，可能要调下
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


color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
ohl_color = TransformT('Color', _enhancer_impl(ImageEnhance.Color, .3, .9))
"""
1、对比度contrast:白色画面(最亮时)下的亮度除以黑色画面(最暗时)下的亮度；
2、色彩饱和度Color:彩度除以明度，指色彩的鲜艳程度，也称色彩的纯度；
3、色调?:向负方向调节会显现红色，正方向调节则增加黄色。适合对肤色对象进行微调；
4、锐度Sharpness:是反映图像平面清晰度和图像边缘锐利程度的一个指标,增强因子为1.0是原始图片。试了一下增强因子设5~10才开始肉眼略有变化,不知道这篇论文为啥只考虑0.1~1.9
5、亮度Brightness:增强因子为0.0将产生黑色图像;为1.0将保持原始图像;>1.0变亮
"""
contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
brightness = TransformT('Brightness', _enhancer_impl(
    ImageEnhance.Brightness))
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))
contour = TransformT(
    'Contour', lambda pil_img, level: pil_img.filter(ImageFilter.CONTOUR))
detail = TransformT(
    'Detail', lambda pil_img, level: pil_img.filter(ImageFilter.DETAIL))
edge_enhance = TransformT(
    'EdgeEnhance', lambda pil_img, level: pil_img.filter(ImageFilter.EDGE_ENHANCE))
sharpen = TransformT(
    'Sharpen', (lambda pil_img, level: pil_img.filter(ImageFilter.SHARPEN)))
max_ = TransformT(
    'Max', lambda pil_img, level: pil_img.filter(ImageFilter.MaxFilter))
min_ = TransformT(
    'Min', lambda pil_img, level: pil_img.filter(ImageFilter.MinFilter))
median = TransformT(
    'Median', lambda pil_img, level: pil_img.filter(ImageFilter.MedianFilter))
gaussian = TransformT(
    'Gaussian', lambda pil_img, level: pil_img.filter(ImageFilter.GaussianBlur))



def _mirrored_enhancer_impl(enhancer, minimum=None, maximum=None):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""

    def impl(pil_img, level):
        mini = min_max_vals.enhancer.min if minimum is None else minimum
        maxi = min_max_vals.enhancer.max if maximum is None else maximum
        assert mini == 0., "This enhancer is used with a strength space that is mirrored around one."
        v = float_parameter(level, maxi - mini) + mini  # going to 0 just destroys it
        if random.random() < .5:
            v = -v
        return enhancer(pil_img).enhance(1. + v)

    return impl


mirrored_color = TransformT('Color', _mirrored_enhancer_impl(ImageEnhance.Color))
mirrored_contrast = TransformT('Contrast', _mirrored_enhancer_impl(ImageEnhance.Contrast))
mirrored_brightness = TransformT('Brightness', _mirrored_enhancer_impl(
    ImageEnhance.Brightness))
mirrored_sharpness = TransformT('Sharpness', _mirrored_enhancer_impl(ImageEnhance.Sharpness))


def CutoutDefault(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v <= 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (0, 0, 0)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


cutout = TransformT('Cutout',
                    lambda img, l: CutoutDefault(img, int_parameter(l, img.size[0] * min_max_vals.cutout.max)))




blend_images = None


def blend(img1, v):
    if blend_images is None:
        print("please set google_transformations.blend_images before using the enlarged_randaug search space.")
    i = np.random.choice(len(blend_images))
    img2 = blend_images[i]
    m = float_parameter(v, .4)
    return Image.blend(img1, img2, m)


sample_pairing = TransformT('SamplePairing', blend)


def set_augmentation_space(augmentation_space, num_strengths, custom_augmentation_space_augs=None):
    global ALL_TRANSFORMS, min_max_vals, PARAMETER_MAX
    assert num_strengths > 0
    PARAMETER_MAX = num_strengths - 1
    if 'wide' in augmentation_space:
        min_max_vals = MinMaxVals(
            shear=MinMax(.0, .99),
            translate=MinMax(0, 32),
            rotate=MinMax(0, 135),
            solarize=MinMax(0, 256),
            posterize=MinMax(2, 8),
            enhancer=MinMax(.01, 2.),
            cutout=MinMax(.0, .6),
        )
    elif ('uniaug' in augmentation_space) or ('randaug' in augmentation_space):
        min_max_vals = MinMaxVals(
            posterize=MinMax(4, 8),
            translate=MinMax(0, 14.4)
        )
    elif 'fixmirror' in augmentation_space:
        min_max_vals = MinMaxVals(
            posterize=MinMax(4, 8),
            enhancer=MinMax(0., .9)
        )
    elif 'fiximagenet' in augmentation_space:
        min_max_vals = MinMaxVals(
            posterize=MinMax(4, 8),
            translate=MinMax(0, 70)
        )
    elif 'fix' in augmentation_space:
        min_max_vals = MinMaxVals(
            posterize=MinMax(4, 8)
        )
    elif 'ohl' in augmentation_space:
        assert PARAMETER_MAX == 2
        min_max_vals = MinMaxVals(
            shear=MinMax(.1, .3),
            translate=MinMax(5, 14),
            rotate=MinMax(10, 30),
            solarize=MinMax(26, 179),
            posterize=MinMax(4, 7),
            enhancer=MinMax(1.3, 1.9),
            cutout=MinMax(.0, .6),
        )
    else:
        min_max_vals = MinMaxVals()

    if 'xlong' in augmentation_space:
        ALL_TRANSFORMS = [
            identity,
            auto_contrast,
            equalize,
            rotate,
            solarize,
            color,
            posterize,
            contrast,
            brightness,
            sharpness,
            shear_x,
            shear_y,
            translate_x,
            translate_y,
            blur,
            invert,
            flip_lr,
            flip_ud,
            cutout,
            crop_bilinear,
            contour,
            detail,
            edge_enhance,
            sharpen,
            max_,
            min_,
            median,
            gaussian
        ]
    elif 'rasubsetof' in augmentation_space:
        r = re.findall(r'rasubsetof(\d+)', augmentation_space)
        assert len(r) == 1
        ALL_TRANSFORMS = random.sample(ALL_TRANSFORMS, int(r[0]))
        print(f"Subsampled {len(ALL_TRANSFORMS)} augs: {ALL_TRANSFORMS}")
    elif 'fixmirror' in augmentation_space:
        ALL_TRANSFORMS = [
            identity,
            auto_contrast,
            equalize,
            rotate,
            solarize,
            mirrored_color,  # enhancer
            posterize,
            mirrored_contrast,  # enhancer
            mirrored_brightness,  # enhancer
            mirrored_sharpness,  # enhancer
            shear_x,
            shear_y,
            translate_x,
            translate_y
        ]
    elif 'long' in augmentation_space:
        ALL_TRANSFORMS = [
            identity,
            auto_contrast,
            equalize,
            rotate,
            solarize,
            color,
            posterize,
            contrast,
            brightness,
            sharpness,
            shear_x,
            shear_y,
            translate_x,
            translate_y,
            # sample_pairing,
            blur,
            invert,
            flip_lr,
            flip_ud,
            cutout
        ]
    elif 'uniaug' in augmentation_space:
        ALL_TRANSFORMS = [
            identity,
            shear_x,
            shear_y,
            translate_x,
            translate_y,
            rotate,
            auto_contrast,
            invert,  # only uniaug
            equalize,
            solarize,
            posterize,
            contrast,
            color,
            brightness,
            sharpness,
            cutout  # only uniaug
        ]
    elif 'autoaug_paper' in augmentation_space:
        ALL_TRANSFORMS = [
            shear_x,
            shear_y,
            translate_x,
            translate_y,
            rotate,
            auto_contrast,
            invert,
            equalize,
            solarize,
            posterize,
            contrast,
            color,
            brightness,
            sharpness,
            cutout,
            sample_pairing
        ]
    elif 'full' in augmentation_space:
        ALL_TRANSFORMS = [
            flip_lr,
            flip_ud,
            auto_contrast,
            equalize,
            invert,
            rotate,
            posterize,
            crop_bilinear,
            solarize,
            color,
            contrast,
            brightness,
            sharpness,
            shear_x,
            shear_y,
            translate_x,
            translate_y,
            cutout,
            blur,
            smooth
        ]
    elif 'ohl' in augmentation_space:
        ALL_TRANSFORMS = [
            shear_x,  # ok
            shear_y,  # ok
            translate_x,  # ok
            translate_y,  # ok
            rotate,  # ok
            ohl_color,  # nok
            posterize,  # ok
            solarize,  # ok
            contrast,  # ok
            sharpness,  # ok
            brightness,  # ok
            auto_contrast,
            equalize,
            invert
        ]
    elif 'custom' in augmentation_space:
        assert custom_augmentation_space_augs is not None

        custom_augmentation_space_augs_mapping = {
            'identity': identity,
            'auto_contrast': auto_contrast,
            'equalize': equalize,
            'rotate': rotate,
            'solarize': solarize,
            'color': color,
            'posterize': posterize,
            'contrast': contrast,
            'brightness': brightness,
            'sharpness': sharpness,
            'shear_x': shear_x,
            'shear_y': shear_y,
            'translate_x': translate_x,
            'translate_y': translate_y,
            # sample_pairing,
            'blur': blur,
            'invert': invert,
            'flip_lr': flip_lr,
            'flip_ud': flip_ud,
            'cutout': cutout,
            'crop_bilinear': crop_bilinear,
            'contour': contour,
            'detail': detail,
            'edge_enhance': edge_enhance,
            'sharpen': sharpen,
            'max_': max_,
            'min_': min_,
            'median': median,
            'gaussian': gaussian
        }
        ALL_TRANSFORMS = []
        ALL_TRANSFORMS += [
            custom_augmentation_space_augs_mapping[aug] for aug in custom_augmentation_space_augs
        ]
        print("CUSTOM Augs set to:", ALL_TRANSFORMS)
    else:
        if 'standard' not in augmentation_space:
            raise ValueError(f"Unknown search space {augmentation_space}")
        ALL_TRANSFORMS = [
            identity,
            auto_contrast,
            equalize,
            rotate,  # extra coin-flip
            solarize,
            color,  # enhancer
            posterize,
            contrast,  # enhancer
            brightness,  # enhancer
            sharpness,  # enhancer
            shear_x,  # extra coin-flip
            shear_y,  # extra coin-flip
            translate_x,  # extra coin-flip
            translate_y  # extra coin-flip
        ]

set_augmentation_space('fixed_standard', 31)


def apply_augmentation(aug_idx, m, img):
    return ALL_TRANSFORMS[aug_idx].pil_transformer(1., m)(img)


def num_augmentations():
    return len(ALL_TRANSFORMS)


class TrivialAugment:
    def __call__(self, img):
        op = random.choices(ALL_TRANSFORMS, k=1)[0]
        # op = flip_ud
        print('op')
        print(op)
        level = random.randint(0, PARAMETER_MAX)
        # level = 0
        print('level')
        print(level)
        img = op.pil_transformer(1., level)(img)
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30]

    def __call__(self, img):
        ops = random.choices(ALL_TRANSFORMS, k=self.n)
        for op in ops:
            img = op.pil_transformer(1., self.m)(img)

        return img


class UniAugment:
    def __call__(self, img):
        ops = random.choices(ALL_TRANSFORMS, k=2)
        for op in ops:
            level = random.randint(0, PARAMETER_MAX)
            img = op.pil_transformer(0.5, level)(img)
        return img


class UniAugmentWeighted:
    def __init__(self, n, probs):
        self.n = n
        self.probs = probs  # [prob of zero augs, prob of one aug, ..]

    def __call__(self, img):
        k = random.choices(range(len(self.probs)), self.probs)[0]
        ops = random.choices(ALL_TRANSFORMS, k=k)
        for op in ops:
            level = random.randint(0, PARAMETER_MAX)
            img = op.pil_transformer(1., level)(img)
        return img

if __name__ == '__main__':
    print('main')
    img = Image.open(r'C:\Users\t-jiahuihe\OneDrive\learn\pytorch\chapter_computer-vision\dog.jpg')
    augmenter = TrivialAugment()
    aug_img = augmenter(img)
    print(aug_img.size)# 原来(600, 744)(宽，长)
    print(F.to_tensor(img).size())# F.to_tensor(img).size() torch.Size([3, 744, 600])
    print(F.to_tensor(aug_img).size())
    plt.figure('image')
    plt.imshow(aug_img)
    plt.show()
    # plt.imshow(img)
    # plt.show()
    