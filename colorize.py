import numpy as np


class Layer(object):

    def __init__(self, alpha, color):

        # alpha for the whole image:
        assert alpha.ndim == 2
        self.alpha = alpha
        [n, m] = alpha.shape[:2]

        color = np.atleast_1d(np.array(color)).astype(np.uint8)
        # color for the image:
        if color.ndim == 1:  # constant color for whole layer
            ncol = color.size
            if ncol == 1:  # grayscale layer
                self.color = color * np.ones((n, m, 3), dtype=np.uint8)
            if ncol == 3:
                self.color = np.ones(
                    (n, m, 3), dtype=np.uint8) * color[None, None, :]
        elif color.ndim == 2:  # grayscale image
            self.color = np.repeat(
                color[:, :, None], repeats=3, axis=2).copy().astype(np.uint8)
        elif color.ndim == 3:  # rgb image
            self.color = color.copy().astype(np.uint8)
        else:
            print(color.shape)
            raise Exception("color datatype not understood")


def color(text_arr, fg_col, bg_col):

    l_text = Layer(alpha=text_arr, color=fg_col)

    layers = [l_text]
    blends = []

    gray_layers = layers.copy()
    gray_blends = blends.copy()
    l_bg_gray = Layer(alpha=255*np.ones_like(text_arr,
                      dtype=np.uint8), color=bg_col)
    gray_layers.append(l_bg_gray)
    gray_blends.append('normal')
    l_normal_gray = merge_down(gray_layers, gray_blends)

    return l_normal_gray.color


def blend(cf, cb, mode='normal'):

    return cf


def merge_down(layers, blends=None):

    nlayers = len(layers)
    if nlayers > 1:
        [n, m] = layers[0].alpha.shape[:2]
        out_layer = layers[-1]
        for i in range(-2, -nlayers-1, -1):
            blend = None
            if blends is not None:
                blend = blends[i+1]
                out_layer = merge_two(
                    fore=layers[i], back=out_layer, blend_type=blend)
        return out_layer
    else:
        return layers[0]


def merge_two(fore, back, blend_type=None):

    a_f = fore.alpha / 255.0
    a_b = back.alpha / 255.0
    c_f = fore.color
    c_b = back.color

    a_r = a_f + a_b - a_f*a_b
    if blend_type != None:
        c_blend = blend(c_f, c_b, blend_type)
        c_r = (((1-a_f)*a_b)[:, :, None] * c_b
               + ((1-a_b)*a_f)[:, :, None] * c_f
               + (a_f*a_b)[:, :, None] * c_blend)
    else:
        c_r = (((1-a_f)*a_b)[:, :, None] * c_b
               + a_f[:, :, None]*c_f)

    return Layer((255 * a_r).astype(np.uint8), c_r.astype(np.uint8))
