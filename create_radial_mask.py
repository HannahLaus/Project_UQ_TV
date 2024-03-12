import math
import numpy as np
import skimage.transform
import torch

"""
Based on https://github.com/jmaces/robust-nets/tree/master/fastmri-radial 
and modified in agreement with their license. 

-----

MIT License

Copyright (c) 2020 Martin Genzel, Jan Macdonald, Maximilian März

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class RadialMaskFunc(object):
    """ Generates a golden angle radial spokes mask.

    Useful for subsampling a Fast-Fourier-Transform.
    Contains radial lines (spokes) through the center of the mask, with
    angles spaced according to the golden angle (~111.25°). The first line
    has angle 0° (horizontal). An offset parameter can be given to skip
    the first `offset*num_lines` lines.

    Parameters
    ----------
    shape : array_like
        A tuple specifying the size of the mask.
    num_lines : int
        Number of radial lines (spokes) in the mask.
    offset : int, optional
        Offset factor for the range of angles in the mask.
    """

    def __init__(self, shape, num_lines, offset=0):
        self.shape = shape
        self.num_lines = num_lines
        self.offset = offset
        self.mask = self._generate_radial_mask(shape, num_lines, offset)

    def __call__(self, shape, seed=None):

        if (self.mask.shape[0] != shape[-3]) or (
            self.mask.shape[1] != shape[-2]
        ):
            return torch.zeros(shape)

        return torch.reshape(
            self.mask, (len(shape) - 3) * (1,) + self.shape + (1,)
        )

    def _generate_radial_mask(self, shape, num_lines, offset=0):
        # generate line template and empty mask
        x, y = shape
        d = math.ceil(np.sqrt(2) * max(x, y))
        line = np.zeros((d, d))
        line[d // 2, :] = 1.0
        out = np.zeros((d, d))
        # compute golden angle sequence
        golden = (np.sqrt(5) - 1) / 2
        angles = (
            180.0
            * golden
            * np.arange(offset * num_lines, (offset + 1) * num_lines)
        )
        # draw lines
        for angle in angles:
            out += skimage.transform.rotate(line, angle, order=0)
        # crop mask to correct size
        out = out[
            d // 2 - math.floor(x / 2) : d // 2 + math.ceil(x / 2),
            d // 2 - math.floor(y / 2) : d // 2 + math.ceil(y / 2),
        ]
        # return binary mask
        return torch.tensor(out > 0)

mask_func = RadialMaskFunc((156,156), 47)
mask = np.array(mask_func.mask)* np.ones((156,156))

mask.dump("radial_mask.dat", protocol=4)
