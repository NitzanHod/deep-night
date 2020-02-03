import torch
from utils.train_utils.cuda_utils import set_cuda
import time
def lab_to_rgb(lab):
    dtype, device = set_cuda()
    lab = lab.to(device).type(dtype)
    l = lab[:, 0, :, :]
    a = lab[:, 1, :, :]
    b = lab[:, 2, :, :]


    y = (l+16) / 116
    x = (a/500) + y
    z = y-(b/200)
#
    #x1 = torch.zeros_like(x).to(device).type(dtype)
    #y1 = torch.zeros_like(y).to(device).type(dtype)
    #z1 = torch.zeros_like(z).to(device).type(dtype)

    x1 = x.clone()
    x1[:] = 0
    y1 = y.clone()
    y1[:] = 0
    z1 = z.clone()
    z1[:] = 0


    x_3 = x**3
    y_3 = y**3
    z_3 = z**3

    x1 = torch.where(x_3 > 0.008856, x_3, ((x - (16 / 116)) / 7.787))
    y1 = torch.where(y_3 > 0.008856, y_3, ((y - (16 / 116)) / 7.787))
    z1 = torch.where(z_3 > 0.008856, z_3, ((z - (16 / 116)) / 7.787))

    x1 *= 0.95047
    y1 *= 1.00000
    z1 *= 1.08883

    r1 = x1 *  3.2406 + y1 * -1.5372 + z1 * -0.4986
    g1 = x1 * -0.9689 + y1 *  1.8758 + z1 *  0.0415
    b1 = x1 *  0.0557 + y1 * -0.2040 + z1 *  1.0570

    #r = torch.zeros_like(r1).to(device).type(dtype)
    #g = torch.zeros_like(g1).to(device).type(dtype)
    #b = torch.zeros_like(b1).to(device).type(dtype)

    r = r1.clone()
    r[:] = 0
    g = g1.clone()
    g[:] = 0
    b = b1.clone()
    b[:] = 0

    r = torch.where(r1 > 0.0031308, (1.055 * r1 ** (1 / 2.4) - 0.055), 12.92 * r1)
    g = torch.where(g1 > 0.0031308, (1.055 * g1 ** (1 / 2.4) - 0.055), 12.92 * g1)
    b = torch.where(b1 > 0.0031308, (1.055 * b1 ** (1 / 2.4) - 0.055), 12.92 * b1)

    r = 255 * torch.clamp(torch.clamp(r, min=0), max=1)
    g = 255 * torch.clamp(torch.clamp(g, min=0), max=1)
    b = 255 * torch.clamp(torch.clamp(b, min=0), max=1)

    return torch.stack((r, g, b),dim=1)


def rgb_to_lab(rgb):
    r = rgb[:, 0, :, :]/255
    g = rgb[:, 1, :, :]/255
    b = rgb[:, 2, :, :]/255

    r1 = r.clone()
    r1[:] = 0
    g1 = g.clone()
    g1[:] = 0
    b1 = b.clone()
    b1[:] = 0

    r1 = torch.where(r > 0.04045, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
    g1 = torch.where(g > 0.04045, ((g + 0.055) / 1.055) ** 2.4, g / 12.92)
    b1 = torch.where(b > 0.04045, ((b + 0.055) / 1.055) ** 2.4, b / 12.92)

    x = (r1 * 0.4124 + g1 * 0.3576 + b1 * 0.1805) / 0.95047
    y = (r1 * 0.2126 + g1 * 0.7152 + b1 * 0.0722) / 1.00000
    z = (r1 * 0.0193 + g1 * 0.1192 + b1 * 0.9505) / 1.08883

    x1 = x.clone()
    x1[:] = 0
    y1 = y.clone()
    y1[:] = 0
    z1 = z.clone()
    z1[:] = 0

    x1 = torch.where(x > 0.008856, x ** (1/3), (7.787 * x) + 16 / 116)
    y1 = torch.where(y > 0.008856, y ** (1 / 3), (7.787 * y) + 16 / 116)
    z1 = torch.where(z > 0.008856, z ** (1 / 3), (7.787 * z) + 16 / 116)

    l, a, b = (116 * y1) - 16, 500 * (x1 - y1), 200 * (y1 - z1)
    return torch.stack((l, a, b),dim=1) #  stack along correct dimension

def main():
    py_pixel = [20, 192, 17]
    py_row = [py_pixel, py_pixel, py_pixel]
    py_list = [py_row, py_row, py_row]
    tt = torch.Tensor(py_list)
    print(tt.size())
    print('res',lab_to_rgb(rgb_to_lab(tt)))


if __name__ == '__main__':
    main()
