# gpu_filters_rgb.py
import numpy as np
from PIL import Image
import pycuda.autoinit 
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import time
import math

# Generadores de Kernel
def generar_kernel_sobel_grande(size: int):
    if size % 2 == 0:
        size += 1
    mitad = size // 2
    Kx = np.zeros((size, size), dtype=np.float32)
    Ky = np.zeros((size, size), dtype=np.float32)
    for y in range(-mitad, mitad + 1):
        for x in range(-mitad, mitad + 1):
            Kx[y + mitad, x + mitad] = float(x)
            Ky[y + mitad, x + mitad] = float(y)
    sumaX = np.sum(np.abs(Kx)) or 1.0
    sumaY = np.sum(np.abs(Ky)) or 1.0
    Kx /= sumaX
    Ky /= sumaY
    return Kx.astype(np.float32), Ky.astype(np.float32)


def generar_kernel_emboss_grande(size: int):
    if size % 2 == 0:
        size += 1
    mitad = size // 2
    K = np.zeros((size, size), dtype=np.float64)
    for y in range(size):
        for x in range(size):
            K[y, x] = (x - mitad) + (y - mitad)
    K = K / (size * size)
    return K.astype(np.float32)


def generar_kernel_gaussiano(size: int, sigma: float):
    if size % 2 == 0:
        size += 1
    mitad = size // 2
    K = np.zeros((size, size), dtype=np.float64)
    s = 0.0
    for y in range(-mitad, mitad + 1):
        for x in range(-mitad, mitad + 1):
            v = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
            K[y + mitad, x + mitad] = v
            s += v
    K = K / (s if s != 0 else 1.0)
    return K.astype(np.float32)


def generar_kernel_sharpen_grande(size: int, sharp_factor: float = 1.0):
    if size % 2 == 0:
        size += 1
    mitad = size // 2
    K = np.zeros((size, size), dtype=np.float64)
    for y in range(-mitad, mitad + 1):
        for x in range(-mitad, mitad + 1):
            idy = y + mitad
            idx = x + mitad
            if x == 0 and y == 0:
                K[idy, idx] = 5.0
            elif (abs(x) == 1 and y == 0) or (abs(y) == 1 and x == 0):
                K[idy, idx] = -1.0
            else:
                K[idy, idx] = 0.0
    s = np.sum(np.abs(K)) or 1.0
    K = K / s
    K = K * sharp_factor
    return K.astype(np.float32)


# Kernel de CUDA
cuda_source = r"""
#include <math.h>

extern "C" {

__global__ void conv2d_float(
    const unsigned char *input,
    const float *kernel,
    float *output,
    int width, int height, int ksize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int half = ksize / 2;
    float acc = 0.0f;

    for (int ky = -half; ky <= half; ++ky) {
        int py = y + ky;
        if (py < 0) py = 0;
        if (py >= height) py = height - 1;
        for (int kx = -half; kx <= half; ++kx) {
            int px = x + kx;
            if (px < 0) px = 0;
            if (px >= width) px = width - 1;
            float pix = (float)input[py * width + px];
            int ki = (ky + half) * ksize + (kx + half);
            float kval = kernel[ki];
            acc += pix * kval;
        }
    }
    output[y * width + x] = acc;
}

__global__ void sobel_float(
    const unsigned char *input,
    const float *kernel_x,
    const float *kernel_y,
    float *output,
    int width, int height, int ksize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int half = ksize / 2;
    float sumX = 0.0f;
    float sumY = 0.0f;

    for (int ky = -half; ky <= half; ++ky) {
        int py = y + ky;
        if (py < 0) py = 0;
        if (py >= height) py = height - 1;
        for (int kx = -half; kx <= half; ++kx) {
            int px = x + kx;
            if (px < 0) px = 0;
            if (px >= width) px = width - 1;
            float pix = (float)input[py * width + px];
            int ki = (ky + half) * ksize + (kx + half);
            sumX += pix * kernel_x[ki];
            sumY += pix * kernel_y[ki];
        }
    }
    output[y * width + x] = sqrtf(sumX * sumX + sumY * sumY);
}

}
"""

mod = SourceModule(cuda_source)
conv2d_float = mod.get_function("conv2d_float")
sobel_float = mod.get_function("sobel_float")


def gpu_conv_channel_float(channel_uint8: np.ndarray, kernel: np.ndarray, block=(16,16,1)):
    channel = np.ascontiguousarray(channel_uint8.astype(np.uint8))
    kernel_c = np.ascontiguousarray(kernel.astype(np.float32))
    h, w = channel.shape
    out = np.empty((h, w), dtype=np.float32)

    in_gpu = drv.mem_alloc(channel.nbytes)
    ker_gpu = drv.mem_alloc(kernel_c.nbytes)
    out_gpu = drv.mem_alloc(out.nbytes)

    drv.memcpy_htod(in_gpu, channel)
    drv.memcpy_htod(ker_gpu, kernel_c)

    bx, by, _ = block
    gx = (w + bx - 1) // bx
    gy = (h + by - 1) // by

    conv2d_float(in_gpu, ker_gpu, out_gpu,
                 np.int32(w), np.int32(h), np.int32(kernel_c.shape[0]),
                 block=block, grid=(gx, gy, 1))

    drv.memcpy_dtoh(out, out_gpu)

    in_gpu.free()
    ker_gpu.free()
    out_gpu.free()

    return out


def gpu_sobel_channel_float(channel_uint8: np.ndarray, Kx: np.ndarray, Ky: np.ndarray, block=(16,16,1)):
    channel = np.ascontiguousarray(channel_uint8.astype(np.uint8))
    Kx_c = np.ascontiguousarray(Kx.astype(np.float32))
    Ky_c = np.ascontiguousarray(Ky.astype(np.float32))

    h, w = channel.shape
    out = np.empty((h, w), dtype=np.float32)

    in_gpu = drv.mem_alloc(channel.nbytes)
    kx_gpu = drv.mem_alloc(Kx_c.nbytes)
    ky_gpu = drv.mem_alloc(Ky_c.nbytes)
    out_gpu = drv.mem_alloc(out.nbytes)

    drv.memcpy_htod(in_gpu, channel)
    drv.memcpy_htod(kx_gpu, Kx_c)
    drv.memcpy_htod(ky_gpu, Ky_c)

    bx, by, _ = block
    gx = (w + bx - 1) // bx
    gy = (h + by - 1) // by

    sobel_float(in_gpu, kx_gpu, ky_gpu, out_gpu,
                np.int32(w), np.int32(h), np.int32(Kx_c.shape[0]),
                block=block, grid=(gx, gy, 1))

    drv.memcpy_dtoh(out, out_gpu)

    in_gpu.free()
    kx_gpu.free()
    ky_gpu.free()
    out_gpu.free()

    return out


def select_filter_params(filter_name: str, ksize: int, sigma: float = None, sharp_factor: float = 1.0):
    name = filter_name.lower()
    if name == "sobel":
        Kx, Ky = generar_kernel_sobel_grande(ksize)
        return ("sobel", Kx, Ky)
    elif name == "emboss":
        K = generar_kernel_emboss_grande(ksize)
        return ("single", K)
    elif name == "gauss":
        sigma = sigma if sigma is not None else max(0.5, ksize / 6.0)
        K = generar_kernel_gaussiano(ksize, sigma)
        return ("single", K)
    elif name == "sharpen":
        K = generar_kernel_sharpen_grande(ksize, sharp_factor)
        return ("single", K)
    else:
        raise ValueError("Filtro inválido (sobel, emboss, gauss, sharpen)")


def run_gpu_filter_rgb(input_path: str,
                       output_path: str,
                       filter_name: str,
                       ksize: int,
                       block=(16,16,1),
                       factor: float = 1.0,
                       offset: float = 0.0,
                       sigma: float = None,
                       sharp_factor: float = 1.0):
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img)
    H, W, C = arr.shape

    mode_and_params = select_filter_params(filter_name, ksize, sigma, sharp_factor)
    mode = mode_and_params[0]
    params = mode_and_params[1:]

    start = time.perf_counter()
    out_channels = []
    for c in range(3):
        channel = arr[:, :, c]

        if mode == "sobel":
            Kx, Ky = params
            conv = gpu_sobel_channel_float(channel, Kx, Ky, block)
            result = conv * factor + offset
        else:
            (K,) = params
            conv = gpu_conv_channel_float(channel, K, block)
            result = conv * factor + offset

        result_clamped = np.clip(result, 0.0, 255.0).astype(np.uint8)
        out_channels.append(result_clamped)

    out_img = np.stack(out_channels, axis=2)
    Image.fromarray(out_img, mode="RGB").save(output_path)

    elapsed = time.perf_counter() - start
    return elapsed, (W, H)

