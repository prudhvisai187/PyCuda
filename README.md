# PyCuda

# GPU-Accelerated Image Processing using PyCUDA

This project demonstrates the application of GPU acceleration using **PyCUDA** to perform image processing tasks (such as convolution-based filtering) on randomly generated images. It highlights the significant performance benefits of using GPU over traditional CPU computation and lays the foundation for extending such techniques to machine learning, big data analytics, and computer vision.

---

## 🚀 Features

- GPU-accelerated 2D convolution on grayscale images
- Implementation of custom CUDA kernel for filtering
- Visualization of original, filtered, and difference images
- Performance and accuracy comparison using metrics
- Uses a Laplacian-like filter for edge detection

---

## 🧠 Technologies Used

- [Python 3.x](https://www.python.org/)
- [NumPy](https://numpy.org/)
- [PyCUDA](https://documen.tician.de/pycuda/)
- [Matplotlib](https://matplotlib.org/)

---


---

## 🔧 How It Works

1. Generates a synthetic 256x256 grayscale image.
2. Defines a 3x3 filter kernel (e.g., Laplacian filter).
3. Transfers image and kernel data to the GPU.
4. Launches a custom CUDA kernel using PyCUDA for convolution.
5. Retrieves the filtered output and visualizes it.
6. Computes difference between original and filtered images.

---

## 📈 Metrics Output

The script displays:
- **Filtered image values** (sample)
- **Visualizations** of the original, filtered, and difference images
- **Metrics** such as minimum, maximum, and mean differences



