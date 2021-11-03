# MeshGradientPy

(Status: In Progress)
Modification of [DonsetPG/MeshGradientPy](https://github.com/DonsetPG/MeshGradientPy) to use [PyTorch](https://pytorch.org/) instead of [TensorFlow](https://www.tensorflow.org/). 

---------------------------------------

- [1. What it does](#1-what-it-does)
- [2. Libraries](#2-libraries)
- [3. Example](#3-example)
  - [3.1. Multiprocessing](#31-multiprocessing)
- [4. Background](#4-background)
  - [4.1. PCE](#41-pce)
  - [4.2. AGS](#42-ags)

---------------------------------------

## 1. What it does
Compute gradients on mesh and unstructured data objects.

This repository aims to fill a gap: no native Python code was available to compute a particular field gradient on a mesh. Some implementations may exist for a structured grid, but this is the first time gradient can be calculated on unstructured mesh (without external libraries such as Paraview or Pyvista).

While Pyvista does provide this functionality, it gets impossible to make it work with other libraries, such as Machine Learning framework. Therefore, we built this library using Tensorflow (which can be replaced with other Deep Learning framework), allowing one to compute gradient on a mesh while performing gradient descent.

## 2. Libraries 

We used numpy and PyTorch (TensorFlow in the [original repo](https://github.com/DonsetPG/MeshGradientPy)) for the computation part and meshio to read/write mesh.

We also developed a multiprocessing version of our functions based on the library Ray. This gives full support to tailored computing power even on large clusters.

You can install a working conda env:
```
conda env create -f conda_env.yaml
```

## 3. Example 

We can read a mesh using meshio: 
```python3
mesh_pv = meshio.read('example.vtu',"vtu")
```
and compute an AGS matrix with the following: 
```python3 
from meshgradient.matrix import build_AGS_matrix
matrix = build_AGS_matrix(mesh)
```

More informations can be found in this [notebook](Example.ipynb).

### 3.1. Multiprocessing

Each matrix can also be computed in a multi processed fashion using the `_multiprocessing` function, such as `build_AGS_matrix_multiprocess`. 

## 4. Background 

We use three different methods to compute the gradient of a field:
* PCE (Per-Cell Linear Estimation)
* AGS (Average Gradient on Star)
* CON (Connectivity, for gradients on boundaries)

You can use any of these three methods or use our built-in functions to compute gradients (that makes the best of these three methods at the same time)

We based our implementation on the one provided by the authors of the paper [Gradient Field Estimation on Triangle Meshes](https://www.researchgate.net/publication/330412652_Gradient_Field_Estimation_on_Triangle_Meshes). We describe below the main ideas of the methods, more details can be found in the paper.

These three methods were built for triangle cells. Any other sort of cells won't be considered. 

### 4.1. PCE 

This method estimates a constant gradient inside each cell. First, we define a linear interpolation at any point $p$ in a cell of a function $f$ with: 

$$
f_\sigma(p) = \sum_{v_i \in \sigma} \lambda_i f_i
$$

where $v_i$ are the vertices of the cell and $\lambda_i$ the barycentric coordinates of $p$ wrt. the vertices. 

With this estimation, for a triangle with 3 vertices $v_i$, $v_j$ and $v_k$, we have:

$$
\nabla f_t = (f_j - f_i) \frac{(v_i - v_k)^\bot}{2A} + (f_k - f_i) \frac{(v_j - v_i)^\bot}{2A}
$$


where $A$ is the area of the triangle.

### 4.2. AGS

Given a node, we can use the PCE method to compute a gradient in each cell $\sigma$ containing the node $v$, thus having:

$$
\nabla f_v = \frac{1}{\sum_{\sigma \in \mathcal{N}(v)} A_\sigma} \sum_{\sigma \in \mathcal{N}(v)} A_\sigma \nabla f_\sigma
$$
