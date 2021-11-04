from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Tuple, Optional, List, Any, Union

import numpy as np

# PyTorch or Tensorflow
from .backend import Tensor, SparseTensor, matmul, multiply, reshape, repeat, cast, expand_dims, clip_by_value, concat, float32

def compute_div_from_grad(
    grad: Union[Tensor, np.ndarray]
) -> Union[Tensor, np.ndarray]:
    """Compute divergence of a 2D vector field from the gradient of such vector field.
    Arguments:
        grad: gradient vector field
    Returns:
        divergence scalar field
    Raises:
    """
    return grad[:, 0] + grad[:, 4]


def compute_sca_grad_from_grad(
    field: Union[Tensor, np.ndarray], grad: Union[Tensor, np.ndarray]
) -> Tensor:
    """Compute scalar gradient operator for NS equations in 2D.
    Arguments:
        field: vector field (velocity)
        grad: gradient field
    Returns:
        results of the operator
    Raises:
    """
    output_x: Tensor = multiply(field[:, 0], grad[:, 0]) + multiply(
        field[:, 1], grad[:, 1]
    )
    output_x = reshape(output_x, (len(output_x), 1))
    output_y: Tensor = multiply(field[:, 0], grad[:, 3]) + multiply(
        field[:, 1], grad[:, 4]
    )
    output_y = reshape(output_y, (len(output_y), 1))
    output: Tensor = concat([output_x, output_y], axis=1)
    return output


def compute_gradient_per_points(
    gradient_matrices: Tuple[SparseTensor],
    F: Union[Tensor, np.ndarray],
    b1: Optional[Union[Tensor, np.ndarray]] = None,
    b2: Optional[Union[Tensor, np.ndarray]] = None,
    b3: Optional[Union[Tensor, np.ndarray]] = None,
    b4: Optional[Union[Tensor, np.ndarray]] = None,
) -> Tensor:
    """Compute gradient on vertex of a scalar fields.

    For most cells, gradient is computed accordingly to AGS methods and for
    the boundaries we use another average of cells gradient

    Arguments:
        gradient_matrices: matrices used to compute gradient (from matrix.py)
        F: scalar field on which gradient is computed
        b1: boundary flag
        b2: boundary flag
        b3: boundary flag
        b4: boundary flag
    Returns:
        The gradient field.
    Raises:
    """
    Sp_AGS_backend: SparseTensor
    Sp_PCE_backend: SparseTensor
    Sp_CON_backend: SparseTensor
    (Sp_AGS_backend, Sp_PCE_backend, Sp_CON_backend) = gradient_matrices

    backend_F: Tensor = cast(expand_dims(F, axis=-1), float32)
    # Compute gradient on points
    gp_F: Tensor = matmul(Sp_AGS_backend, backend_F)

    # Compute gradient on boundaries
    gc_F: Tensor = matmul(Sp_PCE_backend, backend_F)
    gc_F = reshape(gc_F, (Sp_CON_backend.shape[1], 3))
    gb_F: Tensor = matmul(Sp_CON_backend, gc_F)
    gb_F = reshape(gb_F, (gb_F.shape[0] * 3, 1))

    g_F: Tensor
    if b1 is not None:
        mask: Tensor = 1 - clip_by_value(
            (b1 + b2 + b3 + b4), clip_value_min=0.0, clip_value_max=1.0
        )
        mask = repeat(mask, 3)
        mask = reshape(mask, (len(mask), 1))
        mask = cast(mask, float32)
        g_F = multiply(gp_F, mask) + multiply(gb_F, 1 - mask)
    else:
        g_F = gp_F

    g_F = reshape(g_F, (len(F), 3))
    return g_F


def compute_laplacian_scalar_field(
    gradient_matrices: Tuple[SparseTensor],
    F: Union[Tensor, np.ndarray],
    b1: Optional[Union[Tensor, np.ndarray]] = None,
    b2: Optional[Union[Tensor, np.ndarray]] = None,
    b3: Optional[Union[Tensor, np.ndarray]] = None,
    b4: Optional[Union[Tensor, np.ndarray]] = None,
) -> Tensor:
    """Compute laplacian on vertex of a scalar fields.

    Arguments:
        G: matrices to compute gradient
        F: scalar field on which gradient is computed
        b1: boundary flag
        b2: boundary flag
        b3: boundary flag
        b4: boundary flag
    Returns:
        The laplacian field.
    Raises:
    """
    grad_SF: Tensor = compute_gradient_per_points(
        gradient_matrices, F, b1, b2, b3, b4
    )

    grad_grad_SF_x: Tensor = compute_gradient_per_points(
        gradient_matrices, grad_SF[:, 0], b1, b2, b3, b4
    )
    grad_grad_SF_y: Tensor = compute_gradient_per_points(
        gradient_matrices, grad_SF[:, 1], b1, b2, b3, b4
    )
    grad_grad_SF: Tensor = concat([grad_grad_SF_x, grad_grad_SF_y], axis=1)

    laplacian_SF: Tensor = compute_div_from_grad(grad_grad_SF)
    laplacian_SF = reshape(laplacian_SF, (len(laplacian_SF), 1))
    return laplacian_SF


def compute_laplacian_vector_field(
    gradient_matrices: Tuple[SparseTensor],
    F: Union[Tensor, np.ndarray],
    b1: Optional[Union[Tensor, np.ndarray]] = None,
    b2: Optional[Union[Tensor, np.ndarray]] = None,
    b3: Optional[Union[Tensor, np.ndarray]] = None,
    b4: Optional[Union[Tensor, np.ndarray]] = None,
) -> Tensor:
    """Compute laplacian on vertex of a vector fields.

    Arguments:
        gradient_matrices: matrices to compute gradient
        F: scalar field on which gradient is computed
        b1: boundary flag
        b2: boundary flag
        b3: boundary flag
        b4: boundary flag
    Returns:
        The laplacian field.
    Raises:
    """
    laplacian_SF_x: Tensor = compute_laplacian_scalar_field(
        gradient_matrices, F[:, 0], b1, b2, b3, b4
    )
    laplacian_SF_y: Tensor = compute_laplacian_scalar_field(
        gradient_matrices, F[:, 1], b1, b2, b3, b4
    )
    laplacian_SF: Tensor = concat([laplacian_SF_x, laplacian_SF_y], axis=1)
    return laplacian_SF
