import torch


def spectral_radius(mat: torch.Tensor) -> float:
    """
    Compute the spectral radius of a square matrix.

    Args:
        mat (torch.tensor): A square matrix represented as a 2D tensor.

    Returns:
        float: The spectral radius of the matrix
    """
    if mat.ndim != 2:
        raise ValueError("The input matrix must be a 2D tensor.")
    if mat.size(0) != mat.size(1):
        raise ValueError("The input matrix must be square (number of rows must equal number of columns).")
    
    eigenvalues = torch.linalg.eigvals(mat)
    return torch.max(torch.abs(eigenvalues)).item()


if __name__ == "__main__":
    mat = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    temp = spectral_radius(mat)
    print(temp)
    
    rhow = 0.95
    
    mat *= rhow / temp
    
    print(f"New: {spectral_radius(mat)}")
    print(mat)
