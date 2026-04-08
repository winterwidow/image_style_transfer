# Image Style Transfer

## Core Idea:

Images are matrices.

SVD decomposes any matrix:

𝐴 =𝑈Σ𝑉ᵀ

Where:

U → row patterns

Vᵀ → column patterns

Σ → singular values (importance/energy)

Singular values control:

- Texture
- Intensity
- Detail level

So:
Keep U and V from content image

Replace Σ from style image

We inject “style energy” into content structure.

## Process:

1. Convert both images to grayscale

C = Content Image

S= Style Image

2. Compute SVD for both images:

𝐶=𝑈𝑐Σ𝑐𝑉𝑐𝑇

𝑆=𝑈𝑠Σ𝑠𝑉𝑠𝑇

3. Construct new matrix:

T=Uc​Σs​VcT​

U - left singular matrix

- Columns of 𝑈 are orthonormal vectors
- They form a basis for the column space
- They are eigenvectors of 𝐴𝐴𝑇𝑢 = 𝜆𝑢
- In an image:
  - 𝑈 captures vertical structure patterns
  - Row-wise structure
  - Large-scale shapes

Σ — Singular Values Matrix

- Diagonal matrix
- Singular values are non-negative
- Sorted largest to smallest
- Related to square roots of eigenvalues: 𝜎𝑖 = root(𝜆𝑖)

- They measure:
  - Importance of each pattern
  - Energy in each direction
  - Strength of contribution

  Large singular value → that pattern matters a lot

  Small singular value → fine details / noise

Vᵀ — Right Singular Vectors

-
