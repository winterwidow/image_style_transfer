# Image Style Transfer using Linear Algebra (SVD)

## Overview

This project demonstrates **image style transfer using Singular Value Decomposition (SVD)** — a purely **linear algebra–based approach** without using deep learning.

The idea is to represent images as matrices and manipulate their internal structure using matrix decomposition to achieve a stylized effect.

---

## Core Concept

Any image can be represented as a matrix \( A \).  
Using SVD, we decompose it as:

\[
A = U \Sigma V^T
\]

Where:

- **U** → captures structural (row-wise) patterns  
- **Vᵀ** → captures spatial/column patterns  
- **Σ** → contains singular values (energy/importance)

---

## Interpretation in Images

### U (Left Singular Vectors)
- Orthonormal basis of column space  
- Eigenvectors of \( AA^T \)  
- Represents:
  - Large-scale structure  
  - Vertical patterns  
  - Overall shape  

---

### Σ (Singular Values)
- Diagonal matrix  
- Sorted in decreasing order  
- Related to eigenvalues:

\[
\sigma_i = \sqrt{\lambda_i}
\]

Represents:
- Importance of each pattern  
- Texture strength  
- Level of detail  

 Large values → dominant structures  
 Small values → fine details / noise  

---

### Vᵀ (Right Singular Vectors)
- Basis of row space  
- Eigenvectors of \( A^T A \)  
- Represents:
  - Horizontal patterns  
  - Spatial distribution  

---

## Style Transfer Idea

Given:

- **Content Image (C)** → provides structure  
- **Style Image (S)** → provides texture/energy  

We compute:

\[
C = U_c \Sigma_c V_c^T
\]
\[
S = U_s \Sigma_s V_s^T
\]

Then construct:

\[
T = U_c \Sigma_s V_c^T
\]

This keeps:
- Structure from content  
- Style from style image  

---

## Workflow

1. Load content and style images  
2. Convert images into matrix form  
3. Perform SVD on both images  
4. Replace singular values of content with style  
5. Reconstruct the image  
6. Apply enhancements:
   - Region masking (localized transfer)
   - Directional texture extraction
   - Swirl transformation (artistic effect)
   - Detail enhancement

---

## Limitations

- Cannot fully replicate artistic styles like neural networks  
- Lacks local feature understanding  
- Produces approximate style effects  

---


## Technologies Used

- Python  
- NumPy  
- OpenCV  
- Matplotlib  

---