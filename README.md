# Homework for linma2380: Matrix Computations

Given a large matrix $A$, find smaller matrices $Q,H$ such that: $Q^\top AQ = H$

Using Krylov subspaces and Arnoldi iteration.

Given a matrix $A \in \mathbb{R}^{n\times n}$, and $b \in \mathbb{R}^n$ Arnoldi method provides a matrix $Q \in \mathbb{R}^{n\times r}$ where $r<n$ and a Hessenberg matrix $H \in \mathbb{R}^{r\times r}$ such that: $H = Q^\top AQ$

Where $Q$ has orthonormal columns.

Then eigenvalues and eigenvectors of $H$ can be used to produce good approximated for eigenvalues and eigenvectors of $A$.
