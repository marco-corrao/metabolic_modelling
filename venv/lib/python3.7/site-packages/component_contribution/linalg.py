"""A file with some Linear Algebra tools."""
# The MIT License (MIT)
#
# Copyright (c) 2018 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from typing import Tuple

import numpy as np
from numpy.linalg import pinv
from scipy.linalg import qr, svd


class LINALG(object):
    """Some general linear algebra that's missing from scipy."""

    @staticmethod
    def invert_project(
        A: np.ndarray, tol: float = 1e-10
    ) -> Tuple[np.array, float, np.array, np.array]:
        """Use SVD to invert a matrix, while computing the subspace projections.

        Parameters
        ----------
        A : np.ndarray
            A 2D NumPy array
        tol : float, optional
            A threshold on the eigenvalues (below which are considered 0)
            (Default value = 1e-10)

        Returns
        -------
        tuple
            a 4-tuple of (inv(A), rank, P_R, P_N) where the P_R is the
            projection onto the range of A, and P_N is the projection onto the
            nullspace of A.T

        """
        M, N = A.shape

        U, s, Vh = svd(
            A,
            full_matrices=True,
            compute_uv=True,
            check_finite=True,
            lapack_driver="gesvd",
        )

        assert U.shape == (M, M), "SVD decomposition dimensions are wrong"
        assert Vh.shape == (N, N), "SVD decomposition dimensions are wrong"

        S = np.zeros((M, N))
        np.fill_diagonal(S, s)
        inv_A = Vh.T @ pinv(S) @ U.T

        rank = (S > tol).sum()
        P_R = U[:, :rank] @ U[:, :rank].T
        P_N = U[:, rank:] @ U[:, rank:].T

        assert inv_A.shape == (N, M), "Pseudoinverse dimensions are wrong"
        assert P_R.shape == (M, M), "Orthogonal projection dimensions are wrong"
        assert P_N.shape == (M, M), "Orthogonal projection dimensions are wrong"

        return inv_A, rank, P_R, P_N

    @staticmethod
    def row_uniq(A: np.ndarray) -> Tuple[np.array, np.array]:
        """Matrix Row Unique.

        A procedure usually performed before linear regression (i.e. solving
        Ax = y). If the matrix A contains repeating rows, it is advisable to
        combine all of them to one row, and the observed value corresponding
        to that row will be the average of the original observations.

        Parameters
        ----------
        A : np.ndarray
            A 2D NumPy array

        Returns
        -------
        tuple
            a 2-tuple (A_unique, P_row) where A_unique has the same
            number of columns as A, but with unique rows. P_row is a matrix that
            can be used to map the original rows to the ones in A_unique (all
            values in P_row are 0 or 1).

        """
        # convert the rows of A into tuples so we can compare them
        A_tuples = [tuple(A[i, :].flat) for i in range(A.shape[0])]
        A_unique = list(sorted(set(A_tuples), reverse=True))

        # create the projection matrix that maps the rows in A to rows in
        # A_unique
        P_col = np.zeros((len(A_unique), len(A_tuples)), dtype=float)

        for j, tup in enumerate(A_tuples):
            # find the indices of the unique row in A_unique which correspond
            # to this original row in A (represented as 'tup')
            i = A_unique.index(tup)
            P_col[i, j] = 1.0

        return np.array(A_unique, ndmin=2), P_col

    @staticmethod
    def col_uniq(A: np.array) -> Tuple[np.array, np.array]:
        """Matrix Column Unique.

        A procedure usually performed before linear regression (i.e. solving
        Ax = y). If the matrix A contains repeating columns, it is advisable to
        combine all of them to one column, and the observed value corresponding
        to that column will be the average of the original observations.

        Parameters
        ----------
        A : np.ndarray
            A 2D NumPy array

        Returns
        -------
        tuple
            a 2-tuple (A_unique, P_col) where A_unique has the same
            number of rows as A, but with unique columns. P_col is a matrix that
            can be used to map the original columns to the ones in A_unique (all
            values in P_col are 0 or 1).

        """
        A_unique, P_col = LINALG.row_uniq(A.T)
        return A_unique.T, P_col.T

    @staticmethod
    def qr_rank_deficient(A: np.array, tol: float = 1e-10) -> np.array:
        """Rank-revealing row elimination of A.

        Uses the QR decomposition to find the solution to R.T @ R = A.T @ A,
        where R is fully ranked.

        Parameters
        ----------
        A : np.array
            A n-by-m matrix with rank r < min(n, m)
        tol : float, optional
            The tolerance below which an eigenvalue is considered 0.
            The default is 1e-10.

        Returns
        -------
        R : np.array
            A matrix of size r-by-m, that satisfies R.T @ R = A.T @ A

        """
        _, R, perm = qr(A, mode="economic", pivoting=True)
        r = sum(np.abs(np.diag(R)) > tol)

        # invert the permutation in order to apply it on M and return
        # the columns to the original order as in A
        perm = perm.tolist()
        invperm = [perm.index(i) for i in range(len(perm))]

        # keep only the non-zero rows, and reorder the columns according to
        # the original ordering in A
        return R[:r, invperm]
