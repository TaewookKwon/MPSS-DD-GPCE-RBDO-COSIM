import numpy as np
import math as mt
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import warnings

from sympy import symbols, itermonomials, lambdify, Mul
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold

from numpy.polynomial.hermite_e import hermeval
from numpy.polynomial.legendre import legval
from numpy.linalg import svd, pinv, inv
from collections import defaultdict

from scipy.stats import norm
from scipy.special import hermite
from scipy.integrate import quad


class SurrogateModel:

    ## GPCE basis construction
    def P_grlex_n(self, samples, m):
        """
        Sample-based basis construction for the GPCE model

        Parameters:
            samples : (L x N) matrix
            L: number of basis functions
            N: number of samples
            m : maximum polynomial order

        Returns:
            P_num : Monomial Basis
        """
         
        count = np.shape(samples)[1]  # number of variables (columns of the input data)
        variables = symbols(f'x:{count}')  # create the variables x0, x1, x2, ...
        P = sorted(itermonomials(variables, m), key=monomial_key('grlex', list(reversed(variables))))  # monomial basis
        var_dict = dict(zip(variables, samples.T))  # map each variable to its sample values

        # extract the variables only
        variables_from_P = [[factor.as_base_exp()[0] for factor in term.as_ordered_factors()] for term in P]

        row = len(P)
        column = len(samples)
        P_num = np.zeros((row, column))
        P_num[0, :] = 1  # constant term

        for idx, (i, j) in enumerate(zip(P[1:], variables_from_P[1:]), start=1):
            f = lambdify(j, i)
            if isinstance(j, list):
                args = [var_dict[v] for v in j]
                P_num[idx, :] = f(*args)
            else:
                P_num[idx, :] = f(var_dict[j])

        return P_num # return the monomial basis
    
    
    def P_grlex_n_DD(self, samples, m, S): 
        """
        Sample-based basis construction for the DD-GPCE model

        Parameters:
            samples : (L x N) matrix
            m : maximum polynomial order
            S : maximum interaction order

        Returns:
            P_num : Monomial Basis
        """

        count = np.shape(samples)[1] # number of variables (columns of the input data)
        variables = symbols(f'x:{count}') # create the variables x0, x1, x2, ...
        P_full = sorted(itermonomials(variables, m), key=monomial_key('grlex', list(reversed(variables)))) # monomial basis
        # grlex: graded lexicographic order; sorted by total degree, then lexicographically within the same degree

        var_dict = dict(zip(variables, samples.T))
        # map each variable (x0, x1, ..., x9) to its list of sample values,
        # e.g., three samples: (x0, [1, 11, 21]), (x1, [2,12,22]), (x3, [3,13,23])...

        def num_unique_vars(poly): # condition: support-set size <= S
            return len(set([factor.as_base_exp()[0] for factor in poly.as_ordered_factors()]))

        P = [p for p in P_full if num_unique_vars(p) <= S]

        # extract the variables only, x0^2*x1^3 -> [x0, x1]
        variables_from_P = [[factor.as_base_exp()[0] for factor in term.as_ordered_factors()] for term in P]

        row = len(P)
        column = len(samples)
        P_num = np.zeros((row, column))
        P_num[0, :] = 1  # constant term

        for idx, (i, j) in enumerate(zip(P[1:], variables_from_P[1:]), start=1):
            f = lambdify(j, i) # convert the symbolic polynomial to a numerical function
            if isinstance(j, list):
                args = [var_dict[v] for v in j] # sample values of all variables
                P_num[idx, :] = f(*args)
            else:
                P_num[idx, :] = f(var_dict[j])

        return P_num # return the monomial basis, N rows (basis functions) x L columns (samples)
    
    def P_grlex_n_DD_fast(self, samples, m, S):
        """
        Fast sample-based basis construction for the DD-GPCE model without SymPy
        
        Parameters:
            samples : (L x N) matrix
            m : maximum polynomial order
            S : maximum interaction order
            
        Returns:
            P_num : Monomial Basis (N_basis x L)
        """
        import itertools
        
        L, N = samples.shape
        
        # 1. Generate all admissible monomial exponent combinations
        basis_exponents = []
        
        # loop over the total degree from 0 to m (first key of the grlex order)
        for total_degree in range(m + 1):
            # all exponent combinations of N variables for the given degree
            # implemented directly instead of itertools.combinations_with_replacement
            for exp_combination in self._generate_exponent_combinations(N, total_degree):
                # interaction limit: number of variables with nonzero exponents <= S
                non_zero_vars = sum(1 for exp in exp_combination if exp > 0)
                if non_zero_vars <= S:
                    basis_exponents.append(list(exp_combination))
        
        # 2. Sort in graded lexicographic (grlex) order
        def grlex_key(exp_vec):
            total_deg = sum(exp_vec)
            # for equal degree, sort in reverse lexicographic order (from the last variable)
            lex_order = tuple(reversed(exp_vec))
            return (total_deg, lex_order)
        
        basis_exponents.sort(key=grlex_key)
        
        # 3. Evaluate the basis with vectorized NumPy operations
        num_basis = len(basis_exponents)
        P_num = np.zeros((num_basis, L))
        
        for i, exponents in enumerate(basis_exponents):
            # evaluate each basis term at all sample points
            basis_values = np.ones(L)
            for var_idx, power in enumerate(exponents):
                if power > 0:
                    basis_values *= samples[:, var_idx] ** power
            P_num[i, :] = basis_values
        
        return P_num
    
    def _generate_exponent_combinations(self, n_vars, total_degree):
        """
        Generate all exponent combinations for a given number of variables and total degree
        
        Parameters:
            n_vars : number of variables
            total_degree : total degree
            
        Returns:
            Generator of tuples representing exponent combinations
        """
        if n_vars == 1:
            yield (total_degree,)
        elif total_degree == 0:
            yield tuple([0] * n_vars)
        else:
            for first_exp in range(total_degree + 1):
                remaining_degree = total_degree - first_exp
                for rest_combination in self._generate_exponent_combinations(n_vars - 1, remaining_degree):
                    yield tuple([first_exp] + list(rest_combination))


    def PDD_grlex_general(self, samples, m, S, poly_type_list):
        """
        Fast PDD basis construction without SymPy (supports mixed Legendre/Hermite)
        
        :param samples: (L x N) input samples
               - poly_type='legendre': normalized to [-1, 1]
               - poly_type='hermite': normalized to the standard normal N(0,1)
        :param m: maximum polynomial order
        :param S: maximum interaction order
        :param poly_type_list: 'hermite' or 'legendre', or a list of length N
                 e.g., ['legendre', 'legendre', 'hermite']
        :return: matrix A (L x M), multivariate basis in grlex order
        """
        from math import factorial
        
        L, N = samples.shape

        # convert poly_type to a per-variable list
        if isinstance(poly_type_list, str):
            poly_type_list = [poly_type_list] * N
        elif isinstance(poly_type_list, (list, tuple, np.ndarray)):
            poly_type_list = list(poly_type_list)
            if len(poly_type_list) != N:
                raise ValueError(f"poly_type_list length ({len(poly_type_list)}) must match number of variables N ({N})")
        else:
            raise ValueError(f"poly_type_list must be a string or a list-like of length N, got: {type(poly_type_list)}")

        eval_fns = []
        for p_type in poly_type_list:
            if p_type == 'hermite':
                eval_fns.append(hermeval)
            elif p_type == 'legendre':
                eval_fns.append(legval)
            else:
                raise ValueError(f"Unknown poly_type: {p_type}")
        
        # 1. Generate all basis exponent combinations (grlex order)
        basis_exponents = []
        
        # loop over the total degree from 0 to m
        for total_degree in range(m + 1):
            # all exponent combinations of N variables for the given degree
            for exp_combination in self._generate_exponent_combinations(N, total_degree):
                # interaction limit: number of variables with nonzero exponents <= S
                non_zero_vars = sum(1 for exp in exp_combination if exp > 0)
                if non_zero_vars <= S:
                    basis_exponents.append(list(exp_combination))
        
        # 2. Sort in graded lexicographic (grlex) order
        def grlex_key(exp_vec):
            total_deg = sum(exp_vec)
            # for equal degree, sort in reverse lexicographic order (from the last variable)
            lex_order = tuple(reversed(exp_vec))
            return (total_deg, lex_order)
        
        basis_exponents.sort(key=grlex_key)
        
        # 3. Build matrix A - evaluate and normalize the polynomials
        basis_num = len(basis_exponents)
        A = np.zeros((L, basis_num))
        
        for col, powers in enumerate(basis_exponents):
            term_vals = np.ones(L)
            scaling = 1.0
            for i in range(N):
                order = powers[i]
                if order > 0:
                    coeffs = [0] * order + [1]  # e.g., x^2 -> [0, 0, 1]
                    term_vals *= eval_fns[i](samples[:, i], coeffs)

                    # apply the normalization constant according to the poly_type of each variable
                    if poly_type_list[i] == 'legendre':
                        scaling *= np.sqrt(2 * order + 1)
                    elif poly_type_list[i] == 'hermite':
                        scaling /= np.sqrt(factorial(order))

            term_vals_normalized = term_vals * scaling
            A[:, col] = term_vals_normalized

        return A

    def PDD_grlex_general_fast(self, samples, m, S, poly_type_list, block_size=131072):
        """
        Faster PDD basis construction using caches

        - grlex exponent combinations are cached by (N, m, S)
        - 1D polynomial values of each variable are computed once per block
        - each basis term requires only products
        - a smaller block_size reduces memory but may increase the call overhead
        """
        from math import factorial

        samples = np.asarray(samples)
        L, N = samples.shape
        matrix_dtype = np.result_type(samples.dtype, np.float64)

        # convert poly_type to a per-variable list
        if isinstance(poly_type_list, str):
            poly_type_list = [poly_type_list] * N
        elif isinstance(poly_type_list, (list, tuple, np.ndarray)):
            poly_type_list = list(poly_type_list)
            if len(poly_type_list) != N:
                raise ValueError(f"poly_type_list length ({len(poly_type_list)}) must match number of variables N ({N})")
        else:
            raise ValueError(f"poly_type_list must be a string or a list-like of length N, got: {type(poly_type_list)}")

        if block_size is None or block_size <= 0:
            block_size = L

        # initialize the global/instance caches
        if not hasattr(self, "_pdd_fast_cache"):
            self._pdd_fast_cache = {
                "basis_exponents": {},
                "coeffs": {},
            }

        cache = self._pdd_fast_cache

        # 1) grlex basis exponent cache
        basis_key = (N, m, S)
        if basis_key not in cache["basis_exponents"]:
            basis_exponents = []
            for total_degree in range(m + 1):
                for exp_combination in self._generate_exponent_combinations(N, total_degree):
                    non_zero_vars = sum(1 for exp in exp_combination if exp > 0)
                    if non_zero_vars <= S:
                        basis_exponents.append(tuple(exp_combination))

            def grlex_key(exp_vec):
                total_deg = sum(exp_vec)
                lex_order = tuple(reversed(exp_vec))
                return (total_deg, lex_order)

            basis_exponents.sort(key=grlex_key)
            cache["basis_exponents"][basis_key] = tuple(basis_exponents)

        basis_exponents = cache["basis_exponents"][basis_key]

        # 2) polynomial coefficients cache: order -> [0, 0, ..., 1]
        coeff_key = m
        if coeff_key not in cache["coeffs"]:
            cache["coeffs"][coeff_key] = tuple(tuple([0] * order + [1]) for order in range(m + 1))
        coeffs_by_order = cache["coeffs"][coeff_key]

        # 3) products only for each basis term; samples are processed in blocks
        basis_num = len(basis_exponents)
        A = np.empty((L, basis_num), dtype=matrix_dtype)

        for start in range(0, L, block_size):
            end = min(start + block_size, L)
            block_len = end - start
            block_samples = samples[start:end]

            eval_tables = []
            norm_tables = []

            for i, p_type in enumerate(poly_type_list):
                x = block_samples[:, i]

                if p_type == "hermite":
                    eval_fn = hermeval
                    norm_fn = lambda order: 1.0 / np.sqrt(factorial(order))
                elif p_type == "legendre":
                    eval_fn = legval
                    norm_fn = lambda order: np.sqrt(2 * order + 1)
                else:
                    raise ValueError(f"Unknown poly_type: {p_type}")

                table = [np.ones(block_len, dtype=matrix_dtype)]
                norms = [1.0]
                for order in range(1, m + 1):
                    table.append(eval_fn(x, coeffs_by_order[order]))
                    norms.append(norm_fn(order))

                eval_tables.append(table)
                norm_tables.append(norms)

            for col, powers in enumerate(basis_exponents):
                term_vals = np.ones(block_len, dtype=matrix_dtype)
                scaling = 1.0

                for i, order in enumerate(powers):
                    if order > 0:
                        term_vals *= eval_tables[i][order]
                        scaling *= norm_tables[i][order]

                A[start:end, col] = term_vals * scaling

        return A

    
    def PDD_grlex_legendre(self, samples, m, S):
        """
        Fast PDD Legendre PCE basis in grlex order without SymPy
        
        :param samples: L x N input samples normalized to [-1, 1]
        :param m: maximum polynomial order
        :param S: maximum interaction order
        :return: matrix A (L x M), grlex index_set, M: number of basis functions
        """
        L, N = samples.shape
        
        # 1. Generate all basis exponent combinations (grlex order)
        basis_exponents = []
        
        # loop over the total degree from 0 to m
        for total_degree in range(m + 1):
            # all exponent combinations of N variables for the given degree
            for exp_combination in self._generate_exponent_combinations(N, total_degree):
                # interaction limit: number of variables with nonzero exponents <= S
                non_zero_vars = sum(1 for exp in exp_combination if exp > 0)
                if non_zero_vars <= S:
                    basis_exponents.append(list(exp_combination))
        
        # 2. Sort in graded lexicographic (grlex) order
        def grlex_key(exp_vec):
            total_deg = sum(exp_vec)
            # for equal degree, sort in reverse lexicographic order (from the last variable)
            lex_order = tuple(reversed(exp_vec))
            return (total_deg, lex_order)
        
        basis_exponents.sort(key=grlex_key)
        
        # 3. Build matrix A - evaluate the Legendre polynomials
        basis_num = len(basis_exponents)
        A = np.zeros((L, basis_num))
        
        for col, powers in enumerate(basis_exponents):
            term_vals = np.ones(L)
            for i in range(N):
                order = powers[i]
                if order > 0:
                    coeffs = [0] * order + [1]  # e.g., x^2 -> [0, 0, 1]
                    term_vals *= legval(samples[:, i], coeffs)
            
            # normalize by the basis norm
            scaling = 1.0
            for order in powers:
                scaling *= np.sqrt(2 * order + 1)
            
            term_vals_normalized = term_vals * scaling
            A[:, col] = term_vals_normalized
        
        return A
    
    def PDD_grlex_hermite(self, samples, m, S):
        """
        Fast PDD Hermite PCE basis in grlex order without SymPy (probabilists' Hermite, orthonormal version)
        
        :param samples: L x N input samples (each column assumed to be transformed to N(0,1))
        :param m: maximum total degree
        :param S: maximum support-set size (maximum number of variables appearing together)
        :return: matrix A (L x M) - each column is an orthonormal Hermite PCE basis function
        """
        from math import factorial
        
        L, N = samples.shape
        
        # 1. Generate all basis exponent combinations (grlex order)
        basis_exponents = []
        
        # loop over the total degree from 0 to m
        for total_degree in range(m + 1):
            # all exponent combinations of N variables for the given degree
            for exp_combination in self._generate_exponent_combinations(N, total_degree):
                # interaction limit: number of variables with nonzero exponents <= S
                non_zero_vars = sum(1 for exp in exp_combination if exp > 0)
                if non_zero_vars <= S:
                    basis_exponents.append(list(exp_combination))
        
        # 2. Sort in graded lexicographic (grlex) order
        def grlex_key(exp_vec):
            total_deg = sum(exp_vec)
            # for equal degree, sort in reverse lexicographic order (from the last variable)
            lex_order = tuple(reversed(exp_vec))
            return (total_deg, lex_order)
        
        basis_exponents.sort(key=grlex_key)
        
        # 3. Build matrix A - evaluate and normalize the Hermite polynomials
        basis_num = len(basis_exponents)
        A = np.zeros((L, basis_num))
        
        for col, powers in enumerate(basis_exponents):
            term_vals = np.ones(L)
            for i in range(N):
                order = powers[i]
                if order > 0:
                    # hermeval: coeffs = [c0, c1, ..., cn], sum c_j * He_j(x)
                    # for He_order(x) only, use [0]*order + [1]
                    coeffs = [0] * order + [1]
                    term_vals *= hermeval(samples[:, i], coeffs)
            
            # --------- analytic normalization (orthonormal with respect to N(0,1)) ---------
            # 1D: E[He_n(X)^2] = n!  (X ~ N(0,1))
            # ψ_n(x) = He_n(x) / sqrt(n!) → E[ψ_n^2] = 1
            #
            # multivariate: for prod He_{k_i}(X_i)
            # E[(∏ He_{k_i})^2] = ∏ k_i!
            # → scaling = 1 / sqrt(∏ k_i!)
            scaling = 1.0
            for order in powers:
                if order > 0:
                    scaling /= np.sqrt(factorial(order))  # factorial(0)=1 for order=0, so no effect
            
            term_vals_normalized = term_vals * scaling
            A[:, col] = term_vals_normalized
        
        return A
    
    def PDD_grlex_legendre_sympy(self, samples, m, S):
        """
        GPCE-style Legendre PCE basis in grlex order
        :param samples: L x N input samples normalized to [-1, 1]
        :param m: maximum polynomial order
        :param S: maximum interaction order
        :return: matrix A (L x M), grlex index_set, M: number of basis functions
        """
        L, N = samples.shape
        variables = symbols(f'x:{N}')
        monomials_full = sorted(itermonomials(variables, m), key=monomial_key('grlex', list(reversed(variables))))

        # condition: support-set size <= S
        def num_unique_vars(poly):
            return len(set([factor.as_base_exp()[0] for factor in poly.as_ordered_factors()]))

        monomials = [p for p in monomials_full if num_unique_vars(p) <= S]

        index_set = []
        for monomial in monomials:
            exponents = [monomial.as_poly(*variables).degree(v) for v in variables]
            index_set.append(tuple(exponents))

        basis_num = len(index_set)
        A = np.zeros((L, basis_num))
        for col, powers in enumerate(index_set):
            term_vals = np.ones(L)
            for i in range(N):
                order = powers[i]
                coeffs = [0]*order + [1]  # e.g., x^2 -> [0, 0, 1]
                term_vals *= legval(samples[:, i], coeffs)

            # normalize by the basis norm
            scaling = 1.0
            for order in powers:
                scaling *= np.sqrt(2 * order + 1)
            
            term_vals_normalized = term_vals * scaling
            A[:, col] = term_vals_normalized

        return A

    # PCE basis construction
    def PCE_grlex_general_sympy(self, samples, m, poly_type):
        """
        PCE basis construction (Legendre or Hermite)
        :param samples: (L x N) input samples normalized to the standard normal distribution
        :param m: maximum polynomial order
        :param poly_type: 'hermite' or 'legendre'
        :return: matrix A (L x M), multivariate basis exponents index_set in grlex order
        """
        ### normalization to be revised ###

        if poly_type == 'hermite':
            eval_fn = hermeval
        elif poly_type == 'legendre':
            eval_fn = legval
        else:
            raise ValueError(f"Unknown poly_type: {poly_type}")
        
        L, N = samples.shape
        variables = symbols(f'x:{N}')
        monomials = sorted(itermonomials(variables, m), key=monomial_key('grlex', list(reversed(variables))))

        index_set = []
        for monomial in monomials:
            exponents = [monomial.as_poly(*variables).degree(v) for v in variables]
            index_set.append(tuple(exponents))

        basis_num = len(index_set)
        A = np.zeros((L, basis_num))
        for col, powers in enumerate(index_set):
            term_vals = np.ones(L)
            for i in range(N):
                order = powers[i]
                coeffs = [0]*order + [1]  # e.g., x^2 -> [0, 0, 1]
                term_vals *= eval_fn(samples[:, i], coeffs)

            # normalize by the basis norm
            mean_sq = np.mean(term_vals ** 2)      # mean square
            term_vals_normalized = term_vals / np.sqrt(mean_sq)

            A[:, col] = term_vals_normalized

        return A
    
    def PCE_grlex_legendre_sympy(self, samples, m):
        """
        GPCE-style Legendre PCE basis in grlex order
        :param samples: L x N input samples normalized to [-1, 1]
        :param m: maximum polynomial order
        :param S: maximum interaction order
        :return: matrix A (L x M), grlex index_set, M: number of basis functions
        """

        L, N = samples.shape
        variables = symbols(f'x:{N}')
        monomials = sorted(itermonomials(variables, m), key=monomial_key('grlex', list(reversed(variables))))

        index_set = []
        for monomial in monomials:
            exponents = [monomial.as_poly(*variables).degree(v) for v in variables]
            index_set.append(tuple(exponents))

        basis_num = len(index_set)
        A = np.zeros((L, basis_num))
        for col, powers in enumerate(index_set):
            term_vals = np.ones(L)
            for i in range(N):
                order = powers[i]
                coeffs = [0]*order + [1]  # e.g., x^2 -> [0, 0, 1]
                term_vals *= legval(samples[:, i], coeffs)

            # normalize by the basis norm
            mean_sq = np.mean(term_vals ** 2)      # mean square
            term_vals_normalized = term_vals / np.sqrt(mean_sq)

            A[:, col] = term_vals_normalized

        return A
    
    def whitening_transform(self, monomial_matrix, sampling_num):
        """
        Construct the whitening transformation matrix W

        Parameters:
            monomial_matrix : (L x M) design matrix (P)
            sampling num : Sample Num L

        Returns:
            W : (M,M) Whitening Transform Matrix
        """

        PT = np.transpose(monomial_matrix) # (L * M)
        G = (monomial_matrix@PT)/sampling_num # (M * L) * (L * M)

        # eigenvalues of G
        eigvals = np.linalg.eigvals(G)

        # filter negative or near-zero eigenvalues (numerical error)
        negative_eigvals = eigvals[eigvals < -1e-10]  # -1e-10: threshold for numerical stability

        # output
        print("Negative eigenvalues:")
        print(negative_eigvals)
        print(f"G matrix: {len(negative_eigvals)} negative or non-definite eigenvalues.")

        # monomial_matrix: M rows (basis functions) x N columns (samples)
        # P^T = monomial_matrix^T: N rows (samples) x M columns (basis functions)
        # each entry of (monomial_matrix@PT) is a sum over all samples
        Q = np.linalg.cholesky(G)
        W = np.linalg.inv(Q)

        return W
    
    def lasso_dmorph_regression(self, A, b, num_iter=1000000, epsilon=1e-6, lam=0.5, verbose=False, tolerance = 1e-6): ## not verified
        """
        Implements the Lasso-based D-MORPH regression (Eq. 23, 25, 26)

        Parameters:
            A : (M x L) design matrix (P.T)
            b : (M,) response vector
            num_iter : number of recursive iterations
            epsilon : stability term for diagonal weights
            lam : lambda (0~1) weight in Eq. 23
            verbose : print per-iteration info

        Returns:
            c : (L,) final sparse coefficient vector
        """
        M, L = A.shape
        A_pinv = pinv(A)
        I = np.eye(L)
        Phi = I - A_pinv @ A

        # Step 1: Initial D-MORPH solution (Eq. 17)
        PhiD = Phi @ np.eye(L)
        E, S, F_T = svd(PhiD)
        F = F_T.T
        r = np.sum(S > np.max(S) * max(PhiD.shape) * np.finfo(S.dtype).eps)

        E_lr = E[:, - (L - r):]
        F_lr = F[:, - (L - r):]
        c_prev = F_lr @ inv(E_lr.T @ F_lr) @ E_lr.T @ A_pinv @ b
        a0 = c_prev.copy()

        for i in range(num_iter):
            # Step 2: Diagonal weight matrix W (Eq. 20)
            W_diag = np.zeros(L)
            W_diag[1:] = 1 / (np.abs(c_prev[1:]) + epsilon)  # skip j=0
            W = np.diag(W_diag)

            # Step 3: SVD of ΦW (Eq. 24)
            PhiW = Phi @ W
            E_w, T_w, Fw_T = svd(PhiW)
            F_w = Fw_T.T
            r_w = np.sum(T_w > np.max(T_w) * max(PhiW.shape) * np.finfo(T_w.dtype).eps)

            E1 = E_w[:, :r_w]
            Fr = F_w[:, :r_w]
            Elr = E_w[:, - (L - r_w):]
            Flr = F_w[:, - (L - r_w):]

            # Step 4: Eq. 23 - updated coefficient
            part1 = Flr @ inv(Elr.T @ Flr) @ Elr.T @ A_pinv @ b
            part2 = Flr @ inv(Elr.T @ Flr) @ Elr.T @ E1 @ inv(np.diag(T_w[:r_w])) @ Fr.T @ W @ (lam * a0 + (1 - lam) * c_prev)
            c_new = part1 - part2


            delta = np.linalg.norm(c_new - c_prev)
            if delta < tolerance:
                if verbose:
                    print(f"Iter {i+1}: ||Δc|| = {np.linalg.norm(c_new - c_prev):.2e}")
                break

            # Update
            c_prev = c_new.copy()

        return c_new
    
    def dmorph_closed_form(self, A, b, D=None, r=None): ## not verified
        """
        D-MORPH closed-form coefficient computation based on equation (17)

        Parameters:
            A : (M x L) Monomial basis matrix (P.T)
            b : (M,) Response vector
            D : (L x L) Symmetric, PSD diagonal matrix (optional; defaults to identity)
            r : Rank of A (optional; if None, estimated from SVD)

        Returns:
            c : (L,) Coefficient vector
        """
        M, L = A.shape

        # Step 1: Moore-Penrose pseudo-inverse
        A_pinv = pinv(A)

        # Step 2: Default D if not given
        if D is None:
            D = np.eye(L)

        # Step 3: ΦD = (I - A⁺A) @ D
        I = np.eye(L)
        Phi = I - A_pinv @ A
        PhiD = Phi @ D

        # Step 4: SVD of ΦD
        E, T, F_T = svd(PhiD)
        F = F_T.T

        # Step 5: Estimate rank if not given
        if r is None:
            tol = np.max(T) * max(PhiD.shape) * np.finfo(T.dtype).eps
            r = np.sum(T > tol)

        # Step 6: Use last (L - r) columns of E, F
        E_lr = E[:, - (L - r):]
        F_lr = F[:, - (L - r):]

        # Step 7: Final coefficient computation (equation 17)
        middle = inv(E_lr.T @ F_lr)
        c = F_lr @ middle @ E_lr.T @ A_pinv @ b

        return c





    def estimate_lasso_coefficients(
        self,
        A,
        B,
        cv=5,
        n_alphas=100,
        alpha_min_ratio=1e-5,
        random_state=42,
    ):
        """
        Select the optimal alpha by K-fold cross-validation and
        estimate the Lasso GPCE coefficients.

        Parameters
        ----------
        A : ndarray, shape (n_samples, n_features)
            GPCE design matrix.

            The first column is assumed to be the constant basis.
            e.g., A[:, 0] = 1

        B : ndarray
            Single output:
                shape (n_samples,)

            Multiple outputs:
                shape (n_samples, n_outputs)

        cv : int, default=5
            Number of folds for K-fold cross-validation.

        n_alphas : int, default=100
            Number of alpha candidates.

        alpha_min_ratio : float, default=1e-5
            Ratio alpha_min / alpha_max.

        random_state : int, default=42
            Seed for reproducible K-fold splits.

        Returns
        -------
        C : ndarray
            Single output:
                shape (n_features,)

            Multiple outputs:
                shape (n_features, n_outputs)

        Notes
        -----
        The selected alpha is stored in

            self.lasso_alpha_

        The number of selected nonzero non-constant coefficients is stored in

            self.lasso_n_active_
        """

        # ---------------------------------------------------------
        # 1. Validate the input arrays and sizes
        # ---------------------------------------------------------
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        if A.ndim != 2:
            raise ValueError(
                f"A must be a 2D array. Current shape={A.shape}"
            )

        if B.ndim not in (1, 2):
            raise ValueError(
                "B must be a 1D or 2D array. "
                f"Current shape={B.shape}"
            )

        if A.shape[0] != B.shape[0]:
            raise ValueError(
                "The numbers of samples in A and B do not match: "
                f"A.shape[0]={A.shape[0]}, "
                f"B.shape[0]={B.shape[0]}"
            )

        if not np.all(np.isfinite(A)):
            raise ValueError("A contains NaN or Inf.")

        if not np.all(np.isfinite(B)):
            raise ValueError("B contains NaN or Inf.")

        if n_alphas < 2:
            raise ValueError("n_alphas must be at least 2.")

        if not 0.0 < alpha_min_ratio < 1.0:
            raise ValueError(
                "alpha_min_ratio must be between 0 and 1."
            )

        n_samples, n_features = A.shape

        # ---------------------------------------------------------
        # 2. Check that the first column is the constant GPCE basis
        # ---------------------------------------------------------
        constant_value = A[0, 0]

        if np.isclose(constant_value, 0.0):
            raise ValueError(
                "The constant basis value in the first column of A cannot be zero."
            )

        if not np.allclose(
            A[:, 0],
            constant_value,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "The first column of A is not a constant basis. "
                "This function assumes a GPCE matrix whose first column "
                "A[:, 0] is constant."
            )

        # the constant basis is excluded from the L1 penalty
        A_nonconstant = A[:, 1:]

        # ---------------------------------------------------------
        # 3. K-fold splitter
        # ---------------------------------------------------------
        n_splits = min(cv, n_samples)

        if n_splits < 2:
            raise ValueError(
                "At least two samples are required for cross-validation."
            )

        cv_splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

        # ---------------------------------------------------------
        # 4. Internal fit for a single output
        # ---------------------------------------------------------
        def fit_single_output(y):
            y = np.asarray(y, dtype=float)

            # special case with only the constant term
            if n_features == 1:
                coefficients = np.array(
                    [np.mean(y) / constant_value],
                    dtype=float,
                )

                return coefficients, 0.0, 0

            # center the data to match fit_intercept=True, then
            # compute alpha_max
            A_centered = (
                A_nonconstant
                - np.mean(A_nonconstant, axis=0)
            )
            y_centered = y - np.mean(y)

            alpha_max = (
                np.max(
                    np.abs(A_centered.T @ y_centered)
                )
                / n_samples
            )

            # the output is almost constant
            if alpha_max <= np.finfo(float).eps:
                coefficients = np.zeros(
                    n_features,
                    dtype=float,
                )

                coefficients[0] = (
                    np.mean(y) / constant_value
                )

                return coefficients, 0.0, 0

            # generated from large to small alpha
            alpha_grid = alpha_max * np.logspace(
                0.0,
                np.log10(alpha_min_ratio),
                n_alphas,
            )

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    ConvergenceWarning,
                )

                model = LassoCV(
                    alphas=alpha_grid,
                    cv=cv_splitter,
                    fit_intercept=True,
                    max_iter=100_000,
                    tol=1e-8,
                    n_jobs=-1,
                    selection="cyclic",
                )

                model.fit(A_nonconstant, y)

            coefficients = np.zeros(
                n_features,
                dtype=float,
            )

            # model.intercept_ is the constant function value, so
            # divide by the constant basis value if it is not 1
            coefficients[0] = (
                model.intercept_ / constant_value
            )

            coefficients[1:] = model.coef_

            n_active = int(
                np.count_nonzero(
                    np.abs(model.coef_) > 1e-12
                )
            )

            return (
                coefficients,
                float(model.alpha_),
                n_active,
            )

        # ---------------------------------------------------------
        # 5. Single output
        # ---------------------------------------------------------
        if B.ndim == 1:
            C, selected_alpha, n_active = (
                fit_single_output(B)
            )

            # store the selection in class attributes
            self.lasso_alpha_ = selected_alpha
            self.lasso_n_active_ = n_active

            return C

        # ---------------------------------------------------------
        # 6. Multiple outputs
        # select an independent alpha for each output
        # ---------------------------------------------------------
        n_outputs = B.shape[1]

        C = np.zeros(
            (n_features, n_outputs),
            dtype=float,
        )

        selected_alphas = np.zeros(
            n_outputs,
            dtype=float,
        )

        n_active_coefficients = np.zeros(
            n_outputs,
            dtype=int,
        )

        for output_index in range(n_outputs):
            (
                coefficients,
                selected_alpha,
                n_active,
            ) = fit_single_output(
                B[:, output_index]
            )

            C[:, output_index] = coefficients
            selected_alphas[output_index] = (
                selected_alpha
            )
            n_active_coefficients[output_index] = (
                n_active
            )

        self.lasso_alpha_ = selected_alphas
        self.lasso_n_active_ = (
            n_active_coefficients
        )

        return C

    def estimate_ridge_coefficients(
        self,
        A,
        B,
        cv=5,
        alphas=None,
        random_state=42,
    ):
        """
        Select the optimal alpha by K-fold cross-validation and
        estimate the Ridge GPCE coefficients.

        Parameters
        ----------
        A : ndarray, shape (n_samples, n_features)
            GPCE design matrix.

            The first column is assumed to be the constant basis.
            e.g.,
                A[:, 0] = 1

        B : ndarray
            Single output:
                shape (n_samples,)

            Multiple outputs:
                shape (n_samples, n_outputs)

        cv : int, default=5
            Number of folds for K-fold cross-validation.

        alphas : array-like or None
            Candidate Ridge regularization parameters.

            If None, the following range is used.

                10^(-12) ~ 10^4

        random_state : int, default=42
            Random seed for reproducible K-fold splits.

        Returns
        -------
        C : ndarray
            Single output:
                shape (n_features,)

            Multiple outputs:
                shape (n_features, n_outputs)

        Notes
        -----
        The selected alpha is stored in

            self.ridge_alpha_

        The cross-validation score of each output is stored in

            self.ridge_cv_score_
        """

        # ---------------------------------------------------------
        # 1. Convert and validate the input arrays
        # ---------------------------------------------------------
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        if A.ndim != 2:
            raise ValueError(
                "A must be a 2D array. "
                f"Current shape={A.shape}"
            )

        if B.ndim not in (1, 2):
            raise ValueError(
                "B must be a 1D or 2D array. "
                f"Current shape={B.shape}"
            )

        if A.shape[0] != B.shape[0]:
            raise ValueError(
                "The numbers of samples in A and B do not match: "
                f"A.shape[0]={A.shape[0]}, "
                f"B.shape[0]={B.shape[0]}"
            )

        if not np.all(np.isfinite(A)):
            raise ValueError(
                "A contains NaN or Inf."
            )

        if not np.all(np.isfinite(B)):
            raise ValueError(
                "B contains NaN or Inf."
            )

        n_samples, n_features = A.shape

        # ---------------------------------------------------------
        # 2. Alpha candidates
        # ---------------------------------------------------------
        if alphas is None:
            alphas = np.logspace(
                -12,
                4,
                161,
            )
        else:
            alphas = np.asarray(
                alphas,
                dtype=float,
            )

        if alphas.ndim != 1:
            raise ValueError(
                "alphas must be a 1D array."
            )

        if len(alphas) < 2:
            raise ValueError(
                "alphas must contain at least two candidates."
            )

        if not np.all(np.isfinite(alphas)):
            raise ValueError(
                "alphas contains NaN or Inf."
            )

        if np.any(alphas <= 0.0):
            raise ValueError(
                "All alpha candidates for RidgeCV must be positive."
            )

        # sort in ascending order
        alphas = np.sort(np.unique(alphas))

        # ---------------------------------------------------------
        # 3. Check that the first column is the constant basis
        # ---------------------------------------------------------
        constant_value = A[0, 0]

        if np.isclose(constant_value, 0.0):
            raise ValueError(
                "The constant basis value in the first column cannot be zero."
            )

        if not np.allclose(
            A[:, 0],
            constant_value,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "The first column of A is not a constant basis. "
                "This function assumes a GPCE matrix whose first column "
                "A[:, 0] is constant."
            )

        # Exclude the first (constant) basis.
        # The mean term is estimated by the RidgeCV intercept,
        # so that no L2 penalty is applied to the constant coefficient.
        A_nonconstant = A[:, 1:]

        # ---------------------------------------------------------
        # 4. K-fold settings
        # ---------------------------------------------------------
        n_splits = min(cv, n_samples)

        if n_splits < 2:
            raise ValueError(
                "At least two samples are required for cross-validation."
            )

        cv_splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

        # ---------------------------------------------------------
        # 5. Internal fit function for a single output
        # ---------------------------------------------------------
        def fit_single_output(y):
            y = np.asarray(y, dtype=float)

            # special case with only the constant basis
            if n_features == 1:
                coefficients = np.array(
                    [
                        np.mean(y)
                        / constant_value
                    ],
                    dtype=float,
                )

                return coefficients, 0.0, np.nan

            model = RidgeCV(
                alphas=alphas,
                fit_intercept=True,
                cv=cv_splitter,
                scoring="neg_mean_squared_error",
            )

            model.fit(
                A_nonconstant,
                y,
            )

            coefficients = np.zeros(
                n_features,
                dtype=float,
            )

            # the intercept is the constant term of the function value,
            # so the coefficient is obtained by dividing by the constant basis value.
            coefficients[0] = (
                model.intercept_
                / constant_value
            )

            coefficients[1:] = (
                model.coef_
            )

            selected_alpha = float(
                model.alpha_
            )

            cv_score = float(
                model.best_score_
            )

            return (
                coefficients,
                selected_alpha,
                cv_score,
            )

        # ---------------------------------------------------------
        # 6. Single output
        # ---------------------------------------------------------
        if B.ndim == 1:
            (
                C,
                selected_alpha,
                cv_score,
            ) = fit_single_output(B)

            self.ridge_alpha_ = (
                selected_alpha
            )

            self.ridge_cv_score_ = (
                cv_score
            )

            return C

        # ---------------------------------------------------------
        # 7. Multiple outputs
        # select an independent alpha for each output.
        # ---------------------------------------------------------
        n_outputs = B.shape[1]

        C = np.zeros(
            (n_features, n_outputs),
            dtype=float,
        )

        selected_alphas = np.zeros(
            n_outputs,
            dtype=float,
        )

        cv_scores = np.zeros(
            n_outputs,
            dtype=float,
        )

        for output_index in range(n_outputs):
            (
                coefficients,
                selected_alpha,
                cv_score,
            ) = fit_single_output(
                B[:, output_index]
            )

            C[:, output_index] = (
                coefficients
            )

            selected_alphas[output_index] = (
                selected_alpha
            )

            cv_scores[output_index] = (
                cv_score
            )

        self.ridge_alpha_ = (
            selected_alphas
        )

        self.ridge_cv_score_ = (
            cv_scores
        )

        return C

    def define_monomial(self, samples, m):
        """
        Sample-based symbolic basis for sensitivity analysis of PCE and GPCE models

        Parameters:
            samples : (L x N) matrix
            m : maximum polynomial order

        Returns:
            variables_from_P : Monomial Basis
        """
        count = np.shape(samples)[1]  # number of variables
        variables = symbols(f'x:{count}')  # x0, x1, ..., xN

        # all monomials up to order m in grlex order
        P = sorted(itermonomials(variables, m), key=monomial_key('grlex', list(reversed(variables))))

        # list of variables in each monomial
        variables_from_P = [[factor.as_base_exp()[0] for factor in term.as_ordered_factors()] for term in P]

        return variables_from_P

    def define_monomial_DD(self, samples, m, S):
        """
        Sample-based symbolic basis for sensitivity analysis of PDD and DD-GPCE models

        Parameters:
            samples : (L x N) matrix
            m : maximum polynomial order
            S : maximum interaction order

        Returns:
            variables_from_P : Monomial Basis
        """
        count = np.shape(samples)[1]  # number of variables (columns of the input data)
        variables = symbols(f'x:{count}')  # create the variables x0, x1, x2, ...
        P_full = sorted(itermonomials(variables, m), key=monomial_key('grlex', list(reversed(variables))))  # monomial basis
        # grlex: graded lexicographic order; sorted by total degree, then lexicographically within the same degree

        var_dict = dict(zip(variables, samples.T))
        # map each variable (x0, x1, ..., x9) to its list of sample values,
        # e.g., three samples: (x0, [1, 11, 21]), (x1, [2,12,22]), (x3, [3,13,23])...

        # condition: support-set size <= S
        def num_unique_vars(poly):
            return len(set([factor.as_base_exp()[0] for factor in poly.as_ordered_factors()]))

        P = [p for p in P_full if num_unique_vars(p) <= S]
        #print(len(P), P)

        # extract the variables only, x0^2*x1^3 -> [x0, x1]
        variables_from_P = [[factor.as_base_exp()[0] for factor in term.as_ordered_factors()] for term in P]

        return variables_from_P # return the monomial basis, N rows (input variables) x L columns (samples)

    # Sensitivity analysis
    def compute_sensitivity(self, C3, num_vars, variables_from_P):
        """
        First-order, second-order, and total-order sensitivity indices
        :param C3: list of GPCE coefficients (excluding the first coefficient C3[0])
        :param num_vars: number of variables
        :param variables_from_P: list of variables in each term
        :return: first-order, second-order (matrix), and total-order sensitivity indices
        """
        total_variance = np.sum(C3[1:] ** 2)
        
        first_order_sensitivity = np.zeros(num_vars)
        total_order_sensitivity = np.zeros(num_vars)
        second_order_sensitivity = np.zeros((num_vars, num_vars))

        for i in range(1, len(C3)):
            term = variables_from_P[i]
            term_variance = C3[i] ** 2

            for var in range(num_vars):
                x_var = symbols(f'x{var}')
                if isinstance(term, list):
                    if len(term) == 1 and term[0] == x_var:
                        first_order_sensitivity[var] += term_variance
                    if x_var in term:
                        total_order_sensitivity[var] += term_variance
                else:
                    if term == x_var:
                        first_order_sensitivity[var] += term_variance
                    if term == x_var or x_var in Mul.make_args(term):
                        total_order_sensitivity[var] += term_variance

            # second-order interaction sensitivity
            if isinstance(term, list) and len(term) == 2:
                indices = [int(str(t)[1:]) for t in term if str(t).startswith('x')]
                if len(indices) == 2:
                    i1, i2 = indices
                    second_order_sensitivity[i1, i2] += term_variance
                    second_order_sensitivity[i2, i1] += term_variance  # symmetry

        first_order_sensitivity /= total_variance
        total_order_sensitivity /= total_variance
        second_order_sensitivity /= total_variance

        return first_order_sensitivity, second_order_sensitivity, total_order_sensitivity

    def compute_variable_pair_sensitivity(self, C3, variables_from_P):
        """
        Sensitivity of each variable combination of the terms from the GPCE coefficients
        :param C3: GPCE coefficients (numpy array), C3[0] is the constant term
        :param variables_from_P: list of variables in each basis term
        :return: dictionary, variable combination (tuple of str) -> contribution (normalized variance)
        """

        sensitivity_map = defaultdict(float)
        total_variance = np.sum(C3[1:] ** 2)

        for coeff, vars_in_term in zip(C3[1:], variables_from_P[1:]):  # exclude the constant term
            # variable names as strings
            if isinstance(vars_in_term, list):
                key = tuple(sorted([str(v) for v in vars_in_term]))
            else:
                key = (str(vars_in_term),)

            sensitivity_map[key] += coeff**2

        # normalize the variance
        for key in sensitivity_map:
            sensitivity_map[key] /= total_variance

        return sensitivity_map

    def save_basis_sensitivities_to_csv(self, coeff_matrix, variables_from_P, output_dir, base_filename):
        """
        Save the basis-wise sensitivities to a csv file

        :coeff_matrix: C Matrix
        :variables_from_P: symbolic basis
        :output_dir: output file path
        :base_filename: file name
        """
        os.makedirs(output_dir, exist_ok=True)
        num_outputs = coeff_matrix.shape[1]

        for i in range(num_outputs):
            sensitivities = self.compute_variable_pair_sensitivity(coeff_matrix[:, i], variables_from_P)

            filepath = os.path.join(output_dir, f"{base_filename}_{i+1}.csv")
            with open(filepath, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Monomial", "Sensitivity"])
                for basis_vars, sens in sensitivities.items():
                    # increase the variable indices by one, x0->x1, x1->x2
                    adjusted_vars = []
                    for var in basis_vars:
                        if var.startswith('x'):
                            # x0 -> x1, x1 -> x2, ...
                            var_num = int(var[1:]) + 1
                            adjusted_vars.append(f'x{var_num}')
                        else:
                            adjusted_vars.append(var)
                    # tuple to string: ('x1', 'x2') -> "x1_x2"
                    key_str = "_".join(adjusted_vars)
                    writer.writerow([key_str, sens])
                    
    def PCE(self, pce_coefficient,informative_matrix):
        """
        Compute Y_hat (surrogate output) for each sample

        Parameters:
            pce_coefficient : C Matrix
            informative_matrix : A Matrix

        Returns:
            
            Y_hat : Y_hat
        """
        
        #Y_hat=np.sum(informative_matrix*pce_coefficient,axis=1)
        Y_hat = informative_matrix @ pce_coefficient

        return Y_hat

    def calculate_basis_num(self, N,m,S):
        """
        Compute the number of basis functions

        Parameters:
            N : number of variables
            m : maximum polynomial order
            S : maximum interaction order

        Returns:
            basis : number of basis functions
        """

        basis = 1

        for s in range(1,S+1):
            basis = basis + mt.comb(N, s)*mt.comb(m, s)

        return basis


    def plot_prediction_performance(self, Y_true, Y_pred, label="Model", color="#FF7F00", title="Prediction vs True Output"):
        """
        Plot the predictions against the true values and print the goodness-of-fit metrics (Y vs Y_hat)

        :param Y_true: true values (numpy array)
        :param Y_pred: predicted values (numpy array)
        :param label: model name in the legend
        :param color: color of the scatter plot
        :param title: plot title
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(Y_pred, Y_true, '.', label=label, color=color)
        ax.plot(Y_true, Y_true, 'k--', label='y = x')

        ax.set_xlabel(r'$\hat Y$', fontsize=14)
        ax.set_ylabel(r'$Y$', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # goodness-of-fit metrics
        r2 = r2_score(Y_true, Y_pred)
        rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
        nrmse = rmse / (Y_true.max() - Y_true.min())

        print(f"R²: {r2:.5f}")
        print(f"RMSE: {rmse:.5f}")
        print(f"NRMSE: {nrmse:.5f}")

    def whitening_transform_revise(self, NN, m, S, std_vec=None, rho=None, mean_vec=None, ):
        """
        Whitening transformation matrix for a general dimension by Gauss quadrature
        
        Parameters:
            NN: number of variables
            m: maximum polynomial order
            S: interaction order limit
            std_vec: standard deviation of each variable (default: all 0.4)
            mean_vec: mean of each variable (default: all 0)
            corr_matrix: correlation matrix (default: 1 on the diagonal, 0.4 elsewhere)
            
        Returns:
            ORN: whitening matrix (orthonormalization matrix)
        """

        from numpy.polynomial.hermite import hermgauss
        from itertools import combinations
        import math
        
        def GaussHermite_2(n):
            x, w = np.polynomial.hermite.hermgauss(n) # nodes and weights for Gauss-Hermite quadrature
                                                    # x: 1D array of length n with the nodes; w: weights at the nodes
            return x, w
        
        def nGaussDen(x, mu, cov): # PDF of the N-dimensional normal distribution
            if isinstance(x, (int, float)):
                mahDist = (x - mu) / cov * (x - mu)
                num = np.exp((-1 / 2) * mahDist)
                denom = np.sqrt(2 * np.pi**1 * cov)
                output = num / denom
            else:
                N = len(x)
                mahDist = np.dot(np.dot((x - mu), np.linalg.inv(cov)), (x - mu))
                num = np.exp((-1 / 2) * mahDist)
                detCov = np.linalg.det(cov)
                denom = np.sqrt((2 * np.pi) ** N *detCov)
                output = num / denom
            return output
        
        def calculate_basis_num(N,m,S):
            """Compute the number of basis functions"""
            basis = 1
            for s in range(1,S+1):
                basis = basis + mt.comb(N, s)*mt.comb(m, s)
            return basis

        def generate_monomial_orders(NN, mm, SS):
            """Monomial exponent matrix in grlex order"""
            from sympy import symbols
            from sympy.polys.monomials import itermonomials
            from sympy.polys.orderings import monomial_key
            
            variables = symbols(f'x:{NN}')
            monomials_full = sorted(itermonomials(variables, mm), key=monomial_key('grlex', list(reversed(variables))))
            
            # apply the SS limit - interaction order
            def num_unique_vars(poly):
                return len(set([factor.as_base_exp()[0] for factor in poly.as_ordered_factors()]))
            
            monomials = [p for p in monomials_full if num_unique_vars(p) <= SS]
            
            # exponent matrix
            J = []
            for monomial in monomials:
                exponents = [monomial.as_poly(*variables).degree(v) for v in variables]
                J.append(exponents)
            
            return np.array(J)

        # defaults
        if std_vec is None:
            std_vec = np.ones(NN) * 1  # default standard deviation
        else:
            std_vec = np.array(std_vec)
            
        if mean_vec is None:
            mean_vec = np.zeros(NN)  # default mean
        else:
            mean_vec = np.array(mean_vec)
            
        if rho is None:
            # default correlation matrix (1 on the diagonal, 0.4 elsewhere)
            corr_matrix = np.eye(NN)
            #corr_matrix = np.eye(NN) + rho * (np.ones((NN, NN)) - np.eye(NN))
        else:
            corr_matrix = np.eye(NN) + rho * (np.ones((NN, NN)) - np.eye(NN))

        nA = calculate_basis_num(NN, m, S) # number of monomial basis functions

        # monomial exponent matrix (grlex order)
        J = generate_monomial_orders(NN, m, S)
        # print(f"J shape: {J.shape}")
        # print(f"First few rows of J:\n{J[:min(10, len(J))]}")

        # Generate Gauss-Hermite quadrature points and weights
        nGauss = math.ceil((m+1)/2) + 10  # number of nodes for the required integration accuracy
        [nodes, weights] = GaussHermite_2(nGauss)   # node vector and weights for the given number of nodes
        
        # scale the nodes and weights for each variable
        scaled_nodes = np.zeros((nGauss, NN))
        scaled_weights = np.zeros((nGauss, NN))
        
        for i in range(NN):
            scaled_nodes[:, i] = math.sqrt(2) * std_vec[i] * nodes + mean_vec[i] # nodes for the normal distribution
            scaled_weights[:, i] = math.sqrt(1/math.pi) * weights # scale the weights

        GRM = np.zeros((nA, nA))
        for i in range(nA):
            for j in range(i, nA):
                combined_orders = J[i,:] + J[j,:]
                nonzero_vars = np.where(combined_orders != 0)[0]
                n_nonzero = len(nonzero_vars)
                tmp = 0
                
                if n_nonzero == 0: # 0th order * 0th order -> 1
                    tmp = 1

                elif n_nonzero == 1: # only one variable -> 1D integral
                    var_idx = nonzero_vars[0]
                    for k in range(nGauss):
                        tmp += (scaled_nodes[k, var_idx] ** combined_orders[var_idx]) * scaled_weights[k, var_idx]

                elif n_nonzero == 2: # two variables -> 2D integral
                    var1, var2 = nonzero_vars[0], nonzero_vars[1]
                    sub_mean = np.array([mean_vec[var1], mean_vec[var2]])
                    sub_cov = np.array([
                        [std_vec[var1]**2, corr_matrix[var1, var2] * std_vec[var1] * std_vec[var2]],
                        [corr_matrix[var1, var2] * std_vec[var1] * std_vec[var2], std_vec[var2]**2]
                    ])
                    
                    for k1 in range(nGauss):
                        for k2 in range(nGauss):
                            biX = np.array([scaled_nodes[k1, var1], scaled_nodes[k2, var2]])
                            
                            # ratio of the bivariate to univariate densities (importance correction)
                            joint_pdf = nGaussDen(biX, sub_mean, sub_cov)
                            marginal1 = nGaussDen(scaled_nodes[k1, var1], mean_vec[var1], std_vec[var1]**2)
                            marginal2 = nGaussDen(scaled_nodes[k2, var2], mean_vec[var2], std_vec[var2]**2)
                            biTrans = joint_pdf / (marginal1 * marginal2)

                            tmp += ((scaled_nodes[k1, var1] ** combined_orders[var1]) * 
                                   (scaled_nodes[k2, var2] ** combined_orders[var2])) * biTrans * \
                                   scaled_weights[k1, var1] * scaled_weights[k2, var2]
                
                elif n_nonzero == 3: # three variables -> 3D integral
                    var1, var2, var3 = nonzero_vars[0], nonzero_vars[1], nonzero_vars[2]
                    sub_mean = np.array([mean_vec[var1], mean_vec[var2], mean_vec[var3]])
                    sub_cov = np.zeros((3, 3))
                    
                    # 3x3 covariance matrix
                    sub_cov[0, 0] = std_vec[var1]**2
                    sub_cov[1, 1] = std_vec[var2]**2
                    sub_cov[2, 2] = std_vec[var3]**2
                    sub_cov[0, 1] = sub_cov[1, 0] = corr_matrix[var1, var2] * std_vec[var1] * std_vec[var2]
                    sub_cov[0, 2] = sub_cov[2, 0] = corr_matrix[var1, var3] * std_vec[var1] * std_vec[var3]
                    sub_cov[1, 2] = sub_cov[2, 1] = corr_matrix[var2, var3] * std_vec[var2] * std_vec[var3]
                    
                    for k1 in range(nGauss):
                        for k2 in range(nGauss):
                            for k3 in range(nGauss):
                                triX = np.array([scaled_nodes[k1, var1], scaled_nodes[k2, var2], scaled_nodes[k3, var3]])
                                
                                # ratio of the trivariate to univariate densities (importance correction)
                                joint_pdf = nGaussDen(triX, sub_mean, sub_cov)
                                marginal1 = nGaussDen(scaled_nodes[k1, var1], mean_vec[var1], std_vec[var1]**2)
                                marginal2 = nGaussDen(scaled_nodes[k2, var2], mean_vec[var2], std_vec[var2]**2)
                                marginal3 = nGaussDen(scaled_nodes[k3, var3], mean_vec[var3], std_vec[var3]**2)
                                triTrans = joint_pdf / (marginal1 * marginal2 * marginal3)

                                tmp += ((scaled_nodes[k1, var1] ** combined_orders[var1]) * 
                                       (scaled_nodes[k2, var2] ** combined_orders[var2]) *
                                       (scaled_nodes[k3, var3] ** combined_orders[var3])) * triTrans * \
                                       scaled_weights[k1, var1] * scaled_weights[k2, var2] * scaled_weights[k3, var3]
                
                else: # four or more variables - approximate treatment (does not occur for S=3)
                    # approximated by a product of 1D integrals assuming independence
                    tmp = 1
                    for var_idx in nonzero_vars:
                        var_integral = 0
                        for k in range(nGauss):
                            var_integral += (scaled_nodes[k, var_idx] ** combined_orders[var_idx]) * scaled_weights[k, var_idx]
                        tmp *= var_integral
                    
                    # correlation correction factor (simple approximation)
                    if n_nonzero > 3:
                        # correction using the average correlation coefficient
                        avg_corr = 0
                        count = 0
                        for ii in range(len(nonzero_vars)):
                            for jj in range(ii+1, len(nonzero_vars)):
                                avg_corr += abs(corr_matrix[nonzero_vars[ii], nonzero_vars[jj]])
                                count += 1
                        if count > 0:
                            avg_corr /= count
                            tmp *= (1 + avg_corr * (n_nonzero - 3) * 0.1)  # simple correction

                # remove small values for numerical stability
                if abs(tmp) < 1E-15:
                    tmp = 0
                    
                GRM[i, j] = tmp

        # symmetric matrix: fill the rest of GRM by copying the upper triangle
        for iRow in range(nA):
            for iCol in range(iRow + 1, nA):
                GRM[iCol, iRow] = GRM[iRow, iCol]

        # eigenvalue problem
        # eigenvalues and eigenvectors
        D, V = np.linalg.eig(GRM)   #D: eigenvalue, V: eigenvector

        # remove non-positive eigenvalues (replace negative or zero eigenvalues with 1e-13)
        D[D <= 0] = 1E-13
        
        # reconstruct the modified Gram matrix
        GRM_stable = V @ np.diag(D) @ V.T

        # Cholesky decomposition
        try:
            Q = np.linalg.cholesky(GRM_stable)
        except np.linalg.LinAlgError:
            print("Warning: Cholesky decomposition failed. Using SVD-based approach.")
            U, s, Vt = svd(GRM_stable)
            s_sqrt = np.sqrt(np.maximum(s, 1e-13))
            Q = U @ np.diag(s_sqrt)

        # whitening matrix
        ORN = np.linalg.inv(Q)
    
        return ORN
