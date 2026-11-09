# Multiple Constraint Functions Manager
# Classes for managing multiple constraint functions

import numpy as np
import pickle
import math
from scipy.stats import multivariate_normal
import globalVars
import os
import sys

############################# Constraint Functions Parameters ########################################################

# REDLAB_UQ.py (DD-GPCE basis construction) is located in the project root
from REDLAB_UQ import SurrogateModel

# RBDO: target failure probability
target_failure_prob = 1.35e-3  # target failure probability, Phi(-3)

######################################################################################################################



class ConstraintFunction:
    """
    Class for an individual constraint function
    """
    # Uses the shared data of ObjectiveFunction (same GenMat file)
    
    def __init__(self, constraint_id, config, pickle_file_prefix="constraint", genmat_file="GenMat.pickle", 
                 rdo_constraint_func=None):
        self.constraint_id = constraint_id
        self.config = config
        self.pickle_file = globalVars.filepath + f"/data/{pickle_file_prefix}_{constraint_id}.pickle"
        self.genmat_file = genmat_file  # GenMat file name
        self.model = SurrogateModel()
        self.call_count = 0
        self.iter_count = 0
        self.CFNS = None  # score-function GPCE coefficients
        
        # RBDO parameters
        self.target_failure_prob = target_failure_prob  # refers to the module-level variable
        
        # Function for the RDO constraint (can be injected externally)
        self.rdo_constraint_func = rdo_constraint_func
        
        # Basis data (shared data of ObjectiveFunction)
        self.train_basis_con_resp = None
        self.W_CON_RESP = None
        self.train_basis_con_score = None
        self.W_CON_SCORE = None
        
        # Computed results (used instead of pickle files)
        self.TRON = None
        self.INFMR = None
        self.INFMS = None
        self.INFM_RESP = None
        self.INFM_SCORE = None
        self.CFNR0 = None
        self.dv0 = None
        self.data_computed = False

        # Use the shared data of ObjectiveFunction (called once)
        self._setup_genmat_references()

    def _setup_genmat_references(self):
        """Set the reference to the shared GenMat data of ObjectiveFunction"""
        # Requires the ObjectiveFunction import
        from objective_manager import ObjectiveFunction
        
        if ObjectiveFunction._shared_genmat_data is not None:
            self.train_basis_con_resp = ObjectiveFunction._shared_genmat_data["train_basis_con_resp"]
            self.W_CON_RESP = ObjectiveFunction._shared_genmat_data["W_CON_RESP"]
            self.train_basis_con_score = ObjectiveFunction._shared_genmat_data["train_basis_con_score"]
            self.W_CON_SCORE = ObjectiveFunction._shared_genmat_data["W_CON_SCORE"]
            
            # Safeguard: set to read-only
            self.train_basis_con_resp.setflags(write=False)
            self.W_CON_RESP.setflags(write=False)
            self.train_basis_con_score.setflags(write=False)
            self.W_CON_SCORE.setflags(write=False)
            
            print(f"Constraint {self.constraint_id}: GenMat references set up (read-only)")
        else:
            raise RuntimeError("Shared GenMat data not loaded. Make sure ObjectiveFunction.load_shared_genmat() is called first.")

    def calculate_basis_num(self, N, m, S):
        """Number of basis functions"""
        basis = 1
        for s in range(1, S+1):
            basis = basis + math.comb(N, s) * math.comb(m, s)
        return basis
    
    # The former _load_genmat_data method was removed (replaced by the shared data)
    
    def evaluate(self, dV, zz, rsvl, rssc, LR=1):
        """
        Evaluate the constraint function
        """

        globalVars.cntCon += 1
        self.call_count += 1
        self.iter_count += 1
        
        if globalVars.constraint_log:
            print(f"Count constraint func {self.constraint_id}:", self.call_count)
        
        # Extract the settings
        NN = self.config['NN']
        mm = self.config['mm']
        ms = self.config['ms']
        nd = self.config['nd']
        SM = self.config['SM']
        SS = self.config['SS']
        nA_resp_product = self.config['nA_resp_product']
        nA_score_product = self.config['nA_score_product']
        
        # Number of basis functions
        nAo = self.calculate_basis_num(NN, mm, SM)
        nAs = self.calculate_basis_num(NN, ms, SS)
        nA = max(nAo, nAs)
        
        nSampleR = nAo * nA_resp_product
        nSampleS = nAs * nA_score_product
        
        # # Design variables
        # meanX = np.zeros((nd, 1))
        # for i in range(nd):
        #     meanX[i] = dV[i]
            
        # # Input samples
        # xx = np.zeros_like(zz)
        # for i in range(nd):
        #     xx[:, i] = zz[:, i] + meanX[i]
            
        # Match the output size
        rsvl = rsvl[:nSampleR, :]
        rssc = rssc[:nSampleS, :]
        
        
        
        if self.call_count == 1:
            # First call: full computation
            return self._first_evaluation(dV, zz, rsvl, rssc, LR, 
                                        nAo, nAs, nA, nSampleR, nSampleS, nd)
        else:
            # Subsequent calls: reuse the existing results (single-pass update)
            return self._subsequent_evaluation(dV, zz, rsvl, LR, 
                                             nSampleR, nA, nAo, nAs, nd)
    
    def _first_evaluation(self, dV, zz, rsvl, rssc, LR, 
                         nAo, nAs, nA, nSampleR, nSampleS, nd):
        """Evaluation logic for the first call"""
        
        # Basis transformation accounting for skipped_sample_indices
        # rsvl is passed with the skipped rows already removed, so it is used as is
        if globalVars.skipped_sample_indices:
            # Convert 1-based to 0-based indices
            skipped_indices_0based = set(idx - 1 for idx in globalVars.skipped_sample_indices)
            # Collect valid indices sequentially (excluding skipped indices)
            valid_indices = []
            idx = 0
            while len(valid_indices) < nSampleR:
                if idx not in skipped_indices_0based:
                    valid_indices.append(idx)
                idx += 1
            # Build the basis over the required range
            n_extended = idx  # last checked index + 1
            ONB_CON_RESP_full = self.W_CON_RESP @ self.train_basis_con_resp[:, :n_extended]
            INFMR = ONB_CON_RESP_full.T[valid_indices, :]
        else:
            # No skipped samples: use the first rows directly
            ONB_CON_RESP = self.W_CON_RESP @ self.train_basis_con_resp[:, :nSampleR]
            INFMR = ONB_CON_RESP.T
        
        ONB_CON_SCORE = self.W_CON_SCORE @ self.train_basis_con_score[:, :nSampleS]
        INFMS = ONB_CON_SCORE.T
        
        # Constraint-function coefficients (rsvl is passed with skipped rows removed)
        CFNR = np.linalg.inv(INFMR.T @ INFMR) @ INFMR.T @ rsvl
        CFNR = CFNR.squeeze()  # LR=1, so convert to 1D
        
        # Score-function coefficients
        # if self.iter_count == 1:
        CFNS = np.zeros((nAs, nd))
        CFNS = np.linalg.inv(INFMS.T @ INFMS) @ INFMS.T @ rssc
        CFNS[0, :] = 0
        
        self.CFNS = CFNS  # store the score-function GPCE coefficients

        
        # Basis for the response (i, j indices)
        ONB_RESP = self.W_CON_RESP @ self.train_basis_con_resp
        INFM_RESP = ONB_RESP.T
        
        # Basis for the score function (k index)
        ONB_SCORE = self.W_CON_SCORE @ self.train_basis_con_score
        INFM_SCORE = ONB_SCORE.T

        # train_basis_obj = self.model.P_grlex_n_DD(zz, mm, SM)
        # W_CON = self.model.whitening_transform(train_basis_obj, train_basis_obj.shape[1])
        # #W_CON = self.model.whitening_transform_revise(self.config['NN'], mm, SM, stdZ, rho)
        # ONB = W_CON @ train_basis_obj
        # INFM = ONB.T
        
        # Sensitivities
        if globalVars.opt_method_type == "SLSQP":
            sen1, sen2, TRON = self._calculate_sensitivity_first(CFNR, CFNS, LR, nd, nA, nAs, nAo, INFM_RESP, INFM_SCORE, zz)
        else:
            sen1 = np.zeros((1, nd))
            sen2 = np.zeros((1, nd))
            TRON = np.zeros((nA, nA, nAs))
        
        # Constraint value and gradient
        if globalVars.method_type == "RDO":
            if self.rdo_constraint_func is None:
                raise RuntimeError(f"RDO constraint function is not provided for constraint {self.constraint_id}. "
                                 f"Please provide a constraint function using rdo_constraint_func parameter.")
            cnst, grdnV = self.rdo_constraint_func(CFNR, sen1, sen2, LR, nd) #### use the injected function
            
            # DEBUG: print the GPCE constraint value
            if globalVars.constraint_log:
                varOfcnst = np.sum(CFNR[1:, :] ** 2, axis=0)
                meanOfcnst = CFNR[0, :]
                print(f"\n[DEBUG Constraint {self.constraint_id}]")
                print(f"  CFNR[0] (mean) = {meanOfcnst[0]:.6f}")
                print(f"  sqrt(var) = {np.sqrt(varOfcnst)[0]:.6f}")
                print(f"  3*sqrt(var) - mean = {(3*np.sqrt(varOfcnst) - meanOfcnst)[0]:.6f}")
                print(f"  cnst (h(x) <= 0) = {cnst[0]:.6f}\n")
        
        elif globalVars.method_type == "RBDO":
            cnst, grdnV = self._calculate_constraint_and_gradient_RBDO(INFM_RESP, INFM_SCORE, CFNR, nd)
        
        # Store the results in instance variables
        self.TRON = TRON
        self.INFMR = INFMR
        self.INFMS = INFMS
        self.INFM_RESP = INFM_RESP
        self.INFM_SCORE = INFM_SCORE
        self.CFNR0 = CFNR.copy()  # explicit copy
        self.dv0 = dV.copy()
        self.data_computed = True
        
        # # Backup pickle file (optional)
        # saveVars = {"TRON": TRON, "INFMR": INFMR, "INFMS": INFMS,
        #             "INFM_RESP": INFM_RESP, "INFM_SCORE": INFM_SCORE,
        #             "CFNS": CFNS, "CFNR0": self.CFNR0, "dv0": self.dv0}
        
        # with open(self.pickle_file, 'wb') as file:
        #     pickle.dump(saveVars, file)
            
        return cnst, grdnV
    
    def _subsequent_evaluation(self, dV, zz, rsvl, LR, nSampleR, nA, nAo, nAs, nd):
        """Evaluation logic for subsequent calls - uses cached data"""
        
        # Raise an error if the data have not been computed
        if not self.data_computed:
            raise RuntimeError(f"Constraint {self.constraint_id}: Data not computed. Call _first_evaluation first.")
        
        # Use the cached data (instead of loading pickle files)
        TRON = self.TRON
        INFMR = self.INFMR
        INFMS = self.INFMS
        INFM_RESP = self.INFM_RESP
        INFM_SCORE = self.INFM_SCORE
        CFNS = self.CFNS
        CFNR0 = self.CFNR0
        dv0 = self.dv0

        # Update the basis for the new input (accounting for skipped_sample_indices)
        if globalVars.skipped_sample_indices:
            # Extended range accounting for skipped samples
            skipped_indices_0based = set(idx - 1 for idx in globalVars.skipped_sample_indices)
            # Collect valid indices sequentially
            valid_indices = []
            idx = 0
            while len(valid_indices) < nSampleR:
                if idx not in skipped_indices_0based:
                    valid_indices.append(idx)
                idx += 1
            n_extended = idx
            
            # Apply the MPSS transformation
            if globalVars.mpss_option == "SHIFTING":
                zz_extended = zz[:n_extended, :] + (dV - dv0)
            elif globalVars.mpss_option == "SCALING":
                zz_extended = zz[:n_extended, :] * (dV / dv0)
            else:
                raise ValueError(f"Unknown mpss_option: {globalVars.mpss_option}. Supported options are 'SHIFTING' and 'SCALING'.")
            
            # Build the basis over the extended range
            train_basis_con_resp_new_full = self.model.P_grlex_n_DD_fast(zz_extended, self.config['mm'], self.config['SM'])
            ONB_CON_RESP_NEW_full = self.W_CON_RESP @ train_basis_con_resp_new_full
            
            # Select only the valid indices
            INFMR_NEW = ONB_CON_RESP_NEW_full.T[valid_indices, :]
        else:
            # No skipped samples: use the first rows directly
            if globalVars.mpss_option == "SHIFTING":
                zz_new = zz[:nSampleR, :] + (dV - dv0)
            elif globalVars.mpss_option == "SCALING":
                zz_new = zz[:nSampleR, :] * (dV / dv0)
            else:
                raise ValueError(f"Unknown mpss_option: {globalVars.mpss_option}. Supported options are 'SHIFTING' and 'SCALING'.")
            
            train_basis_con_resp_new = self.model.P_grlex_n_DD_fast(zz_new, self.config['mm'], self.config['SM'])
            ONB_CON_RESP_NEW = self.W_CON_RESP @ train_basis_con_resp_new
            INFMR_NEW = ONB_CON_RESP_NEW.T
        
        # Predict the responses with the PCE
        rsvl = self.model.PCE(CFNR0, INFMR_NEW)
        
        # New coefficients
        CFNR = np.linalg.solve(INFMR.T @ INFMR, INFMR.T @ rsvl)
        
        # Sensitivities (nAo required)
        if globalVars.opt_method_type == "SLSQP":
            sen1, sen2 = self._calculate_sensitivity_subsequent(CFNR, CFNS, TRON, LR, nd, nA, nAo, nAs)
        else:
            sen1 = np.zeros((1, nd))
            sen2 = np.zeros((1, nd))
        
        # Constraint value and gradient
        if globalVars.method_type == "RDO":
            if self.rdo_constraint_func is None:
                raise RuntimeError(f"RDO constraint function is not provided for constraint {self.constraint_id}. "
                                 f"Please provide a constraint function using rdo_constraint_func parameter.")
            cnst, grdnV = self.rdo_constraint_func(CFNR, sen1, sen2, LR, nd) #### use the injected function
        elif globalVars.method_type == "RBDO":
            # raise RuntimeError("RBDO subsequent evaluation is not implemented.")
            cnst, grdnV = self._calculate_constraint_and_gradient_RBDO(INFM_RESP, INFM_SCORE, CFNR, nd)

        if globalVars.constraint_log:
            if globalVars.opt_method_type == "SLSQP":
                print(f"Constraint {self.constraint_id} - Sen1:", sen1)
                print(f"Constraint {self.constraint_id} - Sen2:", sen2)
                print(f"Constraint {self.constraint_id} - gradient:", grdnV.squeeze())

            print(f"Constraint {self.constraint_id} - value:", cnst[0])
        
        return cnst, grdnV.squeeze()
    
    def _calculate_sensitivity_first(self, CFNR, CFNS, LR, nd, nA, nAs, nAo, INFM_RESP, INFM_SCORE, zz):
        """Sensitivities (first call)"""
        
        # TRON
        import time
        print(f"[Constraint {self.constraint_id}] Starting TRON calculation: ({INFM_RESP.shape[0]}, {nAo}) x ({INFM_RESP.shape[0]}, {nAo}) x ({INFM_RESP.shape[0]}, {nAs})")
        t_start = time.time()

        TRON = np.einsum('ni,nj,nk->ijk', INFM_RESP, INFM_RESP, INFM_SCORE) / zz.shape[0]
        
        t_end = time.time()
        print(f"[Constraint {self.constraint_id}] TRON calculation completed in {t_end - t_start:.2f} seconds")

        # sen1 unchanged
        sen1 = np.zeros((1, nd))
        for i in range(nd):
            for j in range(min(nA, nAs)):
                sen1[0, i] += CFNR[j] * CFNS[j, i]

        # sen2
        sen2 = np.zeros((1, nd))

        # Special cases: (i=0 or j=0) AND k=i or k=j
        # combines the cases i=0, j!=0, k=j and j=0, i!=0, k=i
        for idx in range(1, min(nAo, nAs)):
            # case i=0, j=idx, k=idx
            sen2[0, :] += CFNR[0] * CFNR[idx] * CFNS[idx, :]
            # case j=0, i=idx, k=idx
            sen2[0, :] += CFNR[idx] * CFNR[0] * CFNS[idx, :]

        # Main case: i>=1, j>=1, k>=0
        # sen2[d] = Σ_i Σ_j Σ_k CFNR[i] * CFNR[j] * CFNS[k,d] * TRON[i,j,k]
        sen2_main = np.einsum('i,j,kd,ijk->d', CFNR[1:], CFNR[1:], CFNS, TRON[1:, 1:, :])
        sen2[0, :] += sen2_main

        # # First moment
        # sen1 = np.zeros((LR, nd))
        # for i in range(LR):
        #     for j in range(nd):
        #         for k in range(min(nAo, nAs)):  # nAo instead of nA
        #             sen1[i, j] += CFNR[k, i] * CFNS[k, j]
        
        # # Second moment and TRON
        # sen2 = np.zeros((LR, nd))
        # TRON = np.zeros((nAo, nAo, nAs))
        
        # for p in range(LR):
        #     for i in range(nAo):  # nAo instead of nA
        #         for j in range(nAo):  # nAo instead of nA
        #             for k in range(nAs):  # limit to the INFM size
        #                 if i == 0 or j == 0 or k == 0:
        #                     if (i == j and i == k and j == k):
        #                         pass
        #                     elif (i == 0 and j != 0):
        #                         if j == k and k < min(nAo, nAs):
        #                             sen2[p, :] += CFNR[i, p] * CFNR[j, p] * CFNS[k, :]
        #                     elif (j == 0 and i != 0):
        #                         if i == k and k < min(nAo, nAs):
        #                             sen2[p, :] += CFNR[i, p] * CFNR[j, p] * CFNS[k, :]
        #                 else:
        #                     index = [i, j, k]
        #                     if nAo <= nAs:  # nAo instead of nA
        #                         index1 = sorted(index)
        #                     else:
        #                         index1 = sorted(index, reverse=True)
        #                     if TRON[index1[0], index1[1], index1[2]] == 0:
        #                         TRON[index1[0], index1[1], index1[2]] = np.sum(
        #                             INFM_RESP[:, i] * INFM_RESP[:, j] * INFM_SCORE[:, k]) / zz.shape[0]
        #                     sen2[p, :] += (CFNR[i, p] * CFNR[j, p] * CFNS[k, :] * 
        #                                  TRON[index1[0], index1[1], index1[2]])
        
        return sen1, sen2, TRON
    
    def _calculate_sensitivity_subsequent(self, CFNR, CFNS, TRON, LR, nd, nA, nAo, nAs):
        """Sensitivities (subsequent calls)"""
        
        # First moment - LR=1, so CFNR is 1D
        sen1 = np.zeros((1, nd))
        for i in range(nd):
            for j in range(min(nA, nAs)):
                sen1[0, i] += CFNR[j] * CFNS[j, i]
        
        # sen2
        sen2 = np.zeros((1, nd))

        # Special cases: (i=0 or j=0) AND k=i or k=j
        # combines the cases i=0, j!=0, k=j and j=0, i!=0, k=i
        for idx in range(1, min(nAo, nAs)):
            # case i=0, j=idx, k=idx
            sen2[0, :] += CFNR[0] * CFNR[idx] * CFNS[idx, :]
            # case j=0, i=idx, k=idx
            sen2[0, :] += CFNR[idx] * CFNR[0] * CFNS[idx, :]

        # Main case: i>=1, j>=1, k>=0
        # sen2[d] = Σ_i Σ_j Σ_k CFNR[i] * CFNR[j] * CFNS[k,d] * TRON[i,j,k]
        sen2_main = np.einsum('i,j,kd,ijk->d', CFNR[1:], CFNR[1:], CFNS, TRON[1:, 1:, :])
        sen2[0, :] += sen2_main

        # # Second moment
        # sen2 = np.zeros((LR, nd))
        # for p in range(LR):
        #     for i in range(nAo):  # nAo instead of nA
        #         for j in range(nAo):  # nAo instead of nA
        #             for k in range(min(nAo, nAs)):  # limit to the TRON size
        #                 if i == 0 or j == 0 or k == 0:
        #                     if (i == j and i == k and j == k):
        #                         pass
        #                     elif (i == 0 and j != 0):
        #                         if j == k:
        #                             sen2[p, :] += CFNR[i, p] * CFNR[j, p] * CFNS[k, :]
        #                     elif (j == 0 and i != 0):
        #                         if i == k:
        #                             sen2[p, :] += CFNR[i, p] * CFNR[j, p] * CFNS[k, :]
        #                 else:
        #                     index = [i, j, k]
        #                     if nAo <= nAs:  # nAo instead of nA
        #                         index1 = sorted(index)
        #                     else:
        #                         index1 = sorted(index, reverse=True)
        #                     sen2[p, :] += (CFNR[i, p] * CFNR[j, p] * CFNS[k, :] * 
        #                                  TRON[index1[0], index1[1], index1[2]])
        
        return sen1, sen2
    
    def _calculate_constraint_and_gradient_RDO(self, CFNR, sen1, sen2, LR, nd):
        """Constraint value and gradient (returned in the original form h(x) <= 0)"""
        
        varOfcnst = np.sum(CFNR[1:, :] ** 2)
        meanOfcnst = CFNR[0, :]
        cnst = (3 * np.sqrt(varOfcnst) - meanOfcnst)
        
        grdnV = np.zeros((nd, LR))
        for i in range(LR):
            # Gradient of the original constraint, grad h(x)
            grdnV[:, i] = (-1 * sen1[i, :].T + 
                          (3/2) * (1/np.sqrt(varOfcnst)) * 
                          (sen2[i, :] - 2 * meanOfcnst[i] * sen1[i, :]).T)
        
        return cnst, grdnV

    def _calculate_constraint_and_gradient_RBDO(self, INFM_RESP, INFM_SCORE, CFNR, nd):
        """
        Reliability constraint value and gradient (returned in the original form h(x) <= 0)
        Constraint: h(d) = P_F - p_tgt <= 0  (the failure probability must not exceed the target)
            ∂h/∂d = ∂P_F/∂d = E[ I{g<=0} * s_k(Z;d) ]
        """

        # 1) Predict the performance function g(Z;d) with the GPCE (response basis)
        # CFNR is already squeezed to 1D, so rsvl is also 1D
        rsvl = self.model.PCE(CFNR, INFM_RESP)   # (L,)

        # 2) Failure probability
        failure_count = np.sum(rsvl <= 0)
        failure_prob  = failure_count / rsvl.shape[0]       # P_F


        # # Debug output for constraint Y1
        # if hasattr(self, 'constraint_id') and self.constraint_id == 1:
        #     # print(f"[DEBUG][Y1] rsvl <= 0 count: {failure_count} / {rsvl.shape[0]}")
        #     # print(f"[DEBUG][Y1] rsvl stats: mean={np.mean(rsvl):.6f}, min={np.min(rsvl):.6f}, max={np.max(rsvl):.6f}")

        #     # Save the CFNR values to CSV
        #     import csv
        #     with open(globalVars.filepath + "/result/CFNR_debug_Y1.csv", mode='a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(CFNR.flatten())

        # # Debug output for constraint Y2
        # if hasattr(self, 'constraint_id') and self.constraint_id == 2:
        #     # print(f"[DEBUG][Y2] rsvl <= 0 count: {failure_count} / {rsvl.shape[0]}")
        #     # print(f"[DEBUG][Y2] rsvl stats: mean={np.mean(rsvl):.6f}, min={np.min(rsvl):.6f}, max={np.max(rsvl):.6f}")

        #     # Save the CFNR values to CSV
        #     import csv
        #     with open(globalVars.filepath + "/result/CFNR_debug_Y2.csv", mode='a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(CFNR.flatten())

        # # Debug output for constraint Y3 only
        # if hasattr(self, 'constraint_id') and self.constraint_id == 3:
        #     # print(f"[DEBUG][Y3] rsvl <= 0 count: {failure_count} / {rsvl.shape[0]}")
        #     # print(f"[DEBUG][Y3] rsvl stats: mean={np.mean(rsvl):.6f}, min={np.min(rsvl):.6f}, max={np.max(rsvl):.6f}")

        #     # Save the CFNR values to CSV
        #     import csv
        #     with open(globalVars.filepath + "/result/CFNR_debug_Y3.csv", mode='a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(CFNR.flatten())

        

        # 3) Reconstruct the score-function samples s(Z;d) (score-function basis)
        if self.CFNS is None:
            raise RuntimeError("CFNS (score GPCE coefficients) is not set. Compute it in _first_evaluation first.")
        s_mat = self.model.PCE(self.CFNS, INFM_SCORE)             # (L, nd)

        # 4) Indicator function I{g<=0} (failure domain)
        I_fail = (rsvl <= 0).astype(float).reshape(-1, 1)  # (L, 1), explicitly 2D

        # 5) ∂P_F/∂d_k ≈ mean( I_fail * s_k )
        grad_failure_prob = (I_fail * s_mat).mean(axis=0)   # (nd,)

        # 6) Gradient of the original constraint, dh/dd = dP_F/dd
        grdnV = grad_failure_prob.reshape(-1, 1)         # (nd, 1)

        # 7) Original constraint: h(d) = P_F - p_tgt <= 0
        cnst = np.array([failure_prob - self.target_failure_prob])

        # Log output
        if globalVars.constraint_log:
            print(f"[RBDO] Pf={failure_prob:.6e}, target={self.target_failure_prob:.6e}, h={cnst[0]:.6e}")

        return cnst, grdnV

class MultiConstraintManager:
    """
    Manager class for multiple constraint functions
    """
    def __init__(self):
        self.constraints = []
        
    def add_constraint(self, constraint_id, config, pickle_file_prefix="constraint", 
                      genmat_file= globalVars.filepath + "/data/GenMat.pickle",
                      rdo_constraint_func=None):
        """
        Add a constraint function
        
        Parameters:
        -----------
        constraint_id : int
            Constraint function ID
        config : dict
            Constraint function settings
        pickle_file_prefix : str
            Prefix of the pickle files
        genmat_file : str
            Path of the GenMat file
        rdo_constraint_func : callable, required for RDO
            Function for the RDO constraint. Signature: func(self, CFNR, sen1, sen2, LR, nd)
            Required for RDO. For RBDO, the constraint is computed by the fixed formulation.
        
        Raises:
        -------
        RuntimeError
            If rdo_constraint_func is None for RDO
        """
        constraint = ConstraintFunction(
            constraint_id, config, pickle_file_prefix, genmat_file,
            rdo_constraint_func
        )
        self.constraints.append(constraint)
        return constraint
    
    def reset_call_counts(self):
        """Reset call_count of all constraint functions to 0"""
        for constraint in self.constraints:
            constraint.call_count = 0
        print(f"Reset call counts for {len(self.constraints)} constraints")

    def evaluate_all(self, dV, zz, rsvl_array, rssc, LR=1):
        """
        Evaluate all constraint functions
        rsvl_array: numpy array of shape (num_samples, num_constraints)
        """
        constraint_values = []
        constraint_gradients = []
        
        if len(self.constraints) != rsvl_array.shape[1]:
            raise ValueError("Number of constraints does not match rsvl_array shape.")
        else:
            for i, constraint in enumerate(self.constraints):
                # rsvl_array[:, i] is 1D, so convert to 2D
                rsvl_single = rsvl_array[:, i].reshape(-1, 1)
                cnst, grad = constraint.evaluate(dV, zz, rsvl_single, rssc, LR)
                constraint_values.append(cnst[0])  # scalar value
                constraint_gradients.append(grad)
        
        return np.array(constraint_values), np.array(constraint_gradients)
    
    def get_constraint_functions_for_scipy(self, zz, rsvl_array, rssc, LR=1):
        """
        Return the list of constraint functions for SciPy optimize
        rsvl_array: numpy array of shape (num_samples, num_constraints)
        """
        constraints = []
        
        for i, constraint in enumerate(self.constraints):
            def make_constraint_func(idx):
                def fun(dV):
                    rsvl_single = rsvl_array[:, idx].reshape(-1, 1)
                    cnst, grad = self.constraints[idx].evaluate(dV, zz, rsvl_single, rssc, LR)
                    return cnst[0] # sign convention for SLSQP
                
                def jac(dV):
                    rsvl_single = rsvl_array[:, idx].reshape(-1, 1)
                    cnst, grad = self.constraints[idx].evaluate(dV, zz, rsvl_single, rssc, LR)
                    return grad.squeeze()   # sign convention for SLSQP
                
                return fun, jac
            
            func, jac = make_constraint_func(i)
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x, f=func: -f(x), ## capture the function as a default argument (closure issue)
                'jac': lambda x, j=jac: -j(x)   ## capture the function as a default argument (closure issue)
            })
        
        return constraints
