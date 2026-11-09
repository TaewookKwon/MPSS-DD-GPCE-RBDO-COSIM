# Multiple objective Functions Manager
# Classes for managing multiple objective functions

import numpy as np
import pickle
import math
from scipy.stats import multivariate_normal
import globalVars
import csv
import os
from datetime import datetime
import os
import sys

############################# objective Functions Parameters ########################################################

# REDLAB_UQ.py (DD-GPCE basis construction) is located in the project root
from REDLAB_UQ import SurrogateModel

######################################################################################################################



class ObjectiveFunction:
    """
    Class for an individual objective function
    """
    # Class-level shared variables
    _shared_genmat_data = None
    _genmat_file_path = None
    _genmat_loaded = False
    
    @classmethod
    def load_shared_genmat(cls, genmat_file):
        """Load the GenMat data once at the class level"""
        if cls._genmat_file_path != genmat_file or not cls._genmat_loaded:
            print(f"Loading shared GenMat data from: {genmat_file}")
            with open(genmat_file, 'rb') as file:
                cls._shared_genmat_data = pickle.load(file)
            cls._genmat_file_path = genmat_file
            cls._genmat_loaded = True
            print(f"Shared GenMat data loaded successfully")
        else:
            print(f"Using cached shared GenMat data")
    
    @classmethod
    def clear_shared_genmat(cls):
        """Release the shared GenMat data"""
        cls._shared_genmat_data = None
        cls._genmat_file_path = None
        cls._genmat_loaded = False
        print("Shared GenMat data cleared")
    
    def __init__(self, objective_id, config, pickle_file_prefix="objective", genmat_file="GenMat.pickle", 
                 rdo_obj_calculate_func=None, obj0_type="mean"):
        self.objective_id = objective_id
        self.config = config
        self.pickle_file = globalVars.filepath + f"/data/{pickle_file_prefix}_{objective_id}.pickle"
        self.genmat_file = genmat_file  # GenMat file name
        self.model = SurrogateModel()
        self.call_count = 0
        self.iter_count = 0
        self.CFNS = None  # score-function GPCE coefficients
        
        # obj0 type ("mean" or "std")
        self.obj0_type = obj0_type
        
        # RBDO parameters
        self.target_failure_prob = 1.35e-3  # target failure probability (not used for the objective)
        
        # Function for the RDO objective (can be injected externally)
        self.rdo_obj_calculate_func = rdo_obj_calculate_func
        
        # Basis data (loaded from genmat_file) - references to the shared data
        self.train_basis_obj_resp = None
        self.W_OBJ_RESP = None
        self.train_basis_obj_score = None
        self.W_OBJ_SCORE = None
        
        # Load the shared GenMat data
        ObjectiveFunction.load_shared_genmat(self.genmat_file)
        self._setup_genmat_references()
        
        # Computed results (used instead of pickle files)
        self.TRON = None
        self.INFMR = None
        self.INFMS = None
        self.INFM_RESP = None
        self.INFM_SCORE = None
        self.CFNR0 = None
        self.dv0 = None
        self.obj0 = 0
        self.data_computed = False

    def _setup_genmat_references(self):
        """Set the reference to the shared GenMat data"""
        if ObjectiveFunction._shared_genmat_data is not None:
            self.train_basis_obj_resp = ObjectiveFunction._shared_genmat_data["train_basis_obj_resp"]
            self.W_OBJ_RESP = ObjectiveFunction._shared_genmat_data["W_OBJ_RESP"]
            self.train_basis_obj_score = ObjectiveFunction._shared_genmat_data["train_basis_obj_score"]
            self.W_OBJ_SCORE = ObjectiveFunction._shared_genmat_data["W_OBJ_SCORE"]
            
            # Safeguard: set to read-only
            self.train_basis_obj_resp.setflags(write=False)
            self.W_OBJ_RESP.setflags(write=False)
            self.train_basis_obj_score.setflags(write=False)
            self.W_OBJ_SCORE.setflags(write=False)
            
            print(f"Objective {self.objective_id}: GenMat references set up (read-only)")
        else:
            raise RuntimeError("Shared GenMat data not loaded")

    def calculate_basis_num(self, N, m, S):
        """Number of basis functions"""
        basis = 1
        for s in range(1, S+1):
            basis = basis + math.comb(N, s) * math.comb(m, s)
        return basis
    
    # The former _load_genmat_data method was removed (replaced by the shared data)
    
    def evaluate(self, dV, zz, rsvl, rssc, LR=1):
        """
        Evaluate the objective function
        """

        # globalVars.cntCon += 1
        self.call_count += 1
        self.iter_count += 1
        
        if globalVars.objective_log:
            print(f"Count objective func {self.objective_id}:", self.call_count)
        
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
            ONB_OBJ_RESP_full = self.W_OBJ_RESP @ self.train_basis_obj_resp[:, :n_extended]
            INFMR = ONB_OBJ_RESP_full.T[valid_indices, :]
        else:
            # No skipped samples: use the first rows directly
            ONB_OBJ_RESP = self.W_OBJ_RESP @ self.train_basis_obj_resp[:, :nSampleR]
            INFMR = ONB_OBJ_RESP.T
        
        # Basis for the score function
        ONB_OBJ_SCORE = self.W_OBJ_SCORE @ self.train_basis_obj_score[:, :nSampleS]
        INFMS = ONB_OBJ_SCORE.T
        
        # Coefficients (rsvl is passed with skipped rows removed), standard least squares
        CFNR = np.linalg.inv(INFMR.T @ INFMR) @ INFMR.T @ rsvl
        
        CFNR = CFNR.squeeze()

        # Score-function coefficients
        # if self.iter_count == 1:
        CFNS = np.zeros((nAs, nd))
        CFNS = np.linalg.inv(INFMS.T @ INFMS) @ INFMS.T @ rssc
        CFNS[0, :] = 0
        

        
        # Basis for the response (i, j indices)
        ONB_RESP = self.W_OBJ_RESP @ self.train_basis_obj_resp
        INFM_RESP = ONB_RESP.T
        
        # Basis for the score function (k index)
        ONB_SCORE = self.W_OBJ_SCORE @ self.train_basis_obj_score
        INFM_SCORE = ONB_SCORE.T

        
        # Sensitivities
        if globalVars.opt_method_type == "SLSQP":
            sen1, sen2, TRON = self._calculate_sensitivity_first(CFNR, CFNS, nd, nA, nAs, nAo, INFM_RESP, INFM_SCORE, zz)
        else:
            sen1 = np.zeros((1, nd))
            sen2 = np.zeros((1, nd))
            TRON = np.zeros((nA, nA, nAs))
        
        varOfobj = np.sum(CFNR[1:]**2)
        meanOfobj = CFNR[0]
        
        # # DEBUG: check CFNR at the initial evaluation
        # if not self.data_computed:
        #     print(f"\n[DEBUG Objective {self.objective_id}] Initial evaluation:")
        #     print(f"  CFNR[0] (mean) = {CFNR[0]:.4f}")
        #     print(f"  sqrt(sum(CFNR[1:]^2)) (std) = {np.sqrt(varOfobj):.4f}")
        #     print(f"  Expected from Table 3: mean=31.5568, std=17.0268\n")

        if(not self.data_computed): ## set the initial value obj0
            if globalVars.initial_val_control:
                self.obj0 = globalVars.obj_initial[self.objective_id - 1]  # prescribed initial value
            else:
                if self.obj0_type == "mean":
                    self.obj0 = meanOfobj  # mean at the initial design
                    print(f"Objective {self.objective_id}: obj0 set to MEAN = {self.obj0}")
                elif self.obj0_type == "std":
                    self.obj0 = np.sqrt(varOfobj)  # standard deviation at the initial design, Var_0
                    print(f"Objective {self.objective_id}: obj0 set to STD = {self.obj0}")
                else:
                    raise ValueError(f"Invalid obj0_type: {self.obj0_type}. Must be 'mean' or 'std'.")

        # Use the external function or the built-in computation
        if self.rdo_obj_calculate_func is not None:
            # Use the externally injected function
            valObj, vObjGrad = self.rdo_obj_calculate_func(CFNR, sen1, sen2, self.obj0, dV, nd)
        else:
            raise RuntimeError("RDO objective function is not provided. Please provide a objective function using rdo_obj_calculate_func parameter.")

        # # Objective value and gradient
        # if globalVars.method_type == "RDO":
        #     if self.rdo_obj_calculate_func is None:
        #         raise RuntimeError(f"RDO objective function is not provided for objective {self.objective_id}. "
        #                          f"Please provide a objective function using rdo_obj_calculate_func parameter.")
        #     cnst, grdnV = self.rdo_obj_calculate_func(self, CFNR, sen1, sen2, dV, nd) #### use the injected function
        
        # elif globalVars.method_type == "RBDO":
        #     cnst, grdnV = self._calculate_value_and_gradient_RBDO(INFM_RESP, INFM_SCORE, CFNR, nd)
        
        # Store the results in instance variables
        self.TRON = TRON
        self.INFMR = INFMR
        self.INFMS = INFMS
        self.CFNS = CFNS
        self.INFM_RESP = INFM_RESP
        self.INFM_SCORE = INFM_SCORE
        self.CFNR0 = CFNR.copy()  # explicit copy
        self.dv0 = dV.copy()
        
        self.data_computed = True
        
        if globalVars.objective_log:
            print("first loop: dV:", dV)
            print("first loop:var of obj: ", valObj)
            print("first loop:mean of obj: ", meanOfobj)
            if (globalVars.opt_method_type == "SLSQP"):
                print("first loop Objsen1: ", sen1)
                print("first loop Objsen2: ", sen2)

        # # Backup pickle file (optional)
        # saveVars = {"TRON": TRON, "INFMR": INFMR, "INFMS": INFMS,
        #             "INFM_RESP": INFM_RESP, "INFM_SCORE": INFM_SCORE,
        #             "CFNS": CFNS, "CFNR0": self.CFNR0, "dv0": self.dv0}
        
        # with open(self.pickle_file, 'wb') as file:
        #     pickle.dump(saveVars, file)
            
        return valObj, vObjGrad.squeeze()
    
    def _subsequent_evaluation(self, dV, zz, rsvl, LR, nSampleR, nA, nAo, nAs, nd):
        """Evaluation logic for subsequent calls - uses cached data"""
        
        # Raise an error if the data have not been computed
        if not self.data_computed:
            raise RuntimeError(f"Objective {self.objective_id}: Data not computed. Call _first_evaluation first.")
        
        # Use the cached data (instead of loading pickle files)
        TRON = self.TRON
        INFMR = self.INFMR
        INFMS = self.INFMS
        INFM_RESP = self.INFM_RESP
        INFM_SCORE = self.INFM_SCORE
        CFNS = self.CFNS
        CFNR0 = self.CFNR0
        dv0 = self.dv0
        #obj0 = self.obj0

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
            train_basis_obj_resp_new_full = self.model.P_grlex_n_DD_fast(zz_extended, self.config['mm'], self.config['SM'])
            ONB_OBJ_RESP_NEW_full = self.W_OBJ_RESP @ train_basis_obj_resp_new_full
            
            # Select only the valid indices
            INFMR_NEW = ONB_OBJ_RESP_NEW_full.T[valid_indices, :]
        else:
            # No skipped samples: use the first rows directly
            if globalVars.mpss_option == "SHIFTING":
                zz_new = zz[:nSampleR, :] + (dV - dv0)
            elif globalVars.mpss_option == "SCALING":
                zz_new = zz[:nSampleR, :] * (dV / dv0)
            else:
                raise ValueError(f"Unknown mpss_option: {globalVars.mpss_option}. Supported options are 'SHIFTING' and 'SCALING'.")
            
            train_basis_obj_resp_new = self.model.P_grlex_n_DD_fast(zz_new, self.config['mm'], self.config['SM'])
            ONB_OBJ_RESP_NEW = self.W_OBJ_RESP @ train_basis_obj_resp_new
            INFMR_NEW = ONB_OBJ_RESP_NEW.T
        
        # Predict the responses with the PCE
        rsvl = self.model.PCE(CFNR0, INFMR_NEW)
        
        # New coefficients
        CFNR = np.linalg.solve(INFMR.T @ INFMR, INFMR.T @ rsvl)
        CFNR = CFNR.squeeze()

        # Sensitivities (nAo required)
        sen1 = np.zeros((1, nd))
        sen2 = np.zeros((1, nd))

        if globalVars.opt_method_type == "SLSQP":
            sen1, sen2 = self._calculate_sensitivity_subsequent(CFNR, CFNS, TRON, nd, nA, nAo, nAs)
        
        varOfobj = np.sum(CFNR[1:]**2)
        meanOfobj = CFNR[0]

        # Use the external function or the built-in computation
        if self.rdo_obj_calculate_func is not None:
            # Use the externally injected function
            valObj, vObjGrad = self.rdo_obj_calculate_func(CFNR, sen1, sen2, self.obj0, dV, nd)
        else:
            raise RuntimeError("RDO objective function is not provided. Please provide a objective function using rdo_obj_calculate_func parameter.")
        
        if globalVars.objective_log:
            print("obj0: ", self.obj0)
            print("meanOfobj: ", meanOfobj)
            print("stdOfobj: ", np.sqrt(varOfobj))
            print("valObj: ", valObj)
            print("dV:", dV)
            if (globalVars.opt_method_type == "SLSQP"):
                print("objSen1: ", (sen1))
                print("objSen2: ", (sen2))
                print("vObjGrad: ", vObjGrad)
        
        if(globalVars.sopt == 'pre'):
            globalVars.stat0 = [meanOfobj, np.sqrt(varOfobj)]
        elif(globalVars.sopt == 'post'):
            globalVars.statf = [meanOfobj, np.sqrt(varOfobj)]
        
        return valObj, vObjGrad.squeeze()
    
    def _calculate_sensitivity_first(self, CFNR, CFNS, nd, nA, nAs, nAo, INFM_RESP, INFM_SCORE, zz):
        """Sensitivities (first call)"""
        
        # TRON
        import time
        print(f"[Objective {self.objective_id}] Starting TRON calculation: ({INFM_RESP.shape[0]}, {nAo}) x ({INFM_RESP.shape[0]}, {nAo}) x ({INFM_RESP.shape[0]}, {nAs})")
        t_start = time.time()

        TRON = np.einsum('ni,nj,nk->ijk', INFM_RESP, INFM_RESP, INFM_SCORE) / zz.shape[0]

        t_end = time.time()
        print(f"[Objective {self.objective_id}] TRON calculation completed in {t_end - t_start:.2f} seconds")

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
        # sen1 = np.zeros((1, nd))
        # for i in range(nd):
        #         for j in range(min(nA, nAs)):
        #             sen1[0, i] += CFNR[j] * CFNS[j, i]
        
        # # Second moment and TRON
        # sen2 = np.zeros((1, nd))
        # TRON = np.zeros((nA, nA, nAs))
        
        # for i in range(nAo):  # nAo instead of nA
        #     for j in range(nAo):  # nAo instead of nA
        #         for k in range(nAs):  # limit to the INFM size
        #             if i == 0 or j == 0 or k == 0:
        #                 if (i == j and i == k and j == k):
        #                     pass
        #                 elif (i == 0 and j != 0):
        #                     if j == k and k < min(nAo, nAs):
        #                         sen2[0, :] += CFNR[i] * CFNR[j] * CFNS[k, :]
        #                 elif (j == 0 and i != 0):
        #                     if i == k and k < min(nAo, nAs):
        #                         sen2[0, :] += CFNR[i] * CFNR[j] * CFNS[k, :]
        #             else:
        #                 index = [i, j, k]
        #                 if nA <= nAs:  
        #                     index1 = sorted(index)
        #                 else:
        #                     index1 = sorted(index, reverse=True)
        #                 if TRON[index1[0], index1[1], index1[2]] == 0:
        #                     TRON[index1[0], index1[1], index1[2]] = np.sum(
        #                         INFM_RESP[:, i] * INFM_RESP[:, j] * INFM_SCORE[:, k]) / zz.shape[0]
        #                 sen2[0, :] += (CFNR[i] * CFNR[j] * CFNS[k, :] * 
        #                                 TRON[index1[0], index1[1], index1[2]])
        
        return sen1, sen2, TRON
    
    def _calculate_sensitivity_subsequent(self, CFNR, CFNS, TRON, nd, nA, nAo, nAs):
        """Sensitivities (subsequent calls)"""
        
        # First moment
        sen1 = np.zeros((1, nd))
        for i in range(nd):
            for j in range(min(nA, nAs)):  # limit to min(nA, nAs)
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
        # sen2 = np.zeros((1, nd))
        # for i in range(nA):  # nAo instead of nA
        #     for j in range(nA):  # nAo instead of nA
        #         for k in range(nAs):  # limit to the TRON size
        #             if i == 0 or j == 0 or k == 0:
        #                 if (i == j and i == k and j == k):
        #                     pass
        #                 elif (i == 0 and j != 0):
        #                     if j == k:
        #                         sen2[0, :] += CFNR[i] * CFNR[j] * CFNS[k, :]
        #                 elif (j == 0 and i != 0):
        #                     if i == k:
        #                         sen2[0, :] += CFNR[i] * CFNR[j] * CFNS[k, :]
        #             else:
        #                 index = [i, j, k]
        #                 if nAo <= nAs:  # nAo instead of nA
        #                     index1 = sorted(index)
        #                 else:
        #                     index1 = sorted(index, reverse=True)
        #                 sen2[0, :] += (CFNR[i] * CFNR[j] * CFNS[k, :] * 
        #                                 TRON[index1[0], index1[1], index1[2]])
        
        return sen1, sen2
    
    # def _calculate_value_and_gradient_RDO(self, CFNR, sen1, sen2, LR, nd):
    #     """Constraint value and gradient (returned in the original form h(x) <= 0)"""
        
    #     varOfcnst = np.sum(CFNR[1:, :] ** 2)
    #     meanOfcnst = CFNR[0, :]
    #     cnst = (3 * np.sqrt(varOfcnst) - meanOfcnst)
        
    #     grdnV = np.zeros((nd, LR))
    #     for i in range(LR):
    #         # Gradient of the original constraint, grad h(x)
    #         grdnV[:, i] = (-1 * sen1[i, :].T + 
    #                       (3/2) * (1/np.sqrt(varOfcnst)) * 
    #                       (sen2[i, :] - 2 * meanOfcnst[i] * sen1[i, :]).T)
        
    #     return cnst, grdnV

    # def _calculate_value_and_gradient_RBDO(self, INFM_RESP, INFM_SCORE, CFNR, nd):
    #     """
    #     Reliability constraint value and gradient (returned in the original form h(x) <= 0)
    #     Constraint: h(d) = P_F - p_tgt <= 0  (the failure probability must not exceed the target)
    #         ∂h/∂d = ∂P_F/∂d = E[ I{g<=0} * s_k(Z;d) ]
    #     """

    #     # 1) Predict the performance function g(Z;d) with the GPCE (response basis)
    #     rsvl = self.model.PCE(CFNR, INFM_RESP)   # (L, 1)

    #     # 2) Failure probability
    #     failure_count = np.sum(rsvl <= 0)
    #     failure_prob  = failure_count / rsvl.shape[0]       # P_F


    #     # Debug output for constraint Y1
    #     if hasattr(self, 'objective_id') and self.objective_id == 1:
    #         # print(f"[DEBUG][Y1] rsvl <= 0 count: {failure_count} / {rsvl.shape[0]}")
    #         # print(f"[DEBUG][Y1] rsvl stats: mean={np.mean(rsvl):.6f}, min={np.min(rsvl):.6f}, max={np.max(rsvl):.6f}")

    #         # Save the CFNR values to CSV
    #         import csv
    #         with open(globalVars.filepath + "/result/CFNR_debug_Y1.csv", mode='a', newline='') as file:
    #             writer = csv.writer(file)
    #             writer.writerow(CFNR.flatten())

    #     # Debug output for constraint Y2
    #     if hasattr(self, 'objective_id') and self.objective_id == 2:
    #         # print(f"[DEBUG][Y2] rsvl <= 0 count: {failure_count} / {rsvl.shape[0]}")
    #         # print(f"[DEBUG][Y2] rsvl stats: mean={np.mean(rsvl):.6f}, min={np.min(rsvl):.6f}, max={np.max(rsvl):.6f}")

    #         # Save the CFNR values to CSV
    #         import csv
    #         with open(globalVars.filepath + "/result/CFNR_debug_Y2.csv", mode='a', newline='') as file:
    #             writer = csv.writer(file)
    #             writer.writerow(CFNR.flatten())

    #     # Debug output for constraint Y3 only
    #     if hasattr(self, 'objective_id') and self.objective_id == 3:
    #         # print(f"[DEBUG][Y3] rsvl <= 0 count: {failure_count} / {rsvl.shape[0]}")
    #         # print(f"[DEBUG][Y3] rsvl stats: mean={np.mean(rsvl):.6f}, min={np.min(rsvl):.6f}, max={np.max(rsvl):.6f}")

    #         # Save the CFNR values to CSV
    #         import csv
    #         with open(globalVars.filepath + "/result/CFNR_debug_Y3.csv", mode='a', newline='') as file:
    #             writer = csv.writer(file)
    #             writer.writerow(CFNR.flatten())

        

    #     # 3) Reconstruct the score-function samples s(Z;d) (score-function basis)
    #     if self.CFNS is None:
    #         raise RuntimeError("CFNS (score GPCE coefficients) is not set. Compute it in _first_evaluation first.")
    #     s_mat = self.model.PCE(self.CFNS, INFM_SCORE)             # (L, nd)

    #     # 4) Indicator function I{g<=0} (failure domain)
    #     I_fail = (rsvl <= 0).astype(float)                  # (L, 1)

    #     # 5) ∂P_F/∂d_k ≈ mean( I_fail * s_k )
    #     grad_failure_prob = (I_fail * s_mat).mean(axis=0)   # (nd,)

    #     # 6) Gradient of the original constraint, dh/dd = dP_F/dd
    #     grdnV = grad_failure_prob.reshape(-1, 1)         # (nd, 1)

    #     # 7) Original constraint: h(d) = P_F - p_tgt <= 0
    #     cnst = np.array([failure_prob - self.target_failure_prob])

    #     # Log output
    #     print(f"[RBDO] Pf={failure_prob:.6e}, target={self.target_failure_prob:.6e}, h={cnst[0]:.6e}")

    #     return cnst, grdnV

class MultiObjectiveManager:
    """
    Manager class for multiple objective functions
    
    Supports two evaluation modes:
    1. evaluate_scalarized: single scalarized objective via a weighted sum
    2. evaluate_multi_objective: for multi-objective algorithms such as NSGA-II
    """
    def __init__(self):
        self.objectives = []
        self.obj0_types = ["std"]  # default: standard deviation
        
    def set_obj0_types(self, obj0_types):
        """
        Set the obj0 types
        
        Parameters:
        -----------
        obj0_types : list
            List of obj0 types for the objectives (e.g., ["mean", "std"])
            "mean": use the mean, "std": use the standard deviation
        """
        self.obj0_types = obj0_types
        print(f"Set obj0 types: {obj0_types}")
        
        # Validity check
        valid_types = ["mean", "std"]
        for obj0_type in obj0_types:
            if obj0_type not in valid_types:
                raise ValueError(f"Invalid obj0_type: {obj0_type}. Must be one of {valid_types}")
                
    def add_objective(self, objective_id, config, pickle_file_prefix="objective", 
                      genmat_file= globalVars.filepath + "/data/GenMat.pickle",
                      rdo_obj_calculate_func=None, obj0_type=None):
        """
        Add an objective function
        
        Parameters:
        -----------
        objective_id : int
            Objective function ID
        config : dict
            Objective function settings
        pickle_file_prefix : str
            Prefix of the pickle files
        genmat_file : str
            Path of the GenMat file
        rdo_obj_calculate_func : callable, optional
            Function for the objective. Signature: func(CFNR, sen1, sen2, obj0, LR, nd)
            If None, the built-in computation is used
        obj0_type : str, optional
            obj0 type ("mean": mean, "std": standard deviation)
            If None, selected automatically from the self.obj0_types list
        """
        # If obj0_type is not specified, select it from the list
        if obj0_type is None:
            raise RuntimeError("obj0_type must be specified when adding an objective.")
        
        objective = ObjectiveFunction(
            objective_id, config, pickle_file_prefix, genmat_file,
            rdo_obj_calculate_func, obj0_type
        )
        self.objectives.append(objective)
        return objective
    
    def reset_call_counts(self):
        """Reset call_count of all objective functions to 0"""
        for objective in self.objectives:
            objective.call_count = 0
        print(f"Reset call counts for {len(self.objectives)} objectives")

    def evaluate_scalarized(self, dV, zz, rsvl_array, rssc, weights, LR=1):
        """
        Evaluate the scalarized objective (weighted sum of the objectives)
        
        Parameters:
        -----------
        dV : array
            Design variables
        zz : array
            Input samples
        rsvl_array : numpy array
            Response array (num_samples, num_objectives)
        rssc : array
            Score samples
        weights : array or list
            Weight of each objective (length = number of objectives)
        LR : int
            Number of responses
            
        Returns:
        --------
        scalar_value : float
            Scalarized objective value
        scalar_gradient : array
            Gradient of the scalarized objective
        """
        if len(self.objectives) != rsvl_array.shape[1]:
            raise ValueError("Number of objectives does not match rsvl_array shape.")
        
        if len(weights) != len(self.objectives):
            raise ValueError("Number of weights must match number of objectives.")
        
        # Evaluate each objective
        objective_values = []
        objective_gradients = []
        
        for i, objective in enumerate(self.objectives):
            rsvl_single = rsvl_array[:, i].reshape(-1, 1)
            value, grad = objective.evaluate(dV, zz, rsvl_single, rssc, LR)
            objective_values.append(value)
            objective_gradients.append(grad)
        
        # Scalarization: f_scalar = sum(w_i * f_i)
        scalar_value = sum(w * val for w, val in zip(weights, objective_values))
        
        # Scalarized gradient: grad f_scalar = sum(w_i * grad f_i)
        scalar_gradient = np.zeros_like(objective_gradients[0])
        for w, grad in zip(weights, objective_gradients):
            scalar_gradient += w * grad
        
        if globalVars.objective_log:
            if len(self.objectives) > 1:
                print(f"Scalarized objective: {scalar_value}")
                print(f"Individual objectives: {[f'{val}' for val in objective_values]}")
            else:
                print(f"Objective: {scalar_value}")

        #print(f"Weights: {weights}")
        
        return scalar_value, scalar_gradient
    
    def evaluate_multi_objective(self, dV, zz, rsvl_array, rssc, LR=1):
        """
        Evaluate multiple objectives (used for NSGA-II, etc.)
        
        Parameters:
        -----------
        dV : array
            Design variables
        zz : array
            Input samples
        rsvl_array : numpy array
            Response array (num_samples, num_objectives)
        rssc : array
            Score samples
        LR : int
            Number of responses
            
        Returns:
        --------
        objective_values : list
            List of objective values
        objective_gradients : list
            List of objective gradients
        """
        if len(self.objectives) != rsvl_array.shape[1]:
            raise ValueError("Number of objectives does not match rsvl_array shape.")
        
        objective_values = []
        objective_gradients = []
        
        for i, objective in enumerate(self.objectives):
            rsvl_single = rsvl_array[:, i].reshape(-1, 1)
            value, grad = objective.evaluate(dV, zz, rsvl_single, rssc, LR)
            objective_values.append(value)
            objective_gradients.append(grad)
        
        if globalVars.objective_log:
            print(f"Multi-objective values: {[f'{val:.6f}' for val in objective_values]}")
        
        return objective_values, objective_gradients

    def save_obj0_values_to_csv(self, iteration_q=None, debug_folder=globalVars.filepath + "/debugging_file"):
        """
        Save the obj0 values of the objectives to a CSV file
        
        Parameters:
        -----------
        iteration_q : int, optional
            Current iteration number (q)
        debug_folder : str
            Output folder path
        """
        # Create the debugging folder
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # File name
        if iteration_q is not None:
            filename = f"obj0_values_q{iteration_q}_{timestamp}.csv"
        else:
            filename = f"obj0_values_{timestamp}.csv"
        
        filepath = os.path.join(debug_folder, filename)
        
        # Collect the obj0 values
        obj0_data = []
        for i, objective in enumerate(self.objectives):
            obj0_data.append({
                'objective_id': objective.objective_id,
                'obj0_value': objective.obj0,
                'obj0_type': objective.obj0_type,
                'data_computed': objective.data_computed,
                'call_count': objective.call_count
            })
        
        # Save as CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['objective_id', 'obj0_value', 'obj0_type', 'data_computed', 'call_count'])
            writer.writeheader()
            writer.writerows(obj0_data)
        
        print(f"obj0 values saved to: {filepath}")
        print(f"Collected obj0 values: {[obj['obj0_value'] for obj in obj0_data]}")
        
        return filepath

    def evaluate_all(self, dV, zz, rsvl_array, rssc, LR=1):
        """
        Function kept for backward compatibility (same as the multi-objective evaluation)
        """
        return self.evaluate_multi_objective(dV, zz, rsvl_array, rssc, LR)
    
