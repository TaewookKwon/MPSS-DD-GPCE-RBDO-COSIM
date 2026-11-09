## Revised 25.09.15
# Added optimizer selection (GA)

import matplotlib.pyplot as plt
import numpy as np
import math as mt
import pickle
from datetime import datetime
from functools import partial
import os
import sys
import globalVars  
# REDLAB_UQ.py (DD-GPCE basis construction) is located in the project root
from REDLAB_UQ import SurrogateModel


############################################# MPSS Parameters ###############################################################
# sys.path.append(os.path.abspath(''))

## 1. Import input sampling function
#from sample_generator import generate_input_samples
#from sample_generator_ex4 import generate_input_samples_ex4


## 2. Set scorefunctions
#from scorefunction_generator_gauss import generate_score_function
from scorefunction_generator_lognormal import generate_score_function


## 3. Import response sampling functions
from response_dataset_generator_project2 import generate_response_dataset, generate_samples
#from response_dataset_generator_ex4 import generate_response_dataset, generate_samples # RDO
#from response_dataset_generator_ex1 import generate_response_dataset, generate_samples # RBDO


## 4. Set Problem defunition

# Project 2: RBDO
NN = 6
nd = 6
NUM_OBJECTIVES = 1
NUM_CONSTRAINTS = 3  # three reliability constraints (PEEQ, Kx, Ky)

x0 = np.array([36, 36, 70, 5, 70, 305])  # initial design: d1-d4 [mm], mean Young's modulus [GPa], mean UTS [MPa]
#x0 = np.array([26.894765993870866, 26.78963336562726, 60.5466610514048, 7.870646571918311, 70.00000019778525, 304.9999999981768])
#x0 = np.array([26.930517536446672, 26.770576760636352, 60.63862817162523, 7.8531500183692255, 70.00000019383073, 304.99999999120695])  # initial design [d1, d2, ...] ## 251231
bnd = [[24, 36.01], [24, 40], [60, 80], [3, 9], [70-1e-6, 70+1e-6], [305-1e-6, 305+1e-6]]  # design bounds; the material means (5th, 6th) are fixed

if globalVars.restart_enabled:
    x0 = np.asarray(globalVars.restart_x0, dtype=float).copy()


# # RBDO Example 1
# NN = 2
# nd = 2
# NUM_OBJECTIVES = 1  # number of objective functions (one)
# NUM_CONSTRAINTS = 3  # number of constraint functions
# x0 = np.array([4, 4])  # initial design
# bnd = [[0, 10], [0, 10]]  # design-variable bounds

## 5. Set Input random variable properties
# standard deviation of input random variable Z

stdZ = np.array([0.02, 0.02, 0.02, 0.02, 0.05, 0.025])

rho1 = 0.4  # correlation among the geometric variables (variables 1-4)
rho2 = 0.05    # correlation between Young's modulus and tensile strength (variables 5-6)

corr = np.eye(NN)  # initialize with ones on the diagonal

# Correlation among the 1st-4th (geometric) variables: rho1
for i in range(4):
    for j in range(4):
        if i != j:
            corr[i, j] = rho1

# Correlation between the 5th-6th variables (Young's modulus - tensile strength): rho2
corr[4, 5] = rho2
corr[5, 4] = rho2

# covariance matrix of input random variable Z, used for the score functions
cova = np.zeros((nd, nd))
for i in range(nd):
    for j in range(i, nd):
        cova[i, j] = stdZ[i] * stdZ[j] * corr[i, j]

for i in range(nd):
    for j in range(nd):
        cova[j, i] = cova[i, j]

seed = 123457


# 6. Set GPCE parameters
# GPCE settings - objective function
GPCE_CONFIG_OBJECTIVE = {
    'NN': NN,        # number of random variables
    'mm': 2,        # polynomial order of the response surrogate (m)
    'ms': 1,        # polynomial order of the score-function expansion
    'nd': nd,        # number of design variables
    'SM': 1,        # interaction order of the response surrogate (S)
    'SS': 1,        # interaction order of the sensitivity (score-function) expansion
    'nA_resp_product': 2,
    'nA_score_product': 10  # sample multiplier for the score-function basis
}

# GPCE settings - constraint functions (base)
GPCE_CONFIG_CONSTRAINT_BASE = {
    'NN': NN,        # number of random variables (same as the objective)
    'mm': 2,        # polynomial order of the response surrogate (m)
    'ms': 1,        # polynomial order of the score-function expansion
    'nd': nd,        # number of design variables (same as the objective)
    'SM': 1,        # interaction order of the response surrogate (S)
    'SS': 1,        # interaction order of the sensitivity (score-function) expansion
    'nA_resp_product': 2,
    'nA_score_product': 10,  # sample multiplier for the score-function basis
}




#################################################################################################################

from objective_manager import MultiObjectiveManager # modify here when switching to RDO
from constraint_manager import MultiConstraintManager



if globalVars.method_type == "RDO":
    if globalVars.opt_method_type in ["NSGA2", "NSGA3"]:
        from run_opt_mpss_multi import runfmincon_mpss, obj_calculate_functions, rdo_constraint_functions, obj0_types
    else:
        from run_opt_mpss import runfmincon_mpss, obj_calculate_functions, rdo_constraint_functions, obj0_types, weights

elif globalVars.method_type == "RBDO":
    if globalVars.opt_method_type in ["NSGA2", "NSGA3"]:
        from run_opt_mpss_multi import runfmincon_mpss, obj_calculate_functions, obj0_types  # RDO functions are not needed for RBDO
    else:
        from run_opt_mpss import runfmincon_mpss, obj_calculate_functions, obj0_types, weights

    rdo_constraint_functions = []
else:
    raise ValueError(f"Unknown method_type: {globalVars.method_type}. Supported types are 'RDO' and 'RBDO'.")

# Initialize global variables
globalVars.cntObj = 0
globalVars.cntCon = 0
globalVars.stat0 = []
globalVars.statf = []
globalVars.filepath_samples = None
globalVars.sampling_counter = 0
if globalVars.restart_enabled:
    globalVars.sampling_counter = globalVars.restart_sampling_counter

filepath_input = globalVars.filepath + "/data/input_samples_zz.csv"

header = [f'X{i+1}' for i in range(NN)]  # X1, X2, ..., XN
meanX = x0.tolist()  # convert to a list for compatibility with the existing code


# Create the settings for each constraint function
constraint_configs = []
for i in range(NUM_CONSTRAINTS):
    config = GPCE_CONFIG_CONSTRAINT_BASE.copy()
    # Use the base settings as they are (can be modified individually if needed)
    constraint_configs.append(config)

## Number of basis functions
def calculate_basis_num(N,m,S):
    basis = 1
    for s in range(1,S+1):
        basis = basis + mt.comb(N, s)*mt.comb(m, s)
    return basis

## Number of basis functions for the objective
nAo_obj = calculate_basis_num(GPCE_CONFIG_OBJECTIVE['NN'],GPCE_CONFIG_OBJECTIVE['mm'],GPCE_CONFIG_OBJECTIVE['SM'])
nAs_obj = calculate_basis_num(GPCE_CONFIG_OBJECTIVE['NN'],GPCE_CONFIG_OBJECTIVE['ms'],GPCE_CONFIG_OBJECTIVE['SS'])
nA_obj = max(nAo_obj, nAs_obj)

nSampleR_obj = nAo_obj*GPCE_CONFIG_OBJECTIVE['nA_resp_product']
nSampleS_obj = nAs_obj*GPCE_CONFIG_OBJECTIVE['nA_score_product']

## Maximum number of basis functions over the constraints
max_nAo_con = 0
max_nAs_con = 0
max_nSampleR_con = 0
max_nSampleS_con = 0


constraint_basis_info = []

for i, config in enumerate(constraint_configs):
    nAo_con = calculate_basis_num(config['NN'], config['mm'], config['SM'])
    nAs_con = calculate_basis_num(config['NN'], config['ms'], config['SS'])
    nSampleR_con = nAo_con * config['nA_resp_product']
    nSampleS_con = nAs_con * config['nA_score_product']
    
    constraint_basis_info.append({
        'nAo': nAo_con,
        'nAs': nAs_con,
        'nSampleR': nSampleR_con,
        'nSampleS': nSampleS_con
    })
    
    max_nAo_con = max(max_nAo_con, nAo_con)
    max_nAs_con = max(max_nAs_con, nAs_con)
    max_nSampleR_con = max(max_nSampleR_con, nSampleR_con)
    max_nSampleS_con = max(max_nSampleS_con, nSampleS_con)

# Overall maximum values
nAo = max(nAo_obj, max_nAo_con)
nAs = max(nAs_obj, max_nAs_con)
nA = max(nAo, nAs)
nSampleR = max(nSampleR_obj, max_nSampleR_con)
nSampleS = max(nSampleS_obj, max_nSampleS_con)
nSample = max(nSampleR, nSampleS)
nSampleO = 1_000_000

print(f"Max basis numbers - nAo: {nAo}, nAs: {nAs}, nA: {nA}")
print(f"Max sample numbers - nSampleR: {nSampleR}, nSampleS: {nSampleS}")
print(f"Constraint basis info:")
for i, info in enumerate(constraint_basis_info):
    print(f"  Constraint {i+1}: nSampleR={info['nSampleR']}, nAo={info['nAo']}")





## Input samples
# seed = 123457
if globalVars.first_samples == "yes":
    pass
    filepath_input_init = os.path.join(globalVars.filepath, "initial", "input_samples_zz.csv")
    
    # Copy initial input samples to data folder
    import shutil
    
    if os.path.exists(filepath_input_init):
        shutil.copy2(filepath_input_init, filepath_input)
        print(f"Copied initial input samples from {filepath_input_init} to {filepath_input}")
    else:
        raise FileNotFoundError(f"Initial input samples file not found: {filepath_input_init}")

else:
    raise RuntimeError("generate_input_samples function is currently disabled when first_samples is 'no'. Please provide input_samples.csv manually in the data folder.")
    #generate_input_samples_ex4(NN, corr, stdZ, seed, filepath_input, header, nSampleO) ## nSample -> nSampleO (one million)
    #generate_input_samples_ex4([1] * NN, stdZ, corr, filepath_input, header, nSampleO, seed)


## Load the input samples
zz = np.loadtxt(filepath_input, delimiter=',', skiprows=1)

if globalVars.mpss_option == "SHIFTING":
    xx = zz + meanX  # add meanX to each column (broadcasting)
elif globalVars.mpss_option == "SCALING":
    xx = zz * meanX  # multiply each column by meanX (broadcasting)
else:
    raise ValueError(f"Unknown mpss_option: {globalVars.mpss_option}. Supported options are 'SHIFTING' and 'SCALING'.")


# Save the xx array to file (for checking)
np.savetxt(globalVars.filepath + "/data/input_samples.csv", xx, delimiter=',', header=','.join(header), comments='')

print("Generating response data...")

# Score-function values
rssc = generate_score_function(zz, nSampleS, cova, stdZ, GPCE_CONFIG_OBJECTIVE)

def generate_basis_matrices(zz, GPCE_CONFIG_OBJECTIVE, GPCE_CONFIG_CONSTRAINT_BASE):
    """
    Generate the basis matrices and save them to a pickle file
    
    Parameters:
    - zz: input sample array
    - GPCE_CONFIG_OBJECTIVE: GPCE settings for the objective
    - GPCE_CONFIG_CONSTRAINT_BASE: GPCE settings for the constraints
    - stdZ: array of standard deviations
    - rho: correlation coefficient
    """
    print("Generating basis matrices...")
    
   
    model = SurrogateModel()


    ### Basis construction (RBDO: full basis required)
    # Objective basis (whitening uses the full array)
    train_basis_obj_resp = model.P_grlex_n_DD_fast(zz, GPCE_CONFIG_OBJECTIVE['mm'], GPCE_CONFIG_OBJECTIVE['SM'])
    W_OBJ_RESP = model.whitening_transform(train_basis_obj_resp, train_basis_obj_resp.shape[1])  # whitening with the full array
    
    train_basis_obj_score = model.P_grlex_n_DD_fast(zz, GPCE_CONFIG_OBJECTIVE['ms'], GPCE_CONFIG_OBJECTIVE['SS'])
    W_OBJ_SCORE = model.whitening_transform(train_basis_obj_score, train_basis_obj_score.shape[1])  # whitening with the full array

    # Constraint basis (whitening uses the full array)
    train_basis_con_resp = model.P_grlex_n_DD_fast(zz, GPCE_CONFIG_CONSTRAINT_BASE['mm'], GPCE_CONFIG_CONSTRAINT_BASE['SM'])
    W_CON_RESP = model.whitening_transform(train_basis_con_resp, train_basis_con_resp.shape[1])  # whitening with the full array

    train_basis_con_score = model.P_grlex_n_DD_fast(zz, GPCE_CONFIG_CONSTRAINT_BASE['ms'], GPCE_CONFIG_CONSTRAINT_BASE['SS'])
    W_CON_SCORE = model.whitening_transform(train_basis_con_score, train_basis_con_score.shape[1])  # whitening with the full array

    ## RDO: only part of the basis is required
    # Objective basis (memory saving: whitening with the full array, only the required part is stored)
    # train_basis_obj_resp_full = model.P_grlex_n_DD_fast(zz, GPCE_CONFIG_OBJECTIVE['mm'], GPCE_CONFIG_OBJECTIVE['SM'])
    # W_OBJ_RESP = model.whitening_transform(train_basis_obj_resp_full, train_basis_obj_resp_full.shape[1])  # whitening with the full array
    # train_basis_obj_resp = train_basis_obj_resp_full[:, :nSampleR_obj].copy()  # copy only the required part
    # del train_basis_obj_resp_full  # free memory
    # #W_OBJ_RESP = model.whitening_transform_revise(GPCE_CONFIG_OBJECTIVE['NN'],GPCE_CONFIG_OBJECTIVE['mm'],GPCE_CONFIG_OBJECTIVE['SM'], stdZ, rho)

    # train_basis_obj_score_full = model.P_grlex_n_DD_fast(zz, GPCE_CONFIG_OBJECTIVE['ms'], GPCE_CONFIG_OBJECTIVE['SS'])
    # W_OBJ_SCORE = model.whitening_transform(train_basis_obj_score_full, train_basis_obj_score_full.shape[1])  # whitening with the full array
    # train_basis_obj_score = train_basis_obj_score_full[:, :nSampleS_obj].copy()  # copy only the required part
    # del train_basis_obj_score_full  # free memory
    # #W_OBJ_SCORE = model.whitening_transform_revise(GPCE_CONFIG_OBJECTIVE['NN'],GPCE_CONFIG_OBJECTIVE['ms'],GPCE_CONFIG_OBJECTIVE['SS'], stdZ, rho)

    # # Constraint basis (memory saving: whitening with the full array, only the required part is stored)
    # train_basis_con_resp_full = model.P_grlex_n_DD_fast(zz, GPCE_CONFIG_CONSTRAINT_BASE['mm'], GPCE_CONFIG_CONSTRAINT_BASE['SM'])
    # W_CON_RESP = model.whitening_transform(train_basis_con_resp_full, train_basis_con_resp_full.shape[1])  # whitening with the full array
    # train_basis_con_resp = train_basis_con_resp_full[:, :nSampleR_obj].copy()  # copy only the required part
    # del train_basis_con_resp_full  # free memory
    # #W_CON_RESP = model.whitening_transform_revise(GPCE_CONFIG_CONSTRAINT_BASE['NN'],GPCE_CONFIG_CONSTRAINT_BASE['mm'],GPCE_CONFIG_CONSTRAINT_BASE['SM'], stdZ, rho)

    # train_basis_con_score_full = model.P_grlex_n_DD_fast(zz, GPCE_CONFIG_CONSTRAINT_BASE['ms'], GPCE_CONFIG_CONSTRAINT_BASE['SS'])
    # W_CON_SCORE = model.whitening_transform(train_basis_con_score_full, train_basis_con_score_full.shape[1])  # whitening with the full array
    # train_basis_con_score = train_basis_con_score_full[:, :nSampleS_obj].copy()  # copy only the required part
    # del train_basis_con_score_full  # free memory


    # Check the sizes and memory usage of the W arrays
    print(f"W array sizes and memory usage:")
    print(f"  - W_OBJ_RESP.shape: {W_OBJ_RESP.shape}")
    print(f"  - W_OBJ_RESP memory: {W_OBJ_RESP.nbytes / (1024**2):.2f} MB")
    print(f"  - W_OBJ_SCORE.shape: {W_OBJ_SCORE.shape}")
    print(f"  - W_OBJ_SCORE memory: {W_OBJ_SCORE.nbytes / (1024**2):.2f} MB")
    print(f"  - W_CON_RESP.shape: {W_CON_RESP.shape}")
    print(f"  - W_CON_RESP memory: {W_CON_RESP.nbytes / (1024**2):.2f} MB")
    print(f"  - W_CON_SCORE.shape: {W_CON_SCORE.shape}")
    print(f"  - W_CON_SCORE memory: {W_CON_SCORE.nbytes / (1024**2):.2f} MB")
    print(f"  - Total W array memory: {(W_OBJ_RESP.nbytes + W_OBJ_SCORE.nbytes + W_CON_RESP.nbytes + W_CON_SCORE.nbytes) / (1024**2):.2f} MB")
    
    print(f"\ntrain_basis arrays:")
    print(f"  - train_basis_obj_resp.shape: {train_basis_obj_resp.shape}")
    print(f"  - train_basis_obj_resp memory: {train_basis_obj_resp.nbytes / (1024**2):.2f} MB")
    print(f"  - train_basis_obj_score.shape: {train_basis_obj_score.shape}")
    print(f"  - train_basis_obj_score memory: {train_basis_obj_score.nbytes / (1024**2):.2f} MB")
    print(f"  - train_basis_con_resp.shape: {train_basis_con_resp.shape}")
    print(f"  - train_basis_con_resp memory: {train_basis_con_resp.nbytes / (1024**2):.2f} MB")
    print(f"  - train_basis_con_score.shape: {train_basis_con_score.shape}")
    print(f"  - train_basis_con_score memory: {train_basis_con_score.nbytes / (1024**2):.2f} MB")
    print(f"  - Total train_basis array memory: {(train_basis_obj_resp.nbytes + train_basis_obj_score.nbytes + train_basis_con_resp.nbytes + train_basis_con_score.nbytes) / (1024**2):.2f} MB")


    saveVar = {"train_basis_obj_resp":train_basis_obj_resp, "W_OBJ_RESP":W_OBJ_RESP,
                "train_basis_con_resp":train_basis_con_resp, "W_CON_RESP":W_CON_RESP,
                "train_basis_obj_score":train_basis_obj_score, "W_OBJ_SCORE":W_OBJ_SCORE,
                "train_basis_con_score":train_basis_con_score, "W_CON_SCORE":W_CON_SCORE,
                # "stdZ": stdZ, "rho": rho}
                }

    # Save the variables to file
    with open(globalVars.filepath + "/data/GenMat.pickle", 'wb') as file:
        pickle.dump(saveVar, file)

    print("zz shape:", zz.shape)

# Generate the basis matrices
generate_basis_matrices(zz, GPCE_CONFIG_OBJECTIVE, GPCE_CONFIG_CONSTRAINT_BASE)

## MPSS sample folder
def get_initial_samples(): ## creates the sample folder and stores its path in the global variables
    """
    Create the samples folder (named by date_time, without duplicates)
    """
    if globalVars.restart_enabled:
        if not os.path.isdir(globalVars.restart_sample_folder):
            raise FileNotFoundError(f"Restart sample folder not found: {globalVars.restart_sample_folder}")
        globalVars.filepath_samples = globalVars.restart_sample_folder
        globalVars.sampling_counter = globalVars.restart_sampling_counter
        print(f"Restart mode: using samples folder: {globalVars.filepath_samples}")
        print(f"Restart mode: sampling counter set to {globalVars.sampling_counter}")
        return

    base_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    samples_base = os.path.join(globalVars.filepath, "samples")
    
    # Try the default folder name
    samples_folder_path = os.path.join(samples_base, base_timestamp)
    
    # If a folder with the same name exists, append a number
    counter = 1
    original_folder = samples_folder_path
    while os.path.exists(samples_folder_path):
        samples_folder_path = f"{original_folder}_{counter:02d}"
        counter += 1
    
    # Create the folder
    os.makedirs(samples_folder_path, exist_ok=False)  # False, since duplicates are excluded above

    globalVars.filepath_samples = samples_folder_path
    print(f"Samples will be saved in: {globalVars.filepath_samples}")

    if globalVars.first_samples == "yes":
        print("!!!!!!!!!!! First samples option is set to 'yes'. Copying initial samples from 'init' folder. !!!!!!!!!!!")
        import shutil
        
        # Path of the initial sample CSV file
        init_csv_path = os.path.join(globalVars.filepath, "initial", "sample_1.csv")
        
        # Check that samples_folder_path is empty
        if os.path.exists(samples_folder_path) and os.listdir(samples_folder_path):
            raise RuntimeError(f"Samples folder is not empty: {samples_folder_path}. Cannot copy initial samples.")
        
        # Check that the initial CSV file exists
        if not os.path.exists(init_csv_path):
            raise FileNotFoundError(f"Initial samples file not found: {init_csv_path}")
        
        # Copy the CSV file to samples_folder_path as sample_1.csv
        target_csv_path = os.path.join(samples_folder_path, "sample_1.csv")
        shutil.copy2(init_csv_path, target_csv_path)
        globalVars.sampling_counter += 1  # increment the sampling counter (== 1)
        
        print(f"Copied initial samples from {init_csv_path} to {target_csv_path}")
    
    else: # == "no": run the analyses to generate the samples
        generate_samples(xx, nSample, nSampleR_obj, NUM_OBJECTIVES, NUM_CONSTRAINTS, constraint_basis_info)

# Set up and create the sample folder
get_initial_samples()

# Evaluation before optimization
globalVars.sopt = "pre"
print("!!!!! Initial evaluation...")

if globalVars.restart_enabled:
    print("Restart mode: skipping main initial response evaluation.")
    rsvl_obj = np.zeros((nSampleR_obj, NUM_OBJECTIVES))
    rsvl_con = np.zeros((constraint_basis_info[0]['nSampleR'], NUM_CONSTRAINTS))
else:
    rsvl_obj = np.zeros((nSampleR_obj, 1))  # objective responses (not used)
    rsvl_con = np.zeros((nSampleO, NUM_CONSTRAINTS))  # constraint responses

    rsvl_obj, rsvl_con = generate_response_dataset(xx, nSample, nSampleR_obj, NUM_OBJECTIVES, NUM_CONSTRAINTS, constraint_basis_info)

# Load the GenMat data once and share it among all objects
from objective_manager import ObjectiveFunction
ObjectiveFunction.load_shared_genmat(globalVars.filepath + "/data/GenMat.pickle")

# Create and set up the objective manager
objective_manager = MultiObjectiveManager()


# Add the objective functions
for i in range(NUM_OBJECTIVES):
    objective_manager.add_objective(
        objective_id=i+1, 
        config=GPCE_CONFIG_OBJECTIVE,
        pickle_file_prefix="objective",
        genmat_file= globalVars.filepath + "/data/GenMat.pickle",
        rdo_obj_calculate_func=obj_calculate_functions[i], # function for the RDO objective
        obj0_type = obj0_types[i], # obj0 type for each objective
    )

# Update the responses of the objective manager
if NUM_OBJECTIVES == 1:
    # Single objective: keep the same interface as the original objFunc
    rsvl_obj_array = rsvl_obj  # shape: (nSampleR_obj, 1)
    weights = [1.0]  # weight 1.0 for a single objective
else:
    # Multiple objectives: rsvl_obj is already of shape (nSampleR_obj, NUM_OBJECTIVES)
    rsvl_obj_array = rsvl_obj  # shape set by generate_response_dataset
    ## weights are defined separately in the parameters

# Wrap the scalarized function of objective_manager with the same interface as objFunc
if globalVars.opt_method_type == "NSGA2" or globalVars.opt_method_type == "NSGA3": ## multi-objective optimization
    def obj_func_wrapper(x, config=None, zz=None, rsvl=None, rssc=None):
        """Wrapper with the same interface as objFunc - multi-objective"""
        # config is accepted for compatibility but not used (objective_manager manages its own config)
        if zz is None or rsvl is None or rssc is None:
            raise ValueError("zz, rsvl, rssc parameters are required")
        return objective_manager.evaluate_multi_objective(x, zz, rsvl, rssc, LR=1)
            
    # Same partial pattern as before
    obj_func_partial = partial(obj_func_wrapper, config=GPCE_CONFIG_OBJECTIVE, zz=zz, rsvl=rsvl_obj_array, rssc=rssc)

else:
    def obj_func_wrapper(x, config=None, zz=None, rsvl=None, rssc=None, weights=None):
        """Wrapper with the same interface as objFunc"""
        # config is accepted for compatibility but not used (objective_manager manages its own config)
        if zz is None or rsvl is None or rssc is None or weights is None:
            raise ValueError("zz, rsvl, rssc, weights parameters are required")
        return objective_manager.evaluate_scalarized(x, zz, rsvl, rssc, weights=weights, LR=1)

    # Same partial pattern as before
    obj_func_partial = partial(obj_func_wrapper, config=GPCE_CONFIG_OBJECTIVE, zz=zz, rsvl=rsvl_obj_array, rssc=rssc, weights=weights)

if globalVars.restart_enabled:
    valObj = None
else:
    valObj, _ = obj_func_partial(x0) # the objective is evaluated inside run_opt_mpss

# Create the MultiConstraintManager for evaluating all constraints
constraint_manager = MultiConstraintManager()



# Check that enough constraint functions are provided
if len(rdo_constraint_functions) > 0 and len(rdo_constraint_functions) < NUM_CONSTRAINTS:
    raise RuntimeError("Not enough RDO constraint functions provided for the number of constraints, in main.py.")
    
# Add all constraint functions
for i, config in enumerate(constraint_configs):
    # Use the RDO function only if provided, otherwise None
    rdo_func = rdo_constraint_functions[i] if i < len(rdo_constraint_functions) else None
    constraint_manager.add_constraint(
        constraint_id=i+1, 
        config=config,
        pickle_file_prefix="constraint",
        genmat_file= globalVars.filepath + "/data/GenMat.pickle",
        rdo_constraint_func=rdo_func  # RDO constraint function (None for RBDO)
    )

# Evaluate all constraint functions
if globalVars.restart_enabled:
    c0_values = []
    c0_gradients = []
    print("Restart mode: main initial objective/constraint evaluation skipped.")
else:
    c0_values, c0_gradients = constraint_manager.evaluate_all(x0, zz, rsvl_con, rssc, LR=1)

    print(f"Initial objective value: {valObj}") # also printed in run_opt_mpss
    print(f"Initial constraint values: {c0_values}")

# Run the optimization
if not globalVars.restart_enabled:
    print(f"Initial constraint values: {c0_values}")

globalVars.sopt = "run"
print("!!!!! Starting optimization...")

[history, searchdir] = runfmincon_mpss(
    x0, bnd, GPCE_CONFIG_OBJECTIVE, constraint_configs, zz, rssc,
    nSample, nSampleR_obj, NUM_OBJECTIVES, NUM_CONSTRAINTS, constraint_basis_info)

xfTmp = np.array(history['x'])
print("Final result:", xfTmp)

# Collect the results
totAns = {"cnt":[1,2], "stat0" : 0, "statf":0, "c0":0, "cf":0}
totAns["cnt"] = [globalVars.cntObj, globalVars.cntCon]

# Evaluation after optimization
globalVars.sopt = "post"
print("!!!!! Final evaluation:")

# New input samples at the final design
if globalVars.mpss_option == "SHIFTING":
    x_final = zz + xfTmp
elif globalVars.mpss_option == "SCALING":
    x_final = zz * xfTmp
else:
    raise ValueError(f"Unknown mpss_option: {globalVars.mpss_option}. Supported options are 'SHIFTING' and 'SCALING'.")

# New responses at the final design
print("Generating final response dataset...")
rsvl_obj_final, rsvl_con_final = generate_response_dataset(x_final, nSample, nSampleR_obj, NUM_OBJECTIVES, NUM_CONSTRAINTS, constraint_basis_info)

# Re-evaluate the objective with the final responses
if globalVars.opt_method_type == "NSGA2" or globalVars.opt_method_type == "NSGA3":
    obj_func_partial_final = partial(obj_func_wrapper, config=GPCE_CONFIG_OBJECTIVE, zz=zz, rsvl=rsvl_obj_final, rssc=rssc)
else:
    obj_func_partial_final = partial(obj_func_wrapper, config=GPCE_CONFIG_OBJECTIVE, zz=zz, rsvl=rsvl_obj_final, rssc=rssc, weights=weights)

final_obj_value, _ = obj_func_partial_final(xfTmp)

print("Final objective value:")
print(final_obj_value)
print("Final design variables:")
print(xfTmp)

## Final verification
# print("!!!!! Final verification with new response data......") ## 
# rsvl_obj, rsvl_con = generate_response_dataset(x_final, nSample, nSampleR_obj, NUM_OBJECTIVES, NUM_CONSTRAINTS, constraint_basis_info) # load the generated samples

# # Re-evaluate the objective with the new responses
# rsvl_obj_array_new = rsvl_obj

# # Create a new partial object reusing the wrapper and weights
# if globalVars.opt_method_type == "NSGA2" or globalVars.opt_method_type == "NSGA3": ## multi-objective optimization
#     obj_func_partial_new = partial(obj_func_wrapper, config=GPCE_CONFIG_OBJECTIVE, zz=zz, rsvl=rsvl_obj_array_new, rssc=rssc)
# else:
#     obj_func_partial_new = partial(obj_func_wrapper, config=GPCE_CONFIG_OBJECTIVE, zz=zz, rsvl=rsvl_obj_array_new, rssc=rssc, weights=weights)
# final_obj_value_new, _ = obj_func_partial_new(xfTmp)  # use the new objective

# cf = constraint.evaluate(xfTmp, zz, rsvl_con[0], rssc, 1)

# totAns["stat0"] = np.array(globalVars.stat0).T if globalVars.stat0 else 0
# totAns["statf"] = np.array(globalVars.stat0).T if globalVars.stat0 else 0
# totAns["c0"] = c0
# totAns["cf"] = cf

# print("totAns: ", totAns)

# Release the shared memory before exiting
print("Clearing shared GenMat data...")
ObjectiveFunction.clear_shared_genmat()
print("Program completed successfully.")



