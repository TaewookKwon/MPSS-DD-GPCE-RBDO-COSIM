# Multi-constraint optimization runner
# Multi-point single-step optimization with multiple constraints
# Optimization driver using multiple constraint functions

import numpy as np
from scipy.optimize import minimize, differential_evolution
from functools import partial
import globalVars


############################### MPSS Parameters ###############################################

## 1. Import objetive function 

# from objective_function_ex1 import objFunc # RBDO

# Import objective manager for multi-objective optimization


## 2. Import constraint function 


## 3. Import response sampling functions
from response_dataset_generator_project2 import generate_response_dataset, generate_samples # RDO
#from response_dataset_generator_ex4 import generate_response_dataset, generate_samples # RDO
#from response_dataset_generator_ex1 import generate_response_dataset, generate_samples # RBDO

## 4. Set MPSS optimization criteria
global beta
beta = globalVars.restart_beta.copy() if globalVars.restart_enabled else [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]

# Termination criteria
epsilon1 = 5*1e-3
epsilon2 = 5*1e-3
epsilon3 = 0.01
epsilon4 = 0.07
epsilon5 = 0.01
epsilon6 = 0.5
epsilon7 = 0.01

tol_constraint = 1e-6
tol_sub = 1e-4
infeasible_attempt_limit = 10 # maximum number of interpolation attempts

## 5. Set Objective value and gradient calculation function for RDO
# RDO Problem
def _calculate_objective_value_and_gradient_mean(CFNR, sen1, sen2, obj0, dV, nd=None):
    """Objective value and gradient"""
    
    varOfobj = np.sum(CFNR[1:]**2)
    meanOfobj = CFNR[0]

    ########## valObj and vObjGrad according to the problem definition ##########
    valObj = meanOfobj / obj0

    # Differentiating the objective directly gives the expression below
    #vObjGrad = np.zeros((nd, 1))
    vObjGrad = ((1/(obj0)) * sen1).T
    
    return valObj, vObjGrad

def _calculate_objective_value_and_gradient_std(CFNR, sen1, sen2, obj0, dV, nd=None):
    """Objective value and gradient"""
    
    varOfobj = np.sum(CFNR[1:]**2)
    meanOfobj = CFNR[0]

    ########## valObj and vObjGrad according to the problem definition ##########
    valObj = np.sqrt(varOfobj) / obj0

    # Differentiating the objective directly gives the expression below
    #vObjGrad = np.zeros((nd, 1))
    vObjGrad = ((1/(2*obj0)) * (1/np.sqrt(varOfobj)) * (sen2 - 2*meanOfobj*sen1)).T
    
    return valObj, vObjGrad

# ex1
# weights = [1.0]

# obj_calculate_functions = [
#     _calculate_objective_value_and_gradient_mean,    
#     # more can be added as needed
# ]

# # # obj0 type for the initial objective value
# obj0_types = ['mean']

# Weights for scalarizing multiple objectives
weights = [1.0]

obj_calculate_functions = [
    _calculate_objective_value_and_gradient_mean,   
    # more can be added as needed
]

# # obj0 type for the initial objective value
obj0_types = ["mean"]

# ## RBDO Problem
# def _calculate_objective_value_and_gradient_RBDO_ex1(CFNR, sen1, sen2, obj0, dV, nd=None):
#     """Objective value and gradient"""
    
#     d1, d2 = dV[0], dV[1]
    
#     # Objective value: c0(d) = -d1 + d2
#     obj_val = -d1 + d2
    
#     # Gradient: dc0/dd1 = -1, dc0/dd2 = 1
#     obj_grad = np.array([-1.0, 1.0])
    
#     #print(f"Objective function: d1={d1:.4f}, d2={d2:.4f}, obj_val={obj_val:.6f}")
    
#     return obj_val, obj_grad

# obj_calculate_functions = [
#     _calculate_objective_value_and_gradient_RBDO_ex1,    
#     # more can be added as needed
# ]

## 6. Set Constraint value and gradient calculation function for RDO
# # RDO Problem
def _calculate_constraint_and_gradient_RDO(CFNR, sen1, sen2, LR, nd):
        """Constraint value and gradient (returned in the original form h(x) <= 0)"""
        
        # LR=1, so CFNR is a 1D array and sen1/sen2 have shape (1, nd)
        varOfcnst = np.sum(CFNR[1:] ** 2)
        meanOfcnst = CFNR[0]
        cnst = np.array([3 * np.sqrt(varOfcnst) - meanOfcnst])
        
        # Gradient of the original constraint, grad h(x)
        grdnV = (-1 * sen1[0, :].T + 
                  (3/2) * (1/np.sqrt(varOfcnst)) * 
                  (sen2[0, :] - 2 * meanOfcnst * sen1[0, :]).T)
        
        return cnst, grdnV

# List of RDO functions for each constraint
rdo_constraint_functions = [
    # _calculate_constraint_and_gradient_RDO,
    # _calculate_constraint_and_gradient_RDO,
    # _calculate_constraint_and_gradient_RDO,
    # _calculate_constraint_and_gradient_RDO,
    # _calculate_constraint_and_gradient_RDO,
    # _calculate_constraint_and_gradient_RDO,
    # _calculate_constraint_and_gradient_RDO,
    # _calculate_constraint_and_gradient_RDO,
    # _calculate_constraint_and_gradient_RDO,
    # _calculate_constraint_and_gradient_RDO,
    # _calculate_constraint_and_gradient_RDO,

    # # more can be added as needed
]

############################################################################################

from objective_manager import MultiObjectiveManager # modify here when switching to RDO
from constraint_manager import MultiConstraintManager # modify here when switching to RDO

# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.optimize import minimize as pymoo_minimize
# from pymoo.core.problem import Problem
# from pymoo.operators.crossover.sbx import SBX
# from pymoo.operators.mutation.pm import PM
# from pymoo.termination import get_termination


### Logging setup
import csv
import os
from datetime import datetime

history = {'x': [], 'fval': []}
searchdir = []

# Global variables for CSV logging
optimization_log = []
csv_file_path = None
step_log_csv_path = None

def initialize_csv_logging(log_dir= globalVars.filepath + "/optimization_logs", NUM_OBJECTIVES=1, nd=1):
    """Initialize the CSV log"""
    global csv_file_path
    
    # Create the log directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # File name including a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_path = os.path.join(log_dir, f"optimization_log_{timestamp}.csv")
    
    # CSV header according to the number of objectives
    headers = ["iteration"]
    
    # obj_value columns according to the number of objectives
    for i in range(NUM_OBJECTIVES):
        headers.append(f"obj_value_{i+1}")
    
    # beta column for each input variable (nd columns)
    for i in range(nd):
        headers.append(f"beta_{i+1}")
    
    # Basic columns (except x, beta, subLb, subUb)
    headers.extend(["constraint_violations", "feasible", 
                   "exit_success", "exit_message", "exit_status"])
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    
    print(f"Optimization log will be saved to: {csv_file_path}")

def initialize_step_log_csv(log_dir= globalVars.filepath + "/step_logs"):
    """Initialize the step-log CSV file"""
    global step_log_csv_path
    
    # Create the log directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # File name including a timestamp (YYMMDD_HHMMSS)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    step_log_csv_path = os.path.join(log_dir, f"step_log_{timestamp}.csv")
    
    # CSV header
    headers = ["iteration", "sizing_iteration", "sample_number", "message", "x0_values", "additional_info"]
    
    with open(step_log_csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    
    print(f"Step log will be saved to: {step_log_csv_path}")

def log_step_message(message, iteration=0, sizing_iteration=0, sample_number=None, x0=None, additional_info=""):
    """Write a step-log message to the CSV file"""
    global step_log_csv_path
    
    if step_log_csv_path is None:
        return
    
    # Current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # up to milliseconds
    
    # Convert x0 to a string
    x0_str = str(x0.tolist()) if x0 is not None else ""
    
    # Log data
    log_data = [
        iteration,
        sizing_iteration,
        sample_number if sample_number is not None else "",
        message,
        x0_str,
        additional_info
    ]
    
    # Append to the CSV file
    try:
        with open(step_log_csv_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(log_data)
    except Exception as e:
        print(f"Error writing to step log CSV: {e}")

def log_optimization_step(x, obj_value, constraint_values=None, iteration=0, current_beta_val=None, subLb=None, subUb=None, df=None, df_prev=None, optimization_result=None):
    """Log an optimization step to the CSV file"""
    global csv_file_path
    
    if csv_file_path is None:
        return
    
    # Check constraint violation (original form: h(x) <= 0 is feasible)
    constraint_violations = 0
    feasible = True
    if constraint_values is not None:
        constraint_violations = sum(1 for cv in constraint_values if cv > tol_constraint)
        feasible = constraint_violations == 0
    
    # Log data
    log_data = [iteration]  # subdomain (q)
    
    # Objective value (scalarized, so one value)
    log_data.append(obj_value)
    
    # beta value for each input variable (size of the beta array)
    if current_beta_val is not None:
        log_data.extend(current_beta_val.tolist() if hasattr(current_beta_val, 'tolist') else current_beta_val)
    
    # Basic data
    log_data.extend([
        constraint_violations,
        feasible,
        getattr(optimization_result, 'success', None) if optimization_result is not None else None,
        getattr(optimization_result, 'message', "N/A") if optimization_result is not None else "N/A",
        getattr(optimization_result, 'status', None) if optimization_result is not None else None
    ])
    
    # Append to the CSV file
    try:
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(log_data)
    except Exception as e:
        print(f"Error writing to CSV log: {e}")

def run_ga_optimization(obj_func_partial, constraint_manager, x0, bounds_mpss, zz, rsvl_con, rssc, nd, NUM_OBJECTIVES, NUM_CONSTRAINTS):
    """
    Optimization with the pymoo GA
    
    Parameters:
    -----------
    obj_func_partial : callable
        Objective function (single or multiple)
    constraint_manager : MultiConstraintManager
        Constraint function manager
    x0 : array_like
        Initial design point
    bounds_mpss : list
        Variable bounds [[lb1, ub1], [lb2, ub2], ...]
    zz, rsvl_con, rssc : arrays
        Data required for the constraint evaluation
    nd : int
        Number of design variables
    NUM_OBJECTIVES : int
        Number of objectives
    NUM_CONSTRAINTS : int
        Number of constraints
    
    Returns:
    --------
    res_like : object
        Result object similar to scipy.optimize.minimize
    """
    
    # pymoo Problem class
    class OptimizationProblem(Problem):
        def __init__(self):
            # Extract the lower/upper bounds from bounds_mpss
            xl = np.array([bound[0] for bound in bounds_mpss])
            xu = np.array([bound[1] for bound in bounds_mpss])
            
            super().__init__(n_var=nd, n_obj=NUM_OBJECTIVES, n_constr=NUM_CONSTRAINTS, xl=xl, xu=xu)
        
        def _evaluate(self, X, out, *args, **kwargs):
            # X shape: (pop_size, nd)
            F = []  # objective values
            G = []  # constraint values
            
            for x in X:
                # Objective evaluation
                f_val, _ = obj_func_partial(x)
                F.append(f_val)
                
                # Constraint evaluation
                g_vals, _ = constraint_manager.evaluate_all(x, zz, rsvl_con, rssc, LR=1)
                G.append(g_vals)
            
            out["F"] = np.array(F).reshape(-1, 1)  # one objective
            out["G"] = np.array(G)  # NUM_CONSTRAINTS constraints
    
    # Problem definition
    problem = OptimizationProblem()
    
    # GA settings
    algorithm = GA(
        pop_size=100,
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20, prob=1.0/nd),  # nd: number of variables
        eliminate_duplicates=True
    )
    
    # Termination criterion
    termination = get_termination("n_gen", 100)  # 100 generations
    
    print("Starting GA optimization...")
    
    # Run the optimization
    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        verbose=True,
        save_history=True
    )
    
    # Convert the result to a form similar to scipy.optimize.minimize
    class GAResult:
        def __init__(self, pymoo_res):
            self.x = pymoo_res.X  # optimum
            self.fun = pymoo_res.F[0] if pymoo_res.F is not None else None  # objective value
            self.success = pymoo_res.X is not None
            self.message = "GA optimization completed"
            self.status = 0 if self.success else 1
            self.nfev = pymoo_res.algorithm.evaluator.n_eval  # number of function evaluations
            self.njev = 0  # GA does not use gradients
    
    return GAResult(res)

def runfmincon_mpss(x0, bnd, GPCE_CONFIG_OBJECTIVE, constraint_configs, zz, rssc,
                          nSample, nSampleR_obj, NUM_OBJECTIVES, NUM_CONSTRAINTS, constraint_basis_info,
                         ):
    """
    MPSS optimization with multiple constraint functions
    
    Parameters:
    -----------
    x0 : array_like
        Initial design variables
    GPCE_CONFIG_OBJECTIVE : dict
        GPCE settings of the objective
    constraint_configs : list of dict
        List of GPCE settings of the constraints
    zz : array_like
        Input samples (shifted)
    rssc : array_like
        Score-function values
    response_functions : list of callable, optional
        List of response functions for the constraints. If None, the defaults are used
    
    Returns:
    --------
    history : dict
        Optimization history
    searchdir : list
        Search direction
    cntObj : int
        Number of objective evaluations
    cntCon : int
        Number of constraint evaluations
    """
    
    # Global variables
    global rdo_constraint_functions, obj_calculate_functions, obj0_types
    nd = GPCE_CONFIG_OBJECTIVE['nd']
    
    # Initialize the CSV log
    initialize_csv_logging(NUM_OBJECTIVES=NUM_OBJECTIVES, nd=nd)
    
    # Initialize the step-log CSV
    initialize_step_log_csv()
    
    # MPSS
    gr = (1+ np.sqrt(5)) / 2  # golden ratio constant
    


    fir = 0
    x0 = np.array(x0)  # make sure it is an array
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")
    if len(x0) != nd:
        raise ValueError(f"x0 length ({len(x0)}) must match nd ({nd})")

    lower_bounds = np.array([b[0] for b in bnd])
    upper_bounds = np.array([b[1] for b in bnd])

    # feasible region estimation
    df = np.zeros(nd)  # array instead of list
    df_prev = upper_bounds.copy()  # used after upper_bounds is defined
    
    # Initialize the subdomain bounds
    subLb = np.zeros(nd)
    subUb = np.zeros(nd)
    subLb_prev = np.zeros(nd)
    subUb_prev = np.zeros(nd)

    beta_init = np.asarray(globals()["beta"], dtype=float).reshape(-1)
    if len(beta_init) != nd:
        raise ValueError(f"Global beta length ({len(beta_init)}) must match nd ({nd})")
    beta = beta_init.copy()
    q = 0 # iteration counter

    objValue = 0
    objValue_prev_subdomain = 0 # optimization result of the previous subdomain (formerly objValueLast)
    objValue_prev = 0  # for the termination condition (formerly objValueLast0)
    constraint_values_prev = []  # initialization

    convergence_flag = False

    ## For RBDO, the RDO constraint functions are set to None
    if hasattr(globalVars, 'method_type') and globalVars.method_type == "RBDO":
        # Empty list for RBDO or if method_type is not defined
        rdo_constraint_functions = []
        print("Note: RDO functions not defined - using RBDO mode or method_type not set")

    # Create the objective manager
    objective_manager = MultiObjectiveManager()

    # Add the objective functions
    for i in range(NUM_OBJECTIVES):
        objective_manager.add_objective(
            objective_id=i+1, 
            config=GPCE_CONFIG_OBJECTIVE,
            pickle_file_prefix="objective",
            genmat_file= globalVars.filepath + "/data/GenMat.pickle",
            rdo_obj_calculate_func=obj_calculate_functions[i], # function for the RDO objective
            obj0_type = obj0_types[i]
        )

    # Create the constraint manager
    constraint_manager = MultiConstraintManager()
    
    # Raise an error if there are fewer functions than constraints
    if globalVars.method_type == "RDO" and len(rdo_constraint_functions) < NUM_CONSTRAINTS:
        raise RuntimeError("Not enough RDO constraint functions provided for the number of constraints, in run_opt_mpss.py.")

    
    # Add the constraint functions
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
    
    rsvl_obj = np.zeros((nSampleR_obj, 1))  # initialize the objective response array (not used)
    max_sampleR = max(constraint_basis_info[j]['nSampleR'] for j in range(NUM_CONSTRAINTS))
    rsvl_con = np.zeros((max_sampleR, NUM_CONSTRAINTS))
    restart_last_available_sample = None
    restart_sample_reuse_active = False

    if globalVars.restart_enabled:
        x0 = np.asarray(globalVars.restart_x0, dtype=float).reshape(-1)
        df_prev = np.asarray(globalVars.restart_df_prev, dtype=float).reshape(-1)
        subLb_prev = np.asarray(globalVars.restart_subLb_prev, dtype=float).reshape(-1)
        subUb_prev = np.asarray(globalVars.restart_subUb_prev, dtype=float).reshape(-1)
        beta = np.asarray(globalVars.restart_beta, dtype=float).reshape(-1)
        restart_last_available_sample = getattr(globalVars, "restart_last_available_sample", None)
        restart_sample_reuse_active = restart_last_available_sample is not None

        for name, values in [
            ("globalVars.restart_x0", x0),
            ("globalVars.restart_df_prev", df_prev),
            ("globalVars.restart_subLb_prev", subLb_prev),
            ("globalVars.restart_subUb_prev", subUb_prev),
            ("globalVars.restart_beta", beta),
        ]:
            if len(values) != nd:
                raise ValueError(f"{name} length ({len(values)}) must match nd ({nd})")

        q = globalVars.restart_completed_iteration
        fir = 1
        x = x0.copy()
        globalVars.sampling_counter = globalVars.restart_sampling_counter

        if globalVars.mpss_option == "SHIFTING":
            xx_restart = zz + df_prev
        elif globalVars.mpss_option == "SCALING":
            xx_restart = zz * df_prev
        else:
            raise ValueError(f"Unknown mpss_option: {globalVars.mpss_option}. Supported options are 'SHIFTING' and 'SCALING'.")

        rsvl_obj_restart, rsvl_con_restart = generate_response_dataset(
            xx_restart, nSample, nSampleR_obj, NUM_OBJECTIVES, NUM_CONSTRAINTS, constraint_basis_info
        )
        if NUM_OBJECTIVES == 1:
            rsvl_obj_restart_array = rsvl_obj_restart
            weights_restart = [1.0]
        else:
            rsvl_obj_restart_array = rsvl_obj_restart
            weights_restart = weights

        def restart_obj_func_wrapper(x, config=None, zz=None, rsvl=None, rssc=None, weights=None):
            if zz is None or rsvl is None or rssc is None or weights is None:
                raise ValueError("zz, rsvl, rssc, weights parameters are required")
            return objective_manager.evaluate_scalarized(x, zz, rsvl, rssc, weights=weights, LR=1)

        obj_func_restart = partial(
            restart_obj_func_wrapper,
            config=GPCE_CONFIG_OBJECTIVE,
            zz=zz,
            rsvl=rsvl_obj_restart_array,
            rssc=rssc,
            weights=weights_restart,
        )
        objValue_prev, _ = obj_func_restart(df_prev)
        objValue_prev_subdomain, _ = obj_func_restart(x0)
        constraint_values_prev, _ = constraint_manager.evaluate_all(x0, zz, rsvl_con_restart, rssc, LR=1)

        print(f"Restart mode: completed iteration set to q={q}")
        print(f"Restart mode: sampling counter set to {globalVars.sampling_counter}")
        print(f"Restart mode: x0 set to {x0}")
        print(f"Restart mode: beta set to {beta}")
        print(f"Restart mode: objValue_prev_subdomain auto-computed from sample_{globalVars.sampling_counter} = {objValue_prev_subdomain}")
        if restart_last_available_sample is not None:
            print(f"Restart mode: reuse precomputed samples up to sample_{restart_last_available_sample}")


    # Iteration of subdomain problems
    while not convergence_flag:
        q = q + 1

        sizing_iteration_count = 0 # (formerly tmp)
        infeasible_attempt_count = 0  # (formerly cntSizing1)
        cntSizingFlag = False # (formerly cntSizing)
        #infeasible_interpolation_flag = False # whether infeasibleCheck is active; if active, convergence is not checked

        while not cntSizingFlag:
            sizing_iteration_count += 1

            # Reset the global variables
            globalVars.cntObj = 0 # reset the objective counter; cntObj=0 triggers the first-call computation
            globalVars.cntCon = 0 # reset the constraint counter; cntCon=0 triggers the first-call computation

            if (fir == 0):
                globalVars.cntObj = 1 
                globalVars.cntCon = 1

            if globalVars.cntObj == 0: # separate counters for managing multiple objectives
                objective_manager.reset_call_counts() # reset the internal counters of the objective manager

            if globalVars.cntCon == 0: # separate counters for managing multiple constraints
                constraint_manager.reset_call_counts() # reset the internal counter of each constraint
            
            if globalVars.mpss_option == "SHIFTING":
                xx = zz + x0  # add x0 to each sample (NumPy broadcasting)
            elif globalVars.mpss_option == "SCALING":
                xx = zz * x0  # multiply each sample by x0 (NumPy broadcasting)
            
            if not (q == 1 and sizing_iteration_count == 1): # except for the feasibility check at the first iteration
                print(f"generate samples in {x0}")
                reuse_existing_sample = False
                next_sample_idx = globalVars.sampling_counter + 1

                # On restart, existing sample files are reused if available,
                # and new samples are generated after the last available sample.
                if (
                    restart_sample_reuse_active
                    and
                    globalVars.restart_enabled
                    and restart_last_available_sample is not None
                    and next_sample_idx <= restart_last_available_sample
                ):
                    existing_sample_path = os.path.join(globalVars.filepath_samples, f"sample_{next_sample_idx}.csv")
                    if os.path.exists(existing_sample_path):
                        globalVars.sampling_counter = next_sample_idx
                        reuse_existing_sample = True
                        print(f"Restart mode: reusing precomputed sample_{next_sample_idx} at {existing_sample_path}")
                    else:
                        print(f"Restart mode: sample_{next_sample_idx} not found at {existing_sample_path}; generating new sample.")

                # Once the reuse range is exceeded, existing samples are no longer checked.
                if (
                    restart_sample_reuse_active
                    and restart_last_available_sample is not None
                    and next_sample_idx > restart_last_available_sample
                ):
                    restart_sample_reuse_active = False
                    print(
                        "Restart mode: sample reuse window exceeded; "
                        "disabling further precomputed sample checks."
                    )

                log_step_message(
                    f"generate samples in {x0}",
                    iteration=q,
                    sizing_iteration=sizing_iteration_count,
                    sample_number=next_sample_idx,
                    x0=x0
                )

                if not reuse_existing_sample:
                    generate_samples(xx, nSample, nSampleR_obj, NUM_OBJECTIVES, NUM_CONSTRAINTS, constraint_basis_info)
            rsvl_obj, rsvl_con = generate_response_dataset(xx, nSample, nSampleR_obj, NUM_OBJECTIVES, NUM_CONSTRAINTS, constraint_basis_info)
        
            # Update the responses of the objective manager
            if NUM_OBJECTIVES == 1:
                # Single objective: keep the same interface as the original objFunc
                rsvl_obj_array = rsvl_obj  # shape: (nSampleR_obj, 1)
                weights_local = [1.0]  # weight 1.0 for a single objective
            else:
                # Multiple objectives: rsvl_obj is already of shape (nSampleR_obj, NUM_OBJECTIVES)
                rsvl_obj_array = rsvl_obj  # shape set by generate_response_dataset
                weights_local = weights
                ## weights are defined separately in the parameters
            
            # Wrap the scalarized function of objective_manager with the same interface as objFunc
            def obj_func_wrapper(x, config=None, zz=None, rsvl=None, rssc=None):
                """Wrapper with the same interface as objFunc"""
                # config is accepted for compatibility but not used (objective_manager manages its own config)
                if zz is None or rsvl is None or rssc is None:
                    raise ValueError("zz, rsvl, rssc parameters are required")
                return objective_manager.evaluate_scalarized(x, zz, rsvl, rssc, weights=weights_local, LR=1)
            
            # Same partial pattern as before
            obj_func_partial = partial(obj_func_wrapper, config=GPCE_CONFIG_OBJECTIVE, zz=zz, rsvl=rsvl_obj_array, rssc=rssc)
            objValue, _ = obj_func_partial(x0) # objective value (based on the responses of the current subdomain)
            

            
            # Constraint values (based on the responses of the current subdomain)
            constraint_values, _ = constraint_manager.evaluate_all(x0, zz, rsvl_con, rssc, LR=1)
            constraints_scipy = constraint_manager.get_constraint_functions_for_scipy(zz, rsvl_con, rssc, LR=1) # constraint list for SLSQP (sign flipped to +)

            fir += 1

            infeasibleCheck = False # (formerly chk)
            for con_value in constraint_values:
                if con_value > tol_constraint: # constraint violated
                    infeasibleCheck = True
                    # infeasible_interpolation_flag = True
                    break
            
            if infeasibleCheck: ## the current design is infeasible
                infeasible_attempt_count += 1
                x0 = df_prev*(1/gr) + x0*(1-1/gr)  # new x0 by golden-ratio interpolation
                df = x0.copy()
                infeasible_msg = f"!!!!!!!!! Infeasible design at iteration {q}, sizing iteration {sizing_iteration_count}, infeasible attempts {infeasible_attempt_count}, df: {df}"
                print(infeasible_msg)
                log_step_message(
                    infeasible_msg,
                    iteration=q,
                    sizing_iteration=sizing_iteration_count,
                    sample_number=globalVars.sampling_counter,
                    x0=x0,
                    additional_info=f"constraint_values: {constraint_values}"
                )
                print(f"constraint values: {constraint_values}")
            else:
                df = x0.copy()
                cntSizingFlag = True  # the current design is feasible
            
            if infeasible_attempt_count >= infeasible_attempt_limit:  # number of consecutive infeasible points
                print("Searching feasible designs fails")
                cntSizingFlag = True  # termination
            
            if (sizing_iteration_count == 1) and (q != 1):
                err1 = np.sqrt((objValue - objValue_prev_subdomain) ** 2)
                # c: current constraint vector, cLast: previous constraint vector
                c = np.array(constraint_values)
                c_prev = np.array(constraint_values_prev)
                err2 = np.sqrt((c - c_prev) ** 2) # err2 is a vector

                ## Step 6-0: feasibility check, whether the constraint surrogates predict the failure domain consistently
                current_feasible = c <= tol_constraint
                previous_feasible = c_prev <= tol_constraint

                # Check feasibility consistency for each constraint
                feasibility_mismatch = (
                    current_feasible != previous_feasible
                )

                if np.any(feasibility_mismatch):
                    beta = beta / gr

                else:
                    # Step 6-1
                    if (err1 <= epsilon3 and np.all(err2 <= epsilon3)): # Step 6-1
                        # increase overall sub-region size
                        beta = beta * (2.0 - 1.0/gr)

                    # Step 6-2
                    elif (err1 >= epsilon4 or np.any(err2 >= epsilon4)): # Step 6-2
                        # decrease overall sub-region size
                        beta = beta / gr

                for i in range(nd):
                    # Is constraint inequality active? 
                    chkL = np.sqrt((x[i] - subLb_prev[i]) ** 2)
                    chkU = np.sqrt((x[i] - subUb_prev[i]) ** 2)
                    if (chkL < epsilon5) or (chkU < epsilon5): # Step 6-3
                        beta[i] = beta[i] * (2.0 - 1.0/gr)

                    err3 = np.sqrt((df[i] - df_prev[i]) ** 2)
                    if (err3 < epsilon6): # Step 6-4
                        beta[i] = beta[i] / gr

                    if (beta[i] < epsilon7): # Step 6-5
                        beta[i] = epsilon7 

                    if (beta[i] > 1):
                        beta[i] = 1 # upper limit of beta (differs from the original MPSS paper)
            

        for i in range(nd):

            subLb[i] = (df[i] - beta[i] * (upper_bounds[i] - lower_bounds[i]) / 2)

            subUb[i] = (df[i] + beta[i] * (upper_bounds[i] - lower_bounds[i]) / 2)

            # restraint to the original design bounds
            subLb[i] = (
                lower_bounds[i]
                if subLb[i] < lower_bounds[i]
                else subLb[i]
            )

            subUb[i] = (
                upper_bounds[i]
                if subUb[i] > upper_bounds[i]
                else subUb[i]
            )

        # # set sub-domain location
        # for i in range(nd):
        #     if (q != 1):
        #         if (abs(x[i] - subLb_prev[i]) < tol_sub):
        #             subLb[i] = df[i] - beta[i] * (upper_bounds[i] - lower_bounds[i]) * gr / (gr + 1)
        #             subUb[i] = df[i] + beta[i] * (upper_bounds[i] - lower_bounds[i]) / (gr + 1)
        #         else:
        #             subLb[i] = df[i] - beta[i] * (upper_bounds[i] - lower_bounds[i]) / 2
        #             subUb[i] = df[i] + beta[i] * (upper_bounds[i] - lower_bounds[i]) / 2

        #         if (abs(x[i] - subUb_prev[i]) < tol_sub):
        #             subLb[i] = df[i] - beta[i] * (upper_bounds[i] - lower_bounds[i]) / (gr + 1)
        #             subUb[i] = df[i] + beta[i] * (upper_bounds[i] - lower_bounds[i]) * gr / (gr + 1)
        #         else:
        #             subLb[i] = df[i] - beta[i] * (upper_bounds[i] - lower_bounds[i]) / 2
        #             subUb[i] = df[i] + beta[i] * (upper_bounds[i] - lower_bounds[i]) / 2
            
        #     else: # (q == 1)
        #         subLb[i] = df[i] - beta[i] * (upper_bounds[i] - lower_bounds[i]) / 2
        #         subUb[i] = df[i] + beta[i] * (upper_bounds[i] - lower_bounds[i]) / 2

            # # restraint to the b.c.
            # subLb[i] = lower_bounds[i] if subLb[i] < lower_bounds[i] else subLb[i]
            # subUb[i] = upper_bounds[i] if subUb[i] > upper_bounds[i] else subUb[i]
        
        
        # Set the subdomain bounds
        bounds_mpss = [[subLb[i], subUb[i]] for i in range(nd)]

        print(f"Sub-region upper bounds: {subUb}")
        print(f"Sub-region lower bounds: {subLb}")


        # Convergence check
        # only for q > 1 (df_prev is not meaningful at the first iteration)
        if q > 1:
            terminate1 = np.sqrt(np.dot((df - df_prev), (df - df_prev)))
            terminate2 = np.sqrt((objValue - objValue_prev) ** 2)
        else:
            terminate1 = float('inf')  # no convergence at the first iteration
            terminate2 = float('inf')

        if (terminate1 < epsilon1):
            convergence_flag = True
            print("terminate1 condition")
            print(f"df_prev: {df_prev}, df: {df}")
        if (terminate2 < epsilon2):
            convergence_flag = True
            print("terminate2 condition")
            print(f"objValue: {objValue}, objValue_prev: {objValue_prev}")
        
        objValue_prev = objValue

        # Start timing the optimization
        import time
        opt_start_time = time.time()
        
        if convergence_flag:
            history['x'] = df.copy()
            history['fval'] = objValue
            
            print("Optimization is completed")
            break
        else:
            # Single-step optimization
            globalVars.cntObj = 1
            globalVars.cntCon = 1
            
            # Select the optimization method
            if globalVars.opt_method_type == "GA":
                print("Using GA (Genetic Algorithm) optimization...")
                res = run_ga_optimization(
                    obj_func_partial, constraint_manager, x0, bounds_mpss, 
                    zz, rsvl_con, rssc, nd, NUM_OBJECTIVES, NUM_CONSTRAINTS
                )

            elif globalVars.opt_method_type == "SLSQP":
                print("Using SLSQP optimization...")
                # local RDO problem
                options = {'disp': 1} # optimization options

                # Call minimize
                res = minimize(obj_func_partial, x0, method='SLSQP', jac=True,
                            bounds=bounds_mpss, constraints=constraints_scipy,
                            options=options)
                
            elif globalVars.opt_method_type == "DE":
                print("Using Differential Evolution optimization...")
                from scipy.optimize import NonlinearConstraint
                
                # Differential evolution options
                de_options = {
                    'disp': True,           # print progress
                    'maxiter': 1000,        # maximum number of generations
                    'popsize': 15,          # population-size multiplier (population = popsize * len(x))
                    'tol': 1e-3,            # relative convergence tolerance
                    'atol': 0,              # absolute tolerance
                    'seed': None,           # random seed (can be set for reproducibility)
                    'workers': 1,           # number of parallel workers (1 = sequential)
                    'updating': 'immediate', # 'immediate' or 'deferred'
                    'polish': True          # polish the final result with L-BFGS-B
                }
                
                # Objective wrapper (DE does not use gradients)
                def objective_for_de(x_input):
                    obj_val, _ = obj_func_partial(x_input)
                    return obj_val
                
                # Constraint wrapper for DE (returns a vector)
                def constraint_for_de(x_input):
                    # all constraint values from constraint_manager (h(x) <= 0 form)
                    con_vals, _ = constraint_manager.evaluate_all(x_input, zz, rsvl_con, rssc, LR=1)
                    return np.array(con_vals)
                
                # NonlinearConstraint object (h(x) <= 0, i.e., -inf <= h(x) <= 0)
                de_constraints = NonlinearConstraint(
                    constraint_for_de,
                    -np.inf,  # lower bound
                    0.0       # upper bound (h(x) <= 0)
                )
                
                # Run differential_evolution
                res = differential_evolution(
                    objective_for_de,
                    bounds=bounds_mpss,
                    constraints=de_constraints,  # NonlinearConstraint
                    x0=x0,  # initial guess (optional for DE)
                    **de_options
                )
            else:
                raise RuntimeError("Unsupported optimization method type.")

            # Stop timing the optimization
            opt_end_time = time.time()
            opt_elapsed_time = opt_end_time - opt_start_time
            print(f"Optimization iteration {q} completed in {opt_elapsed_time:.2f} seconds")

            x = res.x.copy()  # optimization result
            objValue_prev_subdomain, _ = obj_func_partial(x)
            constraint_values_prev, _ = constraint_manager.evaluate_all(x, zz, rsvl_con, rssc, LR=1)

            # Update for the next iteration (after the convergence check)
            df_prev = df.copy()  # store the current df as df_prev of the next iteration
            x0 = x.copy()
            subLb_prev = subLb.copy()
            subUb_prev = subUb.copy()
        
        # Logging at the end of the while loop
        log_optimization_step(x0, objValue_prev_subdomain, constraint_values_prev, q, beta, subLb, subUb, df, df_prev, res)

        
        # Store the results (safe access)
        try:
            history['x'] = res.x
            history['fval'] = res.fun
            # nfev and njev may differ for GA, so handle them safely
            globalVars.cntObj = getattr(res, 'nfev', globalVars.cntObj)
            globalVars.cntCon = getattr(res, 'njev', globalVars.cntCon)
            
        except (NameError, AttributeError):
            # convergence_flag is True from the start or the attribute does not exist
            history['x'] = x0
            history['fval'] = objValue
            # Number of function calls from the global variables
            globalVars.cntObj = globals().get('cntObj', globalVars.cntObj)
            globalVars.cntCon = globals().get('cntCon', globalVars.cntCon)

        # When q == 1 is completed, save the obj0 values of each objective to CSV
        if q == 1:
            try:
                csv_filepath = objective_manager.save_obj0_values_to_csv(iteration_q=q)
                print(f"obj0 values for q={q} saved successfully to {csv_filepath}")
            except Exception as e:
                print(f"Error saving obj0 values: {e}")
        
    return history, searchdir  ##### !!!!!!!! searchdir is currently an empty list
    

    
