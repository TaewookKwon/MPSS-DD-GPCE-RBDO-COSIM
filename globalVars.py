import os
import sys

global cntObj
global cntCon
#global cntRspObj
#global cntRspCon
global stat0
global statf 
global sopt 
global iter_obj 
global iter_con
global valObj
global vObjGrad
#global cntMinimize

global filepath
global filepath_samples
global method_type
global opt_method_type
global mpss_option
global first_samples
global sampling_counter

global objective_log
global constraint_log

global restart_enabled
global restart_completed_iteration
global restart_sample_folder
global restart_sampling_counter
global restart_last_available_sample
global restart_x0
global restart_df_prev
global restart_subLb_prev
global restart_subUb_prev
global restart_beta
global restart_objvalue_prev_subdomain

global initial_val_control
global obj_intial
global con_intial
global skipped_sample_indices
# global objective_function_initialized ## temporary


cntObj = 0
cntCon = 0
#cntRspObj = 0
#cntRspCon = 0
stat0 = 0
statf = 0
sopt = 'hello'
iter_obj = 0
iter_con = 0
valObj = 0
vObjGrad = 0

# cntMinimize = 0
filepath_samples = None
sampling_counter = 0



## Control parameters ##
filepath = os.path.dirname(os.path.abspath(__file__))  # Project root directory (the folder containing this file)
method_type = "RBDO" # "RDO", "RBDO"  / problem type
opt_method_type = "DE" #"SLSQP", "DE", "GA", "NSGA2", "NSGA3"  / optimizer for the local subproblems
mpss_option = "SCALING" # "SHIFTING", "SCALING"  / MPSS transformation (SCALING: X = Z * d)
first_samples = "yes" #"yes", "no" / "yes": use the stored initial samples in ./initial instead of running new analyses

objective_log = False # print a log line at each objective evaluation
constraint_log = False # print a log line at each constraint evaluation



skipped_sample_indices = [] # indices of skipped (failed) high-fidelity samples

############################### Restart Parameters ###############################################

## Restart is applied after the single-step optimum of the last completed iteration, i.e., from the start of the next iteration
restart_enabled = False

initial_val_control = False # True: use obj_initial for normalization / False: compute it from the initial samples
obj_initial = [3.495669695649757] # initial objective value (mean mass, kg) used for normalization

restart_completed_iteration = 2 # number of iterations completed before the restart
restart_sample_folder = filepath + "/samples/260818_204038"
restart_sampling_counter = 2 # sample index at which the restart begins
restart_last_available_sample = 4  # last existing sample file to reuse (reuse up to sample_n)
restart_x0 = [
    28.943369565924993,
    24.40232863755116,
    60.00567317205376,
    8.113850084969764,
    70.00000011636277,
    304.99999980352203,
]
restart_df_prev = [
    33.05758645581085,
    32.003353796728035,
    65.02255604834231,
    6.081030945929681,
    69.99999967539537,
    305.00000009549217,
]
restart_subLb_prev = [
    28.90823350703241,
    24.364033571725933,
    60.0,
    4.008081929054524,
    69.99999908522543,
    304.99999950532225,
]
restart_subUb_prev = [
    36.01,
    39.64267402173014,
    71.93238610459284,
    8.15397996280484,
    70.00000026556532,
    305.0000006856621,
]
restart_beta = [
    0.6909830056250525,
    0.9549150281252629,
    0.6909830056250525,
    0.6909830056250525,
    0.5901699437494742,
    0.5901699437494742,
]
restart_objvalue_prev_subdomain = None  # computed automatically from the restart samples


