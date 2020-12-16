from kfac.kfac_preconditioner import KFACParamScheduler
from kfac.kfac_preconditioner import KFAC as KFAC_EIGEN
from kfac.kfac_preconditioner_inv import KFAC as KFAC_INV
from kfac.kfac_preconditioner_inv_opt import KFAC as KFAC_INV_OPT
from kfac.kfac_preconditioner_opt import KFAC as KFAC_EIGEN_OPT
from kfac.kfac_preconditioner_small_ag import KFAC as KFAC_INV_SMALL_AG
KFAC = KFAC_EIGEN_OPT

kfac_mappers = {
    'eigen': KFAC_EIGEN,
    'eigen_opt': KFAC_EIGEN_OPT,
    'inverse': KFAC_INV,
    'inverse_opt': KFAC_INV_OPT,
    'inv_small_ag': KFAC_INV_SMALL_AG,
        }

def get_kfac_module(kfac='eigen'):
    return kfac_mappers[kfac]
