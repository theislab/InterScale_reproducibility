import os

def load_paths():
    # LRZ home
    if os.path.exists("/dss/dsshome1/05/di93tig"):
        print('LRZ cluster')
        CLUSTER = 'LRZ'
        BASE_DIR_REPO = "/dss/dsshome1/05/di93tig/1_projects" 
        BASE_DIR_PROJECT = "/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale"
    elif os.path.exists("/home/icb/francesca.drummer/"):
        print('HPC cluster')
        CLUSTER = 'HPC'
        BASE_DIR_REPO = "/home/icb/francesca.drummer/1-Projects/"
        BASE_DIR_PROJECT = "/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer"
    else:
        print('unkown')
        CLUSTER = 'unknown'

    return BASE_DIR_REPO, BASE_DIR_PROJECT, 