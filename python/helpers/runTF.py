import os
import numpy as np
import json
import traceback
import logging
import ROOT as r

from keras.models import load_model
import tensorflow as tf

def configLogger(name, loglevel=logging.INFO, filename=None):
    # define a Handler which writes INFO messages or higher to the sys.stderr                                                                                                                                                                                                   
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    console = logging.StreamHandler()
    console.setLevel(loglevel)
    console.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(console)
    if filename:
        logfile = logging.FileHandler(filename)
        logfile.setLevel(loglevel)
        logfile.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s'))
        logger.addHandler(logfile)

logger = logging.getLogger('NanoNN')
configLogger('NanoNN', loglevel=logging.INFO)

class INJetTagsProducer(object):

    def __init__(self, preprocess_path, model_path=None, version=None):
        if version is not None:
            model_path = model_path.format(version=version)
        with open(preprocess_path) as fp:
            self.prep_params = json.load(fp)

        self.n_pf = self.prep_params['pf_features']['var_length']
        self.pf_names = self.prep_params['pf_features']['var_names']
        self.n_sv = self.prep_params['sv_features']['var_length']
        self.sv_names = self.prep_params['sv_features']['var_names']

        self._load_model(model_path)

    def _load_model(self,model_path):
        # receiving matrix                                                                                                                                                                                                                                                    
        RR=[]
        for i in range(self.n_pf):
            row=[]
            for j in range(self.n_pf*(self.n_pf-1)):
                if j in range(i*(self.n_pf-1),(i+1)*(self.n_pf-1)):
                    row.append(1.0)
                else:
                    row.append(0.0)
            RR.append(row)
        RR=np.array(RR)
        RR=np.float32(RR)
        RRT=np.transpose(RR)
        
        # sending matrix                                                                                                                                                                                                                                                      
        RST=[]
        for i in range(self.n_pf):
            for j in range(self.n_pf):
                row=[]
                for k in range(self.n_pf):
                    if k == j:
                        row.append(1.0)
                    else:
                        row.append(0.0)
                RST.append(row)
        rowsToRemove=[]
        for i in range(self.n_pf):
            rowsToRemove.append(i*(self.n_pf+1))
        RST=np.array(RST)
        RST=np.float32(RST)
        RST=np.delete(RST,rowsToRemove,0)
        RS=np.transpose(RST)

        # receiving matrix for the bipartite particle and secondary vertex graph                                                                                                                                                                                              
        RK=[]
        for i in range(self.n_pf):
            row=[]
            for j in range(self.n_pf*self.n_sv):
                if j in range(i*self.n_sv,(i+1)*self.n_sv):
                    row.append(1.0)
                else:
                    row.append(0.0)
            RK.append(row)
        RK=np.array(RK)
        RK=np.float32(RK)
        RKT=np.transpose(RK)
        
        # sending matrix for the bipartite particle and secondary vertex graph                                                                                                                                                                                                
        RV=[]
        for i in range(self.n_sv):
            row=[]
            for j in range(self.n_pf*self.n_sv):
                if j % self.n_sv == i:
                    row.append(1.0)
                else:
                    row.append(0.0)
            RV.append(row)
        RV=np.array(RV)
        RV=np.float32(RV)
        RVT=np.transpose(RV)

        logger.info('Loading model %s' % model_path)        
        self.model = load_model(model_path, custom_objects={'tf': tf,'RK': RK,'RV': RV,'RS': RS,'RR': RR,'RRT': RRT,'RKT': RKT})

if __name__ == '__main__':
    import time
    import uproot
    import argparse
    parser = argparse.ArgumentParser('TEST')
    parser.add_argument('-i', '--input')
    parser.add_argument('-m', '--model')
    parser.add_argument('-p', '--preprocess')
    args = parser.parse_args()

    nn = INJetTagsProducer(args.preprocess, args.model)
