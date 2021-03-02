import os
import uproot
import awkward
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import numpy as np
from collections import Counter

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from PhysicsTools.NanoNN.makeInputs import ParticleNetTagInfoMaker
from PhysicsTools.NanoNN.runPrediction import ParticleNetJetTagsProducer

class InputProducer(Module):

     def __init__(self):
          self.nJets = 2
          self.tagInfoMaker = ParticleNetTagInfoMaker(fatjet_branch='FatJet', pfcand_branch='PFCands', sv_branch='SV', fatpfcand_branch='FatJetPFCands', jetR=0.8)
          self.pnTagger = ParticleNetJetTagsProducer(
               os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetHWW/input/V01/preprocess.json'),
          )
          self.n_pf = self.pnTagger.prep_params['pf_features']['var_length']
          self.pf_names = self.pnTagger.prep_params['pf_features']['var_names']
          self.n_sv = self.pnTagger.prep_params['sv_features']['var_length']
          self.sv_names = self.pnTagger.prep_params['sv_features']['var_names']          
          self.jet_r2 = 0.8 * 0.8
          
     def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree, entriesRange=None):
          self._uproot_tree = uproot.open(inputFile.GetName())['Events']

          self.out = wrappedOutputTree
          self.out.branch("fj_idx", "I", 1)
          self.out.branch("fj_pt", "F", 1)
          self.out.branch("fj_eta", "F", 1)
          self.out.branch("fj_phi", "F", 1)
          self.out.branch("fj_mass", "F", 1)
          self.out.branch("fj_lsf3", "F", 1)
          self.out.branch("fj_deepTagMD_H4qvsQCD", "F", 1)
          self.out.branch("fj_deepTag_HvsQCD", "F", 1)

          self.out.branch("fj_isQCD", "I", 1)
          self.out.branch("fj_isTop", "I", 1)
          self.out.branch("fj_H_WW_4q", "I", 1) 
          self.out.branch("fj_H_WW_elenuqq", "I", 1)
          self.out.branch("fj_H_WW_munuqq", "I", 1)
          self.out.branch("fj_nProngs", "I", 1)
          self.out.branch("fj_dR_W", "F", 1)
          self.out.branch("fj_dR_Wstar", "F", 1)

          for key in self.pf_names:
               self.out.branch(key, "F", self.n_pf)
          for key in self.sv_names:
               self.out.branch(key, "F", self.n_sv)

     def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
          pass

     def analyze(self, event, ievent):
          absolute_event_idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
          event._allFatJets = Collection(event, "FatJet")
          if len(event._allFatJets)>0: 
               table = self._uproot_tree.arrays(['FatJet_pt','FatJet_eta', 'FatJet_phi', 'FatJet_mass', 
                                                 'FatJet_msoftdrop','FatJet_deepTag_H','FatJet_deepTag_QCD','FatJet_deepTag_QCDothers',
                                                 '*FatJetPFCands*', 'PFCands*', 'SV*',
                                                 'GenPart_*'],
                                                namedecode='utf-8', entrystart=absolute_event_idx, entrystop=absolute_event_idx+1) 
               self.tagInfo = self.tagInfoMaker.convert(table,True)
               if self.tagInfo is None: return False
               else: return True
          else:
               return False

     def fill(self, event, ievent):
          absolute_event_idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
          skip = -1
          for idx, fj in enumerate(event._allFatJets):
               if idx>1 : continue
               if (fj.pt < 300 or fj.msoftdrop < 20): 
                    skip = idx;
                    continue
               else:
                    if skip==-1: jidx = idx
                    else: jidx = skip
               fj.idx = jidx
               fj = event._allFatJets[idx]
               outputs = self.pnTagger.pad_one(self.tagInfo, jidx)
               if outputs:
                    self.out.fillBranch("fj_idx", fj.idx)
                    self.out.fillBranch("fj_pt", fj.pt)
                    self.out.fillBranch("fj_eta", fj.eta)
                    self.out.fillBranch("fj_phi", fj.phi)
                    self.out.fillBranch("fj_mass", fj.mass)
                    self.out.fillBranch("fj_lsf3", fj.lsf3)
                    self.out.fillBranch("fj_deepTagMD_H4qvsQCD", fj.deepTagMD_H4qvsQCD)
                    self.out.fillBranch("fj_deepTag_HvsQCD", self.tagInfo["_jet_deepTagHvsQCD"][0][jidx])
                                                                                                                                                               
                    isHiggs = self.tagInfo['_isHiggs']
                    isTop = self.tagInfo['_isTop']
                    if(isHiggs==0 and isTop==0): isQCD = 1
                    else: isQCD = 0
                    self.out.fillBranch("fj_isQCD", isQCD)
                    self.out.fillBranch("fj_isTop", isTop)

                    self.out.fillBranch("fj_H_WW_4q", self.tagInfo["_jet_H_WW_4q"][0][jidx])
                    self.out.fillBranch("fj_H_WW_elenuqq", self.tagInfo["_jet_H_WW_elenuqq"][0][jidx])
                    self.out.fillBranch("fj_H_WW_munuqq", self.tagInfo["_jet_H_WW_munuqq"][0][jidx])
                    self.out.fillBranch("fj_nProngs", self.tagInfo["_jet_nProngs"][0][jidx])
                    self.out.fillBranch("fj_dR_W", self.tagInfo["_jet_dR_W"][0][jidx])
                    self.out.fillBranch("fj_dR_Wstar", self.tagInfo["_jet_dR_Wstar"][0][jidx])

                    for key in self.pf_names:
                         self.out.fillBranch(key, outputs['pf_features'][key])
                    for key in self.sv_names:
                         self.out.fillBranch(key, outputs['sv_features'][key])

                    self.out.fill()
          return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
inputProduder = lambda : InputProducer()
