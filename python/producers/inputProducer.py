import os
import uproot
import awkward
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import numpy as np
from collections import Counter

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from PhysicsTools.NanoNN.helpers.makeInputs import ParticleNetTagInfoMaker
from PhysicsTools.NanoNN.helpers.runPrediction import ParticleNetJetTagsProducer

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
          self.tagInfoLen = 0
          self.fetch_step = 1000
          self.tagInfoMaker.init_file(inputFile, fetch_step=self.fetch_step)
          self.tagInfo = None

          self.out = wrappedOutputTree
          self.out.branch("fj_idx", "I", 1)
          self.out.branch("fj_pt", "F", 1)
          self.out.branch("fj_eta", "F", 1)
          self.out.branch("fj_phi", "F", 1)
          self.out.branch("fj_mass", "F", 1)
          self.out.branch("fj_msoftdrop", "F", 1)
          self.out.branch("fj_lsf3", "F", 1)
          self.out.branch("fj_deepTagMD_H4qvsQCD", "F", 1)
          self.out.branch("fj_deepTag_HvsQCD", "F", 1)
          self.out.branch("fj_PN_H4qvsQCD", "F", 1)

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
               self.tagInfo, self.tagInfoLen = self.tagInfoMaker.load(absolute_event_idx, self.tagInfoLen, True)
               if self.tagInfo is None: 
                    return False
               else: return True
          else:
               return False

     def fill(self, event, ievent):
          absolute_event_idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
          skip = -1
          if(absolute_event_idx<self.fetch_step):
               ieventTag = ievent
          else:
               ieventTag = ievent - self.tagInfoLen
          # print('iev ',ievent, ' taginfo ',self.tagInfoLen, ' abs ',absolute_event_idx)
          # print('evt tag ',ieventTag)
          nevt = 0
          for idx, fj in enumerate(event._allFatJets):
               #print('fj pt bef ',fj.pt, 'msof ',fj.msoftdrop)
               if idx>1 : continue
               if (fj.pt <= 300 or fj.msoftdrop <= 20):
                    skip = idx;
                    continue
               else:
                    if skip==-1: jidx = idx
                    else: jidx = skip
               fj.idx = jidx
               fj = event._allFatJets[idx]
               # print('fj pt ',fj.pt, self.tagInfo['_jetp4'].pt[ieventTag])
               outputs = self.pnTagger.pad_one(self.tagInfo, ieventTag, jidx)
               if outputs:
                    self.out.fillBranch("fj_idx", fj.idx)
                    self.out.fillBranch("fj_pt", fj.pt)
                    self.out.fillBranch("fj_eta", fj.eta)
                    self.out.fillBranch("fj_phi", fj.phi)
                    self.out.fillBranch("fj_mass", fj.mass)
                    self.out.fillBranch("fj_msoftdrop", fj.msoftdrop)
                    self.out.fillBranch("fj_lsf3", fj.lsf3)
                    self.out.fillBranch("fj_deepTagMD_H4qvsQCD", fj.deepTagMD_H4qvsQCD)
                    self.out.fillBranch("fj_deepTag_HvsQCD", self.tagInfo["_jet_deepTagHvsQCD"][ieventTag][jidx])
                    self.out.fillBranch("fj_PN_H4qvsQCD", fj.particleNet_H4qvsQCD)

                    isHiggs = self.tagInfo['_isHiggs']
                    isTop = self.tagInfo['_isTop']
                    if(isHiggs==0 and isTop==0): isQCD = 1
                    else: isQCD = 0
                    self.out.fillBranch("fj_isQCD", isQCD)
                    self.out.fillBranch("fj_isTop", isTop)

                    self.out.fillBranch("fj_H_WW_4q", self.tagInfo["_jet_H_WW_4q"][ieventTag][jidx])
                    self.out.fillBranch("fj_H_WW_elenuqq", self.tagInfo["_jet_H_WW_elenuqq"][ieventTag][jidx])
                    self.out.fillBranch("fj_H_WW_munuqq", self.tagInfo["_jet_H_WW_munuqq"][ieventTag][jidx])
                    self.out.fillBranch("fj_nProngs", self.tagInfo["_jet_nProngs"][ieventTag][jidx])
                    self.out.fillBranch("fj_dR_W", self.tagInfo["_jet_dR_W"][ieventTag][jidx])
                    self.out.fillBranch("fj_dR_Wstar", self.tagInfo["_jet_dR_Wstar"][ieventTag][jidx])

                    for key in self.pf_names:
                         self.out.fillBranch(key, outputs['pf_features'][key])
                    for key in self.sv_names:
                         self.out.fillBranch(key, outputs['sv_features'][key])

                    self.out.fill()
                    nevt+=1

          return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
inputProduder = lambda : InputProducer()
