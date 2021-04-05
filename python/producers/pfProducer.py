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

class pfProducer(Module):

     def __init__(self):
          self.nJets = 2
          self.tagInfoMaker = ParticleNetTagInfoMaker(fatjet_branch='FatJet', pfcand_branch='PFCands', sv_branch='SV', fatpfcand_branch='FatJetPFCands', jetR=0.8)
          self.pnTagger = ParticleNetJetTagsProducer(
               os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetHWW/input/V01/preprocess.json'),
          )
          self.n_pf = 50
          self.pf_names = ["pfcand_pt_log_nopuppi",
                           "pfcand_e_log_nopuppi",
                           "pfcand_etarel",
                           "pfcand_phirel"]
          self.jet_r2 = 0.8 * 0.8

     def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree, entriesRange=None):
          self.tagInfoLen = 0
          self.fetch_step = 1000
          self.tagInfoMaker.init_file(inputFile, fetch_step=self.fetch_step)
          self.tagInfo = None

          self.out = wrappedOutputTree
          self.out.branch("fj_isQCD", "I", 1)
          self.out.branch("fj_isTop", "I", 1)
          self.out.branch("fj_isHiggs", "I", 1)
          for key in self.pf_names:
               self.out.branch(key, "F", self.n_pf)

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
                    isHiggs = self.tagInfo['_isHiggs']
                    isTop = self.tagInfo['_isTop']
                    if(isHiggs==0 and isTop==0): isQCD = 1
                    else: isQCD = 0
                    self.out.fillBranch("fj_isQCD", isQCD)
                    self.out.fillBranch("fj_isTop", isTop)
                    self.out.fillBranch("fj_isHiggs", isHiggs)

                    for key in self.pf_names:
                         self.out.fillBranch(key, outputs['pf_features'][key])

                    self.out.fill()
                    nevt+=1

          return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
inputProduder = lambda : InputProducer()
