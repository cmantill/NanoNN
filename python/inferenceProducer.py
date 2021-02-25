import os
import uproot
import awkward
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import numpy as np
from collections import Counter
from uproot_methods import TLorentzVectorArray

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from PhysicsTools.NanoNN.makeInputs import ParticleNetTagInfoMaker
from PhysicsTools.NanoNN.runPrediction import ParticleNetJetTagsProducer
from PhysicsTools.NanoNN.nnHelper import convert_prob

class InferenceProducer(Module):

     def __init__(self):
          self.tagInfoMaker = ParticleNetTagInfoMaker(fatjet_branch='FatJet', pfcand_branch='PFCands', sv_branch='SV', fatpfcand_branch='FatJetPFCands', jetR=0.8)
          self.pnTaggerMD = ParticleNetJetTagsProducer(
               os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetAK8/MD-2prong/V00/preprocess.json'),
               os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetAK8/MD-2prong/V00/ParticleNetMD.onnx'),
          )

     def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree, entriesRange=None):
          self.out = wrappedOutputTree
          self.out.branch("FatJet_pnXqqVsQCD", "F", lenVar="nFatJet")
          self._uproot_tree = uproot.open(inputFile.GetName())['Events']

     def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
          pass
        
     def analyze(self, event, ievent):
          absolute_event_idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
          jets = Collection(event, "FatJet")
          if len(jets)>0:
               table = self._uproot_tree.arrays(['FatJet_pt','FatJet_eta', 'FatJet_phi', 'FatJet_mass', 'FatJet_msoftdrop', '*FatJetPFCands*', 'PFCands*', 'SV*'],
                                                namedecode='utf-8', entrystart=absolute_event_idx, entrystop=absolute_event_idx+1) 
               tagInfo = self.tagInfoMaker.convert(table)
               outputs = self.pnTaggerMD.predict(tagInfo)
               pn_XqqVsQCD = convert_prob(outputs, ['Xqq'], prefix='prob')
               self.out.fillBranch("FatJet_pnXqqVsQCD", pn_XqqVsQCD[0])
          return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
inferenceProduder = lambda : InferenceProducer()
