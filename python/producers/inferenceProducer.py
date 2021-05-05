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

from PhysicsTools.NanoNN.helpers.makeInputs import ParticleNetTagInfoMaker
from PhysicsTools.NanoNN.helpers.runPrediction import ParticleNetJetTagsProducer
from PhysicsTools.NanoNN.helpers.nnHelper import convert_prob

class InferenceProducer(Module):

     def __init__(self,jetType="AK8"):
          self.jetType = jetType
          self.jet_r = 0.8 if jetType=="AK8" else 1.5
          fatpfcand_branch = "FatJetPFCands" if jetType=="AK8" else "JetPFCandsAK15"
          self.tagInfoMaker = ParticleNetTagInfoMaker(fatjet_branch='FatJet'+self.jetTag, pfcand_branch='PFCands', sv_branch='SV', fatpfcand_branch=fatpfcand_branch, jetR=self.jet_r)
          self._opts = {'WRITE_CACHE_FILE': False}

          self.tagger_versions = ['V01']
          prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data')
          self.pnTaggers = [ParticleNetJetTagsProducer(
                    '%s/ParticleNetHWW/{version}/%s/preprocess.json' % (prefix, self.jetType),
                    '%s/ParticleNetHWW/{version}/%s/particle-net.onnx' % (prefix, self.jetType),
                    version=ver, cache_suffix='tagger') for ver in self.tagger_versions]]

     def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree, entriesRange=None):
          for p in self.pnTaggers:
               p.load_cache(inputFile)

          self.tagInfoMaker.init_file(inputFile, fetch_step=1000)

          self.out = wrappedOutputTree
          self.out.branch("FatJet_pnHelenuqqVsQCD", "F", lenVar="nFatJet")
          self.out.branch("FatJet_pnHmunuqqVsQCD", "F", lenVar="nFatJet")
          self.out.branch("FatJet_pnHqqqqVsQCD", "F", lenVar="nFatJet")

     def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
          if self._opts['WRITE_CACHE_FILE']:
               for p in self.pnTaggers:
                    p.update_cache()

          for f in os.listdir('.'):
               if f.endswith('.h5'):
                    os.remove(f)

     def evalTagger(self, event, jets):
          for j in jets:
               outputs = [p.predict_with_cache(self.tagInfoMaker, event.idx, j.idx, j) for p in self.pnTaggers]
               outputs = ensemble(outputs, np.mean)
               j.pn_Xbb = outputs['probXbb']
               j.pn_Xcc = outputs['probXcc']
               j.pn_Xqq = outputs['probXqq']
               j.pn_QCD = convert_prob(outputs, None, prefix='prob')
            j.pn_XbbVsQCD = convert_prob(j, ['Xbb'], ['QCD'], prefix='pn_')
            j.pn_XccVsQCD = convert_prob(j, ['Xcc'], ['QCD'], prefix='pn_')
            j.pn_XccOrXqqVsQCD = convert_prob(j, ['Xcc', 'Xqq'], ['QCD'], prefix='pn_')

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
