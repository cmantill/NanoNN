import os
import uproot
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import numpy as np
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from PhysicsTools.NanoNN.makeInputs import ParticleNetTagInfoMaker
from PhysicsTools.NanoNN.runPrediction import ParticleNetJetTagsProducer
from PhysicsTools.NanoNN.nnHelper import convert_prob


class TreeProducer(Module):

    def __init__(self):
        self.tagInfoMaker = ParticleNetTagInfoMaker(fatjet_branch='FatJet', pfcand_branch='PFCands', sv_branch='SV', jetR=0.8)
        self.pnTaggerMD = ParticleNetJetTagsProducer(
            os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetAK8/MD-2prong/V00/ParticleNetMD.onnx'),
            os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetAK8/MD-2prong/V00/preprocess.json'),
        )

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self._uproot_tree = uproot.open(inputFile.GetName())['Events']
        self._tagInfo = None

        # build table for all entries
        table = self._uproot_tree.arrays(
            ['FatJet_pt','FatJet_eta', 'FatJet_phi', 'FatJet_mass', '*FatJetPFCands*', 'PFCands*', 'SV*'],
            namedecode='utf-8')
        self._tagInfo = self.tagInfoMaker.convert(table)

        # predict for selected entries
        self.out = wrappedOutputTree

        # now run over all events and all jets in entriesRange
        _runParticleNet(entriesRange)


    def _runParticleNet(self, entriesRange):
        if self._tagInfo is not None:
            # run prediction
            entry_idx = absolute_event_idx - self._uproot_start
            # uncomment these lines for consistency check
            p4 = self._tagInfo['_jetp4'][entry_idx][jet.idx]
            print('pt,eta,phi', (jet.pt, jet.eta, jet.phi), (p4.pt, p4.eta, p4.phi))
            if jet:
                outputs = self.pnTaggerMD.predict_one(self._tagInfo, entry_idx, jet.idx)
                jet.pn_XqqVsQCD = convert_prob(outputs, ['Xqq'], prefix='prob')
            else:
                outputs = self.pnTaggerMD.predict(self._tagInfo)
                pn_XqqVsQCD = convert_prob(outputs, ['Xqq'], prefix='prob')
        else:
            jet.pn_XqqVsQCD = -1000

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        event._allFatJets = Collection(event, "FatJet")
        pn_XbbVsQCD = []
        pn_XqqVsQCD  = []
        pn_H4qVsQCD = []
        pnv0_H4qVsQCD = []
        pnv0_HlnuqqVsQCD = []

        fj_XbbVsQCD,fj_XqqVsQCD,fj_H4qVsQCD = self._runParticleNet(event)

        def toArray(arr, pred):
            for i in pred: arr.append(float(i))
        toArray(pn_XqqVsQCD,fj_XqqVsQCD)
        self.out.fillBranch("FatJet_pnXqqVsQCD", pn_XqqVsQCD)

        return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
treeProducer = lambda : TreeProducer()
