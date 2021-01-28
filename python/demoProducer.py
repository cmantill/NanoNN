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
        self.pnTagger = ParticleNetJetTagsProducer(
            os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetAK8/General/V00/ParticleNet.onnx'),
            os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetAK8/General/V00/preprocess.json'),
        )
        self.pnTaggerHWW4q = ParticleNetJetTagsProducer(
            os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetHWW/v0/hww4qh5v0/model_hwwh5v04mfsel_ep19.onnx'),
            os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetHWW/v0/hww4qh5v0/preprocess.json'),
        )
        #self.pnTaggerHWWlnuqq = ParticleNetJetTagsProducer(
        #    os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetHWW/v0/hwwlnuqqh5v0/model_hwwh5v0hlnuqq_ep19.onnx'),
        #    os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetHWW/v0/hwwlnuqqh5v0/preprocess.json'),
        #)
        self.pnTaggerHWWlnuqq = ParticleNetJetTagsProducer(
            os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetHWW/v0/hwwlnuqqh5/model_hwwh5hlnuqq_ep21.onnx'),
            os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetHWW/v0/hwwlnuqqh5/preprocess.json'),
        )
    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        # set up uproot
        self._uproot_tree = uproot.open(inputFile.GetName())['Events']
        self._uproot_fetch_step = 1 #next(self._uproot_tree.clusters())[1]
        self._uproot_start = 0
        self._uproot_stop = 0
        self._tagInfo = None

        self.out = wrappedOutputTree
        self.out.branch("FatJet_pnXbbVsQCD", "F", lenVar="nFatJet")
        self.out.branch("FatJet_pnXqqVsQCD", "F", lenVar="nFatJet")
        self.out.branch("FatJet_pnH4qVsQCD", "F", lenVar="nFatJet")
        self.out.branch("FatJet_pnv0H4qVsQCD", "F", lenVar="nFatJet")
        self.out.branch("FatJet_pnv0HlnuqqVsQCD", "F", lenVar="nFatJet")

    def _runParticleNet(self, event, jet=None):
        absolute_event_idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
        if absolute_event_idx >= self._uproot_stop:
            # needs to fetch next batch
            self._uproot_start = absolute_event_idx
            self._uproot_stop = self._uproot_start + self._uproot_fetch_step
            table = self._uproot_tree.arrays(
                ['FatJet_pt',
                 'FatJet_eta', 'FatJet_phi', 'FatJet_mass', '*FatJetPFCands*', 'PFCands*', 'SV*'],
                namedecode='utf-8', entrystart=self._uproot_start, entrystop=self._uproot_stop)
            self._tagInfo = self.tagInfoMaker.convert(table)

        if self._tagInfo is not None:
            # run prediction
            entry_idx = absolute_event_idx - self._uproot_start
            # uncomment these lines for consistency check
            #p4 = self._tagInfo['_jetp4'][entry_idx][jet.idx]
            #print('pt,eta,phi', (jet.pt, jet.eta, jet.phi), (p4.pt, p4.eta, p4.phi))
            if jet:
                outputs = self.pnTaggerMD.predict_one(self._tagInfo, entry_idx, jet.idx)
                jet.pn_XbbVsQCD = convert_prob(outputs, ['Xbb'], prefix='prob')
                jet.pn_XqqVsQCD = convert_prob(outputs, ['Xqq'], prefix='prob')
                outputs = self.pnTagger.predict_one(self._tagInfo, entry_idx, jet.idx)
                jet.pn_H4qVsQCD = convert_prob(outputs, ['Hqqqq'], prefix='prob')
                outputs = self.pnTaggerHWW4q.predict_one(self._tagInfo, entry_idx, jet.idx)
                jet.pnv0_H4qVsQCD = convert_prob(outputs, ['label_H_WW_qqqq'], bkgs = ['fj_isQCD'])
                outputs = self.pnTaggerHWWlnuqq.predict_one(self._tagInfo, entry_idx, jet.idx)
                #print(outputs)
                jet.pnv0_HlnuqqVsQCD = convert_prob(outputs, ['label_H_WW_lnuqq'], bkgs = ['fj_isQCD'])
                #print(jet.pnv0_HlnuqqVsQCD)
            else:
                outputs = self.pnTaggerMD.predict(self._tagInfo)
                pn_XbbVsQCD = convert_prob(outputs, ['Xbb'], prefix='prob')
                pn_XqqVsQCD = convert_prob(outputs, ['Xqq'], prefix='prob')
                outputs = self.pnTagger.predict(self._tagInfo)
                pn_H4qVsQCD = convert_prob(outputs, ['Hqqqq'], prefix='prob')
                return pn_XbbVsQCD[0],pn_XqqVsQCD[0],pn_H4qVsQCD[0]
        else:
            jet.pn_XbbVsQCD = -1000
            jet.pn_XqqVsQCD = -1000
            jet.pn_H4qVsQCD = -1000
            jet.pnv0_H4qVsQCD = -1000
            jet.pnv0_HlnuqqVsQCD = -1000

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        event._allFatJets = Collection(event, "FatJet")
        pn_XbbVsQCD = []
        pn_XqqVsQCD  = []
        pn_H4qVsQCD = []
        pnv0_H4qVsQCD = []
        pnv0_HlnuqqVsQCD = []
        '''
        fj_XbbVsQCD,fj_XqqVsQCD,fj_H4qVsQCD = self._runParticleNet(event)

        def toArray(arr, pred):
            for i in pred: arr.append(float(i))

        toArray(pn_XbbVsQCD,fj_XbbVsQCD)
        toArray(pn_XqqVsQCD,fj_XqqVsQCD)
        toArray(pn_H4qVsQCD,fj_H4qVsQCD)
        '''
        # to predict jet by jet - it seems to take same time

        for idx, fj in enumerate(event._allFatJets):
            fj.idx = idx
            fj = event._allFatJets[idx]
            self._runParticleNet(event, fj)

            pn_XbbVsQCD.append(float(fj.pn_XbbVsQCD))
            pn_XqqVsQCD.append(float(fj.pn_XqqVsQCD))
            pn_H4qVsQCD.append(float(fj.pn_H4qVsQCD))
            pnv0_H4qVsQCD.append(float(fj.pnv0_H4qVsQCD))
            pnv0_HlnuqqVsQCD.append(float(fj.pnv0_HlnuqqVsQCD))

        self.out.fillBranch("FatJet_pnXbbVsQCD", pn_XbbVsQCD)
        self.out.fillBranch("FatJet_pnXqqVsQCD", pn_XqqVsQCD)
        self.out.fillBranch("FatJet_pnH4qVsQCD", pn_H4qVsQCD)
        self.out.fillBranch("FatJet_pnv0H4qVsQCD", pnv0_H4qVsQCD)
        self.out.fillBranch("FatJet_pnv0HlnuqqVsQCD", pnv0_HlnuqqVsQCD)
        return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
treeProducer = lambda : TreeProducer()
