import os
import itertools
import ROOT
import random
ROOT.PyConfig.IgnoreCommandLineOptions = True
import numpy as np
from collections import Counter
from operator import itemgetter

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from PhysicsTools.NanoNN.helpers.jetmetCorrector import JetMETCorrector, rndSeed
from PhysicsTools.NanoNN.helpers.triggerHelper import passTrigger
from PhysicsTools.NanoNN.helpers.utils import closest, sumP4, polarP4, configLogger, get_subjets, deltaPhi, deltaR
from PhysicsTools.NanoNN.helpers.nnHelper import convert_prob, ensemble

import logging
logger = logging.getLogger('nano')
configLogger('nano', loglevel=logging.INFO)

lumi_dict = {2016: 35.92, 2017: 41.53, 2018: 59.74}

class _NullObject:
    '''An null object which does not store anything, and does not raise exception.'''
    def __bool__(self):
        return False
    def __nonzero__(self):
        return False
    def __getattr__(self, name):
        pass
    def __setattr__(self, name, value):
        pass

class METObject(Object):
    def p4(self):
        return polarP4(self, eta=None, mass=None)

class hhbbWWProducer(Module):
    
    def __init__(self, year="2017", **kwargs):
        self.year = year
        self._opts = {'run_tagger': True, 'tagger_versions': ['V01'],
                      'WRITE_CACHE_FILE': False, 'option': "1"}
        for k in kwargs:
            self._opts[k] = kwargs[k]
            
        # set up tagger
        if self._opts['run_tagger']:
            from PhysicsTools.NanoNN.helpers.makeInputs import ParticleNetTagInfoMaker
            from PhysicsTools.NanoNN.helpers.runPrediction import ParticleNetJetTagsProducer
            self.tagInfoMakers = {
                'AK8': ParticleNetTagInfoMaker(fatjet_branch='FatJet', pfcand_branch='PFCands', sv_branch='SV', fatpfcand_branch="FatJetPFCands", jetR=0.8),
                'AK15': ParticleNetTagInfoMaker(fatjet_branch='FatJetAK15', pfcand_branch='PFCands', sv_branch='SV', fatpfcand_branch="JetPFCandsAK15", jetR=1.5)
            }

            prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data')
            self.pnTaggers = {
                'AK8': [ParticleNetJetTagsProducer('%s/ParticleNetHWW/{version}/AK8/preprocess.json' % (prefix), 
                                                   '%s/ParticleNetHWW/{version}/AK8/particle-net.onnx' % (prefix),
                                                   version=ver, cache_suffix='tagger') for ver in self._opts['tagger_versions']],
                'AK15': [ParticleNetJetTagsProducer('%s/ParticleNetHWW/{version}/AK15/preprocess.json' % (prefix), 
                                                    '%s/ParticleNetHWW/{version}/AK15/particle-net.onnx'% (prefix),
                                                    version=ver, cache_suffix='tagger') for ver in self._opts['tagger_versions']]
            }
            
        # selection
        if self._opts['option']=="1": print('Select Events with FatJet1 pT > 250 GeV and msoftdrop > 20 GeV')
        else: print('No selection')

    def beginJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        
        # remove all possible h5 cache files
        for f in os.listdir('.'):
            if f.endswith('.h5'):
                os.remove(f)
                
        if self._opts['run_tagger']:
            for key,item in self.pnTaggers.items():
                for p in item:
                    p.load_cache(inputFile)
            self.tagInfoMakers['AK8'].init_file(inputFile, fetch_step=1000)
            self.tagInfoMakers['AK15'].init_file(inputFile, fetch_step=1000)

        self.out = wrappedOutputTree

        # weight variables
        self.out.branch("weight", "F")
        
        # event variables
        self.out.branch("met", "F")
        self.out.branch("ht", "F")

        # fatjets
        for jetprefix in ['fatJet','ak15fatJet']:
            for idx in ([1, 2, 3]):
                prefix = '%s%i' % (jetprefix,idx)
                self.out.branch(prefix + "Pt", "F")
                self.out.branch(prefix + "Eta", "F")
                self.out.branch(prefix + "Phi", "F")
                self.out.branch(prefix + "Mass", "F")
                self.out.branch(prefix + "MassSD", "F")
                self.out.branch(prefix + "PNetXbb", "F")
                self.out.branch(prefix + "lsf3", "F")
                self.out.branch(prefix + "deepTagMD_H4qvsQCD", "F")
                self.out.branch(prefix + "deepTag_H", "F")
                self.out.branch(prefix + "deepTag_HvsQCD", "F")
                self.out.branch(prefix + "PN_H4qvsQCD", "F")
            
                # new branches
                self.out.branch(prefix + "pnMD_H4qVsQCD", "F")
                self.out.branch(prefix + "pnMD_HelenuqqVsQCD", "F")
                self.out.branch(prefix + "pnMD_HmunuqqVsQCD", "F")
                
                # matching
                if self.isMC:
                    self.out.branch(prefix + "H_WW_4q", "I", 1)
                    self.out.branch(prefix + "H_WW_elenuqq", "I", 1)
                    self.out.branch(prefix + "H_WW_munuqq", "I", 1)
                    self.out.branch(prefix + "H_WW_taunuqq", "I", 1)
                    self.out.branch(prefix + "dR_W", "F", 1)
                    self.out.branch(prefix + "dR_Wstar", "F", 1)
                    self.out.branch(prefix + "dR_HWW_daus", "F", 1)
                    self.out.branch(prefix + "dR_Hbb_daus", "F", 1)

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        if self._opts['run_tagger'] and self._opts['WRITE_CACHE_FILE']:
            for p in self.pnMassRegressions:
                p.update_cache()
                
        # remove all h5 cache files
        if self._opts['run_tagger']:
            for f in os.listdir('.'):
                if f.endswith('.h5'):
                    os.remove(f)

        if self.isMC:
            cwd = ROOT.gDirectory
            outputFile.cd()
            cwd.cd()
                    
    def correctJetsAndMET(self, event):
        # correct Jets and MET
        event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
        event._allJets = Collection(event, "Jet")
        event.met = METObject(event, "METFixEE2017") if self.year == 2017 else METObject(event, "MET")
        event._allFatJets = Collection(event, "FatJet")
        event._ak15FatJets = Collection(event, "FatJetAK15")

        # link fatjet to subjets 
        for idx, fj in enumerate(event._allFatJets):
            fj.idx = idx
            fj.Xbb = (fj.particleNetMD_Xbb/(1. - fj.particleNetMD_Xcc - fj.particleNetMD_Xqq))
        for idx, fj in enumerate(event._ak15FatJets):
            fj.idx = idx
            fj.Xbb = (fj.ParticleNetMD_probXbb/(1. - fj.ParticleNetMD_probXcc - fj.ParticleNetMD_probXqq))

        # sort fat jets
        event._ptFatJets = sorted(event._allFatJets, key=lambda x: x.pt, reverse=True)  # sort by pt        

        # select jets
        event.fatjets = [fj for fj in event._ptFatJets if fj.pt > 200 and abs(fj.eta) < 2.4 and (fj.jetId & 2)]
        event.ak15fatjets = [fj for fj in event._ak15FatJets if fj.pt > 200 and abs(fj.eta) < 2.4 and (fj.jetId & 2)]
        event.ak4jets = [j for j in event._allJets if j.pt > 25 and abs(j.eta) < 2.4 and (j.jetId & 4)]
        event.ht = sum([j.pt for j in event.ak4jets])

    def evalTagger(self, event, jets, jetType="AK8"):
        for i, j in enumerate(jets):
            if self._opts['run_tagger'] and j.pt >= 171:
                outputs = [p.predict_with_cache(self.tagInfoMakers[jetType], event.idx, j.idx, j, False, is_masklow=True) for p in self.pnTaggers[jetType]]
                outputs = ensemble(outputs, np.mean)
                j.pn_4q = outputs['fj_H_WW_4q']
                j.pn_elenuqq = outputs['fj_H_WW_elenuqq']
                j.pn_munuqq = outputs['fj_H_WW_munuqq']
                j.pn_QCD = outputs['fj_isQCD']
            else:
                j.pn_4q = -1
                j.pn_elenuqq = -1
                j.pn_munuqq = -1
                j.pn_QCD = -1
            j.pnMD_H4qVsQCD = convert_prob(j, ['4q'], ['QCD'], prefix='pn_')
            j.pnMD_HelenuqqVsQCD = convert_prob(j, ['elenuqq'], ['QCD'], prefix='pn_')
            j.pnMD_HmunuqqVsQCD = convert_prob(j, ['munuqq'], ['QCD'], prefix='pn_')

    def fillBaseEventInfo(self, event):
        self.out.fillBranch("ht", event.ht)
        self.out.fillBranch("met", event.met.pt)
        self.out.fillBranch("weight", event.gweight)

    def _get_filler(self, obj):
        def filler(branch, value, default=0):
            self.out.fillBranch(branch, value if obj else default)
        return filler

    def fillFatJetInfo(self, event, fatjets, jetType="AK8"):
        for idx in ([1, 2, 3]):
            prefix = 'fatJet%i' % idx
            coneSize = 0.8
            if jetType=='AK15': 
                prefix = 'ak15fatJet%i'%idx
                coneSize = 1.5
            fj = fatjets[idx - 1] if len(fatjets)>idx-1 else _NullObject()
            fill_fj = self._get_filler(fj)
            fill_fj(prefix + "Pt", fj.pt)
            fill_fj(prefix + "Eta", fj.eta)
            fill_fj(prefix + "Phi", fj.phi)
            fill_fj(prefix + "Mass", fj.mass)
            fill_fj(prefix + "MassSD", fj.msoftdrop)
            fill_fj(prefix + "PNetXbb", fj.Xbb)
            if jetType=="AK8":
                fill_fj(prefix + "lsf3", fj.lsf3)
                fill_fj(prefix + "deepTagMD_H4qvsQCD", fj.deepTagMD_H4qvsQCD)
                fill_fj(prefix + "deepTag_HvsQCD", fj.deepTag_H/(fj.deepTag_H+fj.deepTag_QCD+fj.deepTag_QCDothers) if fj else -1000)
                fill_fj(prefix + "PN_H4qvsQCD", fj.particleNet_H4qvsQCD)
            else:
                fill_fj(prefix + "lsf3", -1000)
                fill_fj(prefix + "deepTagMD_H4qvsQCD", -1000)
                fill_fj(prefix + "deepTag_HvsQCD", -1000)
                fill_fj(prefix + "PN_H4qvsQCD", fj.ParticleNet_probHqqqq/(fj.ParticleNet_probHqqqq+fj.ParticleNet_probQCDb+fj.ParticleNet_probQCDbb+fj.ParticleNet_probQCDc+fj.ParticleNet_probQCDcc+fj.ParticleNet_probQCDothers) if fj else -1000)
            fill_fj(prefix + "pnMD_H4qVsQCD", fj.pnMD_H4qVsQCD)
            fill_fj(prefix + "pnMD_HelenuqqVsQCD", fj.pnMD_HelenuqqVsQCD)
            fill_fj(prefix + "pnMD_HmunuqqVsQCD", fj.pnMD_HmunuqqVsQCD)
          
            # matching variables
            if self.isMC:
                dr_HWW_W = fj.dr_HWW_W if fj.dr_HWW_W else 99
                dR_HWW_Wstar = fj.dr_HWW_Wstar if fj.dr_HWW_Wstar else 99
                if fj:
                    fill_fj(prefix + "H_WW_4q", 1 if (fj.dr_HWW_qqqq < coneSize and dr_HWW_W < coneSize and dR_HWW_Wstar < coneSize) else 0)
                    fill_fj(prefix + "H_WW_elenuqq", 1 if (fj.dr_HWW_elenuqq < coneSize and dr_HWW_W < coneSize and dR_HWW_Wstar < coneSize) else 0)
                    fill_fj(prefix + "H_WW_munuqq", 1 if (fj.dr_HWW_munuqq < coneSize and dr_HWW_W < coneSize and dR_HWW_Wstar < coneSize) else 0)
                    fill_fj(prefix + "H_WW_taunuqq", 1 if (fj.dr_HWW_taunuqq < coneSize and dr_HWW_W < coneSize and dR_HWW_Wstar < coneSize) else 0)
                else:
                    fill_fj(prefix + "H_WW_4q", 0)
                    fill_fj(prefix + "H_WW_elenuqq", 0)
                    fill_fj(prefix + "H_WW_munuqq", 0)
                    fill_fj(prefix + "H_WW_taunuqq", 0)
                fill_fj(prefix + "dR_W", dr_HWW_W)
                fill_fj(prefix + "dR_Wstar", dR_HWW_Wstar)
                fill_fj(prefix + "dR_HWW_daus", max([deltaR(fj, dau) for dau in fj.genHww.daus]) if fj.genHww else 99)
                fill_fj(prefix + "dR_Hbb_daus", max([deltaR(fj, dau) for dau in fj.genHbb.daus]) if fj.genHbb else 99)

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        
        # fill histograms
        event.gweight = 1
        if self.isMC:
            event.gweight = event.genWeight / abs(event.genWeight)

        # correct jets
        self.correctJetsAndMET(event)          
        
        # basic jet selection 
        probe_jets = [fj for fj in event.fatjets if fj.pt > 200]
        probe_jets_ak15 = [fj for fj in event.ak15fatjets if fj.pt > 200]
        if len(probe_jets) < 2 and len(probe_jets_ak15) < 2:
            return False

        # load gen history 
        from PhysicsTools.NanoNN.producers.inputProducer import InputProducer
        inputProd = InputProducer('AK8')
        inputProd.loadGenHistory(event, probe_jets)
        inputProd.loadGenHistory(event, probe_jets_ak15)

        # apply selection
        passSelAK8 = False
        passSelAK15 = False
        if self._opts['option'] == "1":
            if(len(probe_jets)>1 and (probe_jets[0].pt > 300 and probe_jets[1].pt > 300 and probe_jets[0].msoftdrop>20 and probe_jets[1].msoftdrop>20)): passSelAK8 = True
            if(len(probe_jets_ak15)>1 and (probe_jets_ak15[0].pt > 300 and probe_jets_ak15[1].pt > 300 and probe_jets_ak15[0].msoftdrop>20 and probe_jets_ak15[1].msoftdrop>20)): passSelAK15 = True
        if not passSelAK8 and not passSelAK15: return False
        
        # fill base info
        self.fillBaseEventInfo(event)

        # evaluate tagger 
        if passSelAK8: 
            self.evalTagger(event, probe_jets)
            self.fillFatJetInfo(event, probe_jets)

        if passSelAK15:
            self.evalTagger(event, probe_jets_ak15, "AK15")
            self.fillFatJetInfo(event, probe_jets_ak15, "AK15")

        return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def hhbbWWProducerFromConfig():
    import yaml
    with open('hhbbWW_cfg.json') as f:
        cfg = yaml.safe_load(f)
        year = cfg['year']
        jetType = cfg['jetType']
        return hhbbWWProducer(**cfg)
