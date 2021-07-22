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

class triggerEfficiency():
    def __init__(self, year):
        self._year = year
        trigger_files = {'data': {2016: os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_2016.root'),
                                  2017: os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_2017.root'),
                                  2018: os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_2018.root')}[self._year],
                         'mc': {2016: os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_Summer16.root'),
                                2017: os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_Fall17.root'),
                                2018: os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_Fall18.root')}[self._year]
                     }

        self.triggerHists = {}
        for key,tfile in trigger_files.items():
            triggerFile = ROOT.TFile.Open(tfile)
            self.triggerHists[key]={
                'all': triggerFile.Get("efficiency_ptmass"),
                '0.9': triggerFile.Get("efficiency_ptmass_Xbb0p0To0p9"),
                '0.95': triggerFile.Get("efficiency_ptmass_Xbb0p9To0p95"),
                '0.98': triggerFile.Get("efficiency_ptmass_Xbb0p95To0p98"),
                '1.0': triggerFile.Get("efficiency_ptmass_Xbb0p98To1p0")
            }
            for key,h in self.triggerHists[key].items():
                h.SetDirectory(0)
            triggerFile.Close()
        
    def getEfficiency(self, pt, mass, xbb=-1, mcEff=False):
        triggerHists = self.triggerHists['mc'] if mcEff else self.triggerHists['data']
        if xbb < 0.9 and xbb>=0:
            thist = triggerHists['0.9']
        elif xbb < 0.95 and xbb>=0.9:
            thist = triggerHists['0.95']
        elif xbb < 0.98 and xbb>=0.95:
            thist = triggerHists['0.98']
        elif xbb <= 1.0 and xbb>=0.98:
            thist = triggerHists['1.0']
        else:
            thist = triggerHists['all']

        # constrain to histogram bounds
        if mass > thist.GetXaxis().GetXmax() * 0.999: 
            tmass = thist.GetXaxis().GetXmax() * 0.999
        elif mass < 0: 
            tmass = 0.001
        else: 
            tmass = mass
            
        if pt > thist.GetYaxis().GetXmax() * 0.999:
            tpt = thist.GetYaxis().GetXmax() * 0.999
        elif pt < 0:
            tpt = 0.001
        else:
            tpt  = pt

        trigEff = thist.GetBinContent(thist.GetXaxis().FindFixBin(tmass), 
                                      thist.GetYaxis().FindFixBin(tpt))
        return trigEff
        
class hh4bProducer(Module):
    
    def __init__(self, year, **kwargs):
        self.year = year
        self.jetType = 'ak8'
        self._jetConeSize = 0.8
        self._fj_name = 'FatJet'
        self._sj_name = 'SubJet'
        self._fj_gen_name = 'GenJetAK8'
        self._sj_gen_name = 'SubGenJetAK8'
        self._jmeSysts = {'jec': False, 'jes': None, 'jes_source': '', 'jes_uncertainty_file_prefix': '',
                          'jer': None, 'met_unclustered': None, 'smearMET': True, 'applyHEMUnc': False}
        self._opts = {'run_mass_regression': True, 'mass_regression_versions': ['ak8V01a', 'ak8V01b', 'ak8V01c'],
                      'WRITE_CACHE_FILE': False, 'option': "5", 'allJME': False}
        for k in kwargs:
            if k in self._jmeSysts:
                self._jmeSysts[k] = kwargs[k]
            else:
                self._opts[k] = kwargs[k]
        self._needsJMECorr = any([self._jmeSysts['jec'],
                                  self._jmeSysts['jes'],
                                  self._jmeSysts['jer'],
                                  self._jmeSysts['met_unclustered'],
                                  self._jmeSysts['applyHEMUnc']])
        self._allJME = self._opts['allJME']
        if self._allJME: self._needsJMECorr = False

        logger.info('Running %s channel for %s jets with JME systematics %s, other options %s',
                    self._opts['option'], self.jetType, str(self._jmeSysts), str(self._opts))
        
        # set up mass regression
        if self._opts['run_mass_regression']:
            from PhysicsTools.NanoNN.helpers.makeInputs import ParticleNetTagInfoMaker
            from PhysicsTools.NanoNN.helpers.runPrediction import ParticleNetJetTagsProducer
            self.tagInfoMaker = ParticleNetTagInfoMaker(
                fatjet_branch='FatJet', pfcand_branch='PFCands', sv_branch='SV', jetR=self._jetConeSize)
            prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data')
            self.pnMassRegressions = [ParticleNetJetTagsProducer(
                '%s/MassRegression/%s/{version}/preprocess.json' % (prefix, self.jetType),
                '%s/MassRegression/%s/{version}/particle_net_regression.onnx' % (prefix, self.jetType),
                version=ver, cache_suffix='mass') for ver in self._opts['mass_regression_versions']]

        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
        self.DeepCSV_WP_L = {2016: 0.2217, 2017: 0.1522, 2018: 0.1241}[self.year]
        self.DeepCSV_WP_M = {2016: 0.6321, 2017: 0.4941, 2018: 0.4184}[self.year]
        self.DeepCSV_WP_T = {2016: 0.8953, 2017: 0.8001, 2018: 0.7527}[self.year]
        
        self.DeepFlavB_WP_L = {2016: 0.0521, 2017: 0.0521, 2018: 0.0494}[self.year]
        self.DeepFlavB_WP_M = {2016: 0.3033, 2017: 0.3033, 2018: 0.2770}[self.year]
        self.DeepFlavB_WP_T = {2016: 0.7489, 2017: 0.7489, 2018: 0.7264}[self.year]
        
        # jet met corrections
        # jet mass scale/resolution: https://github.com/cms-nanoAOD/nanoAOD-tools/blob/a4b3c03ca5d8f4b8fbebc145ddcd605c7553d767/python/postprocessing/modules/jme/jetmetHelperRun2.py#L45-L58
        self._jmsValues = {2016: [1.00, 0.9906, 1.0094],
                           2017: [1.0016, 0.978, 0.986], # tuned to our top control region
                           2018: [0.997, 0.993, 1.001]}[self.year]
        self._jmrValues = {2016: [1.00, 1.0, 1.09],  # tuned to our top control region
                           2017: [1.03, 1.00, 1.07],
                           2018: [1.065, 1.031, 1.099]}[self.year]

        self._jmsValuesReg = {2016: [1.00, 0.998, 1.002],
                           2017: [1.002, 0.996, 1.008],
                           2018: [0.994, 0.993, 1.001]}[self.year]
        self._jmrValuesReg = {2016: [1.028, 1.007, 1.063],
                           2017: [1.026, 1.009, 1.059],
                           2018: [1.031, 1.006, 1.075]}[self.year]

        if self._needsJMECorr:
            self.jetmetCorr = JetMETCorrector(year=self.year, jetType="AK4PFchs", **self._jmeSysts)
            self.fatjetCorr = JetMETCorrector(year=self.year, jetType="AK8PFPuppi", **self._jmeSysts)
            self.subjetCorr = JetMETCorrector(year=self.year, jetType="AK4PFPuppi", **self._jmeSysts)
            self._allJME = False

        if self._allJME:
            # self.applyHEMUnc = False
            self.applyHEMUnc = self._jmeSysts['applyHEMUnc']
            # self.jetmetCorrectors = {'nominal': JetMETCorrector(year=self.year, jetType="AK4PFchs", jer='nominal', applyHEMUnc=self.applyHEMUnc),
            #                          'JERUp': JetMETCorrector(year=self.year, jetType="AK4PFchs", jer='up'),
            #                          'JERDown': JetMETCorrector(year=self.year, jetType="AK4PFchs", jer='down'),
            #                          'JESUp': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes='up'),
            #                          #'JESDown': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes='down'),
            #                          #'METUnUp': JetMETCorrector(year=self.year, jetType="AK4PFchs", met_unclustered='up'),
            #                          #'METUnDown': JetMETCorrector(year=self.year, jetType="AK4PFchs", met_unclustered='down'),
            # }
            self.fatjetCorrectors = {'nominal': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jer='nominal', applyHEMUnc=self.applyHEMUnc),
                                     'JERUp': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jer='up'),
                                     'JERDown': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jer='down'),
                                     'JESUp': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes='up'),
                                     'JESDown': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes='down')
            }
            self._jmeLabels = self.fatjetCorrectors.keys()
        else:
            self._jmeLabels = []

        # for bdt
        #self._sfbdt_files = [
        #    os.path.expandvars(
        #        '$CMSSW_BASE/src/PhysicsTools/NanoHRTTools/data/sfBDT/ak15/xgb_train_qcd.model.%d' % idx)
        #    for idx in range(10)]
        #self._sfbdt_vars = ['fj_2_tau21', 'fj_2_sj1_rawmass', 'fj_2_sj2_rawmass',
        #                    'fj_2_ntracks_sv12', 'fj_2_sj1_sv1_pt', 'fj_2_sj2_sv1_pt']

        # selection
        if self._opts['option']=="5": print('Select Events with FatJet1 pT > 200 GeV and PNetXbb > 0.8 only')
        elif self._opts['option']=="10": print('Select FatJets with pT > 200 GeV and tau3/tau2 < 0.54 only')
        else: print('No selection')

        # trigger Efficiency
        self._teff = triggerEfficiency(self.year)

    def beginJob(self):
        if self._needsJMECorr:
            self.jetmetCorr.beginJob()
            self.fatjetCorr.beginJob()
            self.subjetCorr.beginJob()
            
        if self._allJME:
            for key,corr in self.fatjetCorrectors.items():
                #self.jetmetCorrectors[key].beginJob()
                self.fatjetCorrectors[key].beginJob()

        # for bdt
        # self.xgb = XGBEnsemble(self._sfbdt_files, self._sfbdt_vars)

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        
        # remove all possible h5 cache files
        for f in os.listdir('.'):
            if f.endswith('.h5'):
                os.remove(f)
                
        if self._opts['run_mass_regression']:
            for p in self.pnMassRegressions:
                p.load_cache(inputFile)
            self.tagInfoMaker.init_file(inputFile, fetch_step=1000)
                
        self.out = wrappedOutputTree
        
        # weight variables
        self.out.branch("weight", "F")
        #self.out.branch("weightLHEScaleUp", "F")
        #self.out.branch("weightLHEScaleDown", "F")  

        # event variables
        self.out.branch("met", "F")
        self.out.branch("metphi", "F")
        #self.out.branch("npvs", "F")
        self.out.branch("ht", "F")
        self.out.branch("passmetfilters", "O")
        self.out.branch("l1PreFiringWeight", "F")
        self.out.branch("l1PreFiringWeightUp", "F")
        self.out.branch("l1PreFiringWeightDown", "F")
        self.out.branch("triggerEffWeight", "F")
        self.out.branch("triggerEff3DWeight", "F")
        self.out.branch("triggerEffMCWeight", "F")
        self.out.branch("triggerEffMC3DWeight", "F")

        # fatjets
        for idx in ([1, 2]):
            prefix = 'fatJet%i' % idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "Mass", "F")
            self.out.branch(prefix + "MassSD", "F")
            self.out.branch(prefix + "MassSD_UnCorrected", "F")
            self.out.branch(prefix + "MassRegressed", "F")
            self.out.branch(prefix + "MassRegressed_UnCorrected", "F")
            self.out.branch(prefix + "PNetXbb", "F")
            self.out.branch(prefix + "PNetQCDb", "F")
            self.out.branch(prefix + "PNetQCDbb", "F")
            self.out.branch(prefix + "PNetQCDc", "F")
            self.out.branch(prefix + "PNetQCDcc", "F")
            self.out.branch(prefix + "PNetQCDothers", "F")
            self.out.branch(prefix + "Tau3OverTau2", "F")
            
            self.out.branch(prefix + "GenMatchIndex", "I")
            self.out.branch(prefix + "HasMuon", "O")
            self.out.branch(prefix + "HasElectron", "O")
            self.out.branch(prefix + "HasBJetCSVLoose", "O")
            self.out.branch(prefix + "HasBJetCSVMedium", "O")
            self.out.branch(prefix + "HasBJetCSVTight", "O")
            self.out.branch(prefix + "OppositeHemisphereHasBJet", "O")

            # here we form the MHH system w. mass regressed
            self.out.branch(prefix + "PtOverMHH", "F")
            self.out.branch(prefix + "PtOverMSD", "F")
            self.out.branch(prefix + "PtOverMRegressed", "F")

            # uncertainties
            if self.isMC:
                self.out.branch(prefix + "MassSD_JMS_Down", "F")
                self.out.branch(prefix + "MassSD_JMS_Up", "F")
                self.out.branch(prefix + "MassSD_JMR_Down", "F")
                self.out.branch(prefix + "MassSD_JMR_Up", "F")
                self.out.branch(prefix + "MassRegressed_JMS_Down", "F")
                self.out.branch(prefix + "MassRegressed_JMS_Up", "F")
                self.out.branch(prefix + "MassRegressed_JMR_Down", "F")
                self.out.branch(prefix + "MassRegressed_JMR_Up", "F")

                self.out.branch(prefix + "PtOverMHH_JMS_Down", "F")
                self.out.branch(prefix + "PtOverMHH_JMS_Up", "F")
                self.out.branch(prefix + "PtOverMHH_JMR_Down", "F")
                self.out.branch(prefix + "PtOverMHH_JMR_Up", "F")

                if self._allJME:
                    for syst in self._jmeLabels:
                        if syst == 'nominal': continue
                        self.out.branch(prefix + "Pt" + "_" + syst, "F")
                        self.out.branch(prefix + "PtOverMHH" + "_" + syst, "F")

        # dihiggs variables
        self.out.branch("hh_pt", "F")
        self.out.branch("hh_eta", "F")
        self.out.branch("hh_phi", "F")
        self.out.branch("hh_mass", "F")

        if self.isMC:
            self.out.branch("hh_pt_JMS_Down", "F")
            self.out.branch("hh_pt_JMS_Up", "F")
            self.out.branch("hh_eta_JMS_Down", "F")
            self.out.branch("hh_eta_JMS_Up", "F")
            self.out.branch("hh_mass_JMS_Down", "F")
            self.out.branch("hh_mass_JMS_Up", "F")

            self.out.branch("hh_pt_JMR_Down", "F")
            self.out.branch("hh_pt_JMR_Up", "F")
            self.out.branch("hh_eta_JMR_Down", "F")
            self.out.branch("hh_eta_JMR_Up", "F")
            self.out.branch("hh_mass_JMR_Down", "F")
            self.out.branch("hh_mass_JMR_Up", "F")

        if self.isMC and self._allJME:
            for syst in self._jmeLabels:
                if syst == 'nominal': continue
                self.out.branch("hh_pt" + "_" +syst, "F")
                self.out.branch("hh_eta" + "_" +syst, "F")
                self.out.branch("hh_mass" + "_" + syst, "F")

        self.out.branch("deltaEta_j1j2", "F")
        self.out.branch("deltaPhi_j1j2", "F")
        self.out.branch("deltaR_j1j2", "F")
        self.out.branch("ptj2_over_ptj1", "F")
        self.out.branch("mj2_over_mj1", "F")

        
        # for phase-space overlap removal with VBFHH->4b boosted analysis
        # small jets
        self.out.branch("isVBFtag", "I")
        self.out.branch("dijetmass", "F")
        for idx in ([1, 2]):
            prefix = 'vbfjet%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "Mass", "F")
            
            prefix = 'vbffatJet%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "PNetXbb", "F")
            
        # more small jets
        for idx in ([1, 2, 3, 4]):
            prefix = 'jet%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
        self.out.branch("nBTaggedJets", "I")

        # leptons
        for idx in ([1, 2]):
            prefix = 'lep%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "Id", "I")

        # gen variables
        for idx in ([1, 2]):
            prefix = 'genHiggs%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            
    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        if self._opts['run_mass_regression'] and self._opts['WRITE_CACHE_FILE']:
            for p in self.pnMassRegressions:
                p.update_cache()
                
        # remove all h5 cache files
        if self._opts['run_mass_regression']:
            for f in os.listdir('.'):
                if f.endswith('.h5'):
                    os.remove(f)

        if self.isMC:
            cwd = ROOT.gDirectory
            outputFile.cd()
            cwd.cd()
                    
    def loadGenHistory(self, event, fatjets):
        # gen matching
        if not self.isMC:
            return
            
        try:
            genparts = event.genparts
        except RuntimeError as e:
            genparts = Collection(event, "GenPart")
            for idx, gp in enumerate(genparts):
                if 'dauIdx' not in gp.__dict__:
                    gp.dauIdx = []
                    if gp.genPartIdxMother >= 0:
                        mom = genparts[gp.genPartIdxMother]
                        if 'dauIdx' not in mom.__dict__:
                            mom.dauIdx = [idx]
                        else:
                            mom.dauIdx.append(idx)
            event.genparts = genparts

        def isHadronic(gp):
            if len(gp.dauIdx) == 0:
                raise ValueError('Particle has no daughters!')
            for idx in gp.dauIdx:
                if abs(genparts[idx].pdgId) < 6:
                    return True
            return False

        def getFinal(gp):
            for idx in gp.dauIdx:
                dau = genparts[idx]
                if dau.pdgId == gp.pdgId:
                    return getFinal(dau)
            return gp
               
        lepGenTops = []
        hadGenTops = []
        hadGenWs = []
        hadGenZs = []
        hadGenHs = []
        
        for gp in genparts:
            if gp.statusFlags & (1 << 13) == 0:
                continue
            if abs(gp.pdgId) == 6:
                for idx in gp.dauIdx:
                    dau = genparts[idx]
                    if abs(dau.pdgId) == 24:
                        genW = getFinal(dau)
                        gp.genW = genW
                        if isHadronic(genW):
                            hadGenTops.append(gp)
                        else:
                            lepGenTops.append(gp)
                    elif abs(dau.pdgId) in (1, 3, 5):
                        gp.genB = dau
            elif abs(gp.pdgId) == 24:
                if isHadronic(gp):
                    hadGenWs.append(gp)
            elif abs(gp.pdgId) == 23:
                if isHadronic(gp):
                    hadGenZs.append(gp)
            elif abs(gp.pdgId) == 25:
                if isHadronic(gp):
                    hadGenHs.append(gp)
                         
        for parton in itertools.chain(lepGenTops, hadGenTops):
            parton.daus = (parton.genB, genparts[parton.genW.dauIdx[0]], genparts[parton.genW.dauIdx[1]])
            parton.genW.daus = parton.daus[1:]
        for parton in itertools.chain(hadGenWs, hadGenZs, hadGenHs):
            parton.daus = (genparts[parton.dauIdx[0]], genparts[parton.dauIdx[1]])
            
        for fj in fatjets:
            fj.genH, fj.dr_H, fj.genHidx = closest(fj, hadGenHs)
            fj.genZ, fj.dr_Z, fj.genZidx = closest(fj, hadGenZs)
            fj.genW, fj.dr_W, fj.genWidx = closest(fj, hadGenWs)
            fj.genT, fj.dr_T, fj.genTidx = closest(fj, hadGenTops)
            fj.genLepT, fj.dr_LepT, fj.genLepidx = closest(fj, lepGenTops)

        return hadGenHs
               
    def selectLeptons(self, event):
        # do lepton selection
        event.vbfLeptons = [] # usef for vbf removal
        event.looseLeptons = []  # used for lepton counting
        event.cleaningElectrons = []
        event.cleaningMuons = []
        
        electrons = Collection(event, "Electron")
        for el in electrons:
            el.Id = el.charge * (11)
            if el.pt > 7 and abs(el.eta) < 2.5 and abs(el.dxy) < 0.05 and abs(el.dz) < 0.2:
                event.vbfLeptons.append(el)
            if el.pt > 35 and abs(el.eta) <= 2.5 and el.miniPFRelIso_all <= 0.2 and el.cutBased:
                event.looseLeptons.append(el)
            if el.pt > 30 and el.mvaFall17V2noIso_WP90:
                event.cleaningElectrons.append(el)
                
        muons = Collection(event, "Muon")
        for mu in muons:
            mu.Id = mu.charge * (13)
            if mu.pt > 5 and abs(mu.eta) < 2.4 and abs(mu.dxy) < 0.05 and abs(mu.dz) < 0.2:
                event.vbfLeptons.append(mu)
            if mu.pt > 30 and abs(mu.eta) <= 2.4 and mu.tightId and mu.miniPFRelIso_all <= 0.2:
                event.looseLeptons.append(mu)
            if mu.pt > 30 and mu.looseId:
                event.cleaningMuons.append(mu)

        event.looseLeptons.sort(key=lambda x: x.pt, reverse=True)
        event.vbfLeptons.sort(key=lambda x: x.pt, reverse=True)

    def correctJetsAndMET(self, event):
        # correct Jets and MET
        event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
        event._allJets = Collection(event, "Jet")
        event.met = METObject(event, "METFixEE2017") if self.year == 2017 else METObject(event, "MET")
        event._allFatJets = Collection(event, self._fj_name)
        event.subjets = Collection(event, self._sj_name)  # do not sort subjets after updating!!
        
        # JetMET corrections
        if self._needsJMECorr:
            rho = event.fixedGridRhoFastjetAll
            self.jetmetCorr.setSeed(rndSeed(event, event._allJets))
            self.jetmetCorr.correctJetAndMET(jets=event._allJets, lowPtJets=Collection(event, "CorrT1METJet"),
                                             met=event.met, rawMET=METObject(event, "RawMET"),
                                             defaultMET=METObject(event, "MET"),
                                             rho=rho, genjets=Collection(event, 'GenJet') if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)
            event._allJets = sorted(event._allJets, key=lambda x: x.pt, reverse=True) 
        
            # correct fatjets
            self.fatjetCorr.setSeed(rndSeed(event, event._allFatJets))
            self.fatjetCorr.correctJetAndMET(jets=event._allFatJets, met=None, rho=rho,
                                             genjets=Collection(event, self._fj_gen_name) if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)

            # correct subjets
            self.subjetCorr.setSeed(rndSeed(event, event.subjets))
            self.subjetCorr.correctJetAndMET(jets=event.subjets, met=None, rho=rho,
                                             genjets=Collection(event, self._sj_gen_name) if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)

        # all JetMET corrections
        if self._allJME:
            rho = event.fixedGridRhoFastjetAll
            event._FatJets = {}
            extra=0
            for key,corr in self.fatjetCorrectors.items():          
                if key=='nominal':
                    self.fatjetCorrectors[key].setSeed(rndSeed(event, event._allFatJets))
                    self.fatjetCorrectors[key].correctJetAndMET(jets=event._allFatJets, met=None, rho=rho,
                                                                genjets=Collection(event, self._fj_gen_name) if self.isMC else None,
                                                                isMC=self.isMC, runNumber=event.run)
                else:
                    event._FatJets[key] = Collection(event, self._fj_name)
                    self.fatjetCorrectors[key].setSeed(rndSeed(event, event._FatJets[key], extra))
                    self.fatjetCorrectors[key].correctJetAndMET(jets=event._FatJets[key], met=None, rho=rho,
                                                                genjets=Collection(event, self._fj_gen_name) if self.isMC else None,
                                                                isMC=self.isMC, runNumber=event.run)
                    for idx, fj in enumerate(event._FatJets[key]):
                        fj.idx = idx
                        fj.is_qualified = True
                        fj.Xbb = (fj.ParticleNetMD_probXbb/(1. - fj.ParticleNetMD_probXcc - fj.ParticleNetMD_probXqq))
                        fj.t32 = (fj.tau3/fj.tau2) if fj.tau2 > 0 else -1
                        fj.msoftdropJMS = fj.msoftdrop*self._jmsValues[0]

        # link fatjet to subjets 
        for idx, fj in enumerate(event._allFatJets):
            fj.idx = idx
            fj.is_qualified = True
            fj.subjets = get_subjets(fj, event.subjets, ('subJetIdx1', 'subJetIdx2'))
            fj.Xbb = (fj.ParticleNetMD_probXbb/(1. - fj.ParticleNetMD_probXcc - fj.ParticleNetMD_probXqq))
            fj.t32 = (fj.tau3/fj.tau2) if fj.tau2 > 0 else -1
            if self.isMC:
                fj.msoftdropJMS = fj.msoftdrop*self._jmsValues[0]
            else:
                fj.msoftdropJMS = fj.msoftdrop
            # do we need to recompute the softdrop mass?
            # fj.msoftdrop = sumP4(*fj.subjets).M()
            
            corr_mass_JMRUp = random.gauss(0.0, self._jmrValues[2] - 1.)
            corr_mass = max(self._jmrValues[0]-1.,0.)/(self._jmrValues[2]-1.) * corr_mass_JMRUp
            corr_mass_JMRDown = max(self._jmrValues[1]-1.,0.)/(self._jmrValues[2]-1.) * corr_mass_JMRUp
            fj.msoftdrop_corr = fj.msoftdropJMS*(1.+corr_mass)
            fj.msoftdrop_JMS_Down = fj.msoftdrop_corr*(self._jmsValues[1]/self._jmsValues[0])
            fj.msoftdrop_JMS_Up = fj.msoftdrop_corr*(self._jmsValues[2]/self._jmsValues[0])
            fj.msoftdrop_JMR_Down = fj.msoftdropJMS*(1.+corr_mass_JMRDown)
            fj.msoftdrop_JMR_Up = fj.msoftdropJMS*(1.+corr_mass_JMRUp)


        # sort fat jets
        event._ptFatJets = sorted(event._allFatJets, key=lambda x: x.pt, reverse=True)  # sort by pt
        event._xbbFatJets = sorted(event._allFatJets, key=lambda x: x.Xbb, reverse = True) # sort by PnXbb score  
        
        # select jets
        event.fatjets = [fj for fj in event._xbbFatJets if fj.pt > 200 and abs(fj.eta) < 2.4 and (fj.jetId & 2)]
        event.ak4jets = [j for j in event._allJets if j.pt > 25 and abs(j.eta) < 2.4 and (j.jetId & 4)]
        event.ht = sum([j.pt for j in event.ak4jets])

        # vbf
        # for ak4: pick a pair of opposite eta jets that maximizes pT
        # pT>25 GeV, |eta|<4.7, lepton cleaning event
        event.vbffatjets = [fj for fj in event._ptFatJets if fj.pt > 200 and abs(fj.eta) < 2.4 and (fj.jetId & 2) and closest(fj, event.looseLeptons)[1] >= 0.8]
        vbfjetid = 3 if self.year == 2017 else 2
        event.vbfak4jets = [j for j in event._allJets if j.pt > 25 and abs(j.eta) < 4.7 and (j.jetId >= vbfjetid) \
                            and ( (j.pt < 50 and j.puId>=6) or (j.pt > 50) ) and closest(j, event.vbffatjets)[1] > 1.2 \
                            and closest(j, event.vbfLeptons)[1] >= 0.4]

        # b-tag AK4 jet selection - these jets don't have a kinematic selection
        event.bljets = []
        event.bmjets = []
        event.btjets = []
        event.bmjetsCSV = []
        for j in event._allJets:
            if j.btagDeepFlavB > self.DeepFlavB_WP_L:
                event.bljets.append(j)
            if j.btagDeepFlavB > self.DeepFlavB_WP_M:
                event.bmjets.append(j)
            if j.btagDeepFlavB > self.DeepFlavB_WP_T:
                event.btjets.append(j)  
            if j.btagDeepB > self.DeepCSV_WP_M:
                event.bmjetsCSV.append(j)

        jpt_thr = 40; jeta_thr = 2.5;
        if self.year == 2016:
            jpt_thr = 30; jeta_thr = 2.4;
        event.bmjets = [j for j in event.bmjets if j.pt > jpt_thr and abs(j.eta) < jeta_thr and (j.jetId >= 4) and (j.puId >=2)]
        self.nBTaggedJets = len(event.bmjets)

        # sort and select variations of jets
        if self._allJME:
            event.fatjetsJME = {}
            for syst in self._jmeLabels:
                if syst == 'nominal': continue
                xbbordered = sorted(event._FatJets[syst], key=lambda x: x.Xbb, reverse = True) 
                event.fatjetsJME[syst] = [fj for fj in xbbordered if fj.pt > 200 and abs(fj.eta) < 2.4 and (fj.jetId & 2)]
 
    def evalMassRegression(self, event, jets):
        for i,j in enumerate(jets):
            if self._opts['run_mass_regression']:
                outputs = [p.predict_with_cache(self.tagInfoMaker, event.idx, j.idx, j) for p in self.pnMassRegressions]
                j.regressed_mass = ensemble(outputs, np.median)['mass']
                if self.isMC:
                    j.regressed_massJMS = j.regressed_mass*self._jmsValuesReg[0]
                else:
                    j.regressed_massJMS = j.regressed_mass

                corr_mass_JMRUp = random.gauss(0.0, self._jmrValuesReg[2] - 1.)
                corr_mass = max(self._jmrValuesReg[0]-1.,0.)/(self._jmrValuesReg[2]-1.) * corr_mass_JMRUp
                corr_mass_JMRDown = max(self._jmrValuesReg[1]-1.,0.)/(self._jmrValuesReg[2]-1.) * corr_mass_JMRUp
                j.regressed_mass_corr = j.regressed_massJMS*(1.+corr_mass)
                j.regressed_mass_JMS_Down = j.regressed_mass_corr*(self._jmsValuesReg[1]/self._jmsValuesReg[0])
                j.regressed_mass_JMS_Up = j.regressed_mass_corr*(self._jmsValuesReg[2]/self._jmsValuesReg[0])
                j.regressed_mass_JMR_Down = j.regressed_massJMS*(1.+corr_mass_JMRDown)
                j.regressed_mass_JMR_Up = j.regressed_massJMS*(1.+corr_mass_JMRUp)

                if self._allJME:
                    for syst in self._jmeLabels:
                        if syst == 'nominal': continue
                        if len(event.fatjetsJME[syst])>i:
                            event.fatjetsJME[syst][i].regressed_mass = j.regressed_mass
                            event.fatjetsJME[syst][i].regressed_massJMS = j.regressed_massJMS
                            
            else:
                j.regressed_mass = 0          
                j.regressed_massJMS = 0

    def fillBaseEventInfo(self, event, fatjets, hadGenHs):
        self.out.fillBranch("ht", event.ht)
        self.out.fillBranch("met", event.met.pt)
        self.out.fillBranch("metphi", event.met.phi)
        self.out.fillBranch("weight", event.gweight)
        #self.out.fillBranch("npvs", event.PV.npvs)

        # qcd weights
        """
        https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#Factorization_and_renormalizatio
        ['LHE scale variation weights (w_var / w_nominal)',
        ' [0] is renscfact=0.5d0 facscfact=0.5d0 ',
        ' [1] is renscfact=0.5d0 facscfact=1d0 ',
        ' [2] is renscfact=0.5d0 facscfact=2d0 ',
        ' [3] is renscfact=1d0 facscfact=0.5d0 ',
        ' [4] is renscfact=1d0 facscfact=1d0 ',
        ' [5] is renscfact=1d0 facscfact=2d0 ',
        ' [6] is renscfact=2d0 facscfact=0.5d0 ',
        ' [7] is renscfact=2d0 facscfact=1d0 ',
        ' [8] is renscfact=2d0 facscfact=2d0 ']
        """
        # compute envelope for weights [1,2,3,4,6,8]?

        # for PDF weights
        # need to determine if there are replicas or hessian eigenvectors?
        # 
        # if len(event.LHEPdfWeight)>0:
        # (1) get average of weights
        # (2) then sum ( weight - average )**2
        # (3) then take sqrt(sum/(nweights-1))
        # weight up: 1.0+stddev, down: 1.0-stddev (max and min of 13?)

        met_filters = bool(
            event.Flag_goodVertices and
            event.Flag_globalSuperTightHalo2016Filter and
            event.Flag_HBHENoiseFilter and
            event.Flag_HBHENoiseIsoFilter and
            event.Flag_EcalDeadCellTriggerPrimitiveFilter and
            event.Flag_BadPFMuonFilter
        )
        if self.year in (2017, 2018):
            met_filters = met_filters and event.Flag_ecalBadCalibFilterV2
        if not self.isMC:
            met_filters = met_filters and event.Flag_eeBadScFilter
        self.out.fillBranch("passmetfilters", met_filters)

        # L1 prefire weights
        if self.isMC and (self.year == 2016 or self.year == 2017):
            self.out.fillBranch("l1PreFiringWeight", event.L1PreFiringWeight_Nom)
            self.out.fillBranch("l1PreFiringWeightUp", event.L1PreFiringWeight_Up)
            self.out.fillBranch("l1PreFiringWeightDown", event.L1PreFiringWeight_Dn)
        else:
            self.out.fillBranch("l1PreFiringWeight", 1.0)
            self.out.fillBranch("l1PreFiringWeightUp", 1.0)
            self.out.fillBranch("l1PreFiringWeightDown", 1.0)

        # trigger weights
        tweight = 1.0
        tweight_mc = 1.0
        tweight_3d = 1.0
        tweight_3d_mc = 1.0
        if self.isMC:
            if len(fatjets)>1:
                tweight = 1.0 - (1.0 - self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS))*(1.0 - self._teff.getEfficiency(fatjets[1].pt, fatjets[1].msoftdropJMS))
                tweight_mc = 1.0 - (1.0 - self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, -1, True))*(1.0 - self._teff.getEfficiency(fatjets[1].pt, fatjets[1].msoftdropJMS, -1, True))
                tweight_3d = 1.0 - (1.0 - self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, fatjets[0].Xbb))*(1.0 - self._teff.getEfficiency(fatjets[1].pt, fatjets[1].msoftdropJMS, fatjets[1].Xbb))
                tweight_3d_mc = 1.0 - (1.0 - self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, fatjets[0].Xbb, True))*(1.0 - self._teff.getEfficiency(fatjets[1].pt, fatjets[1].msoftdropJMS, fatjets[1].Xbb, True))
            else:
                if len(fatjets)>0:
                    tweight = self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS)
                    tweight_mc = self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, -1, True)
                    tweight_3d = self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, fatjets[0].Xbb)
                    tweight_3d_mc = self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, fatjets[0].Xbb, True)
        self.out.fillBranch("triggerEffWeight", tweight)
        self.out.fillBranch("triggerEff3DWeight", tweight_3d)
        self.out.fillBranch("triggerEffMCWeight", tweight_mc)
        self.out.fillBranch("triggerEffMC3DWeight", tweight_3d_mc)

        # fill gen higgs info
        if hadGenHs and self.isMC:
            if len(hadGenHs)>0:
                self.out.fillBranch("genHiggs1Pt", hadGenHs[0].pt)
                self.out.fillBranch("genHiggs1Eta", hadGenHs[0].eta)
                self.out.fillBranch("genHiggs1Phi", hadGenHs[0].phi)
                if len(hadGenHs)>1:
                    self.out.fillBranch("genHiggs2Pt", hadGenHs[1].pt)
                    self.out.fillBranch("genHiggs2Eta", hadGenHs[1].eta)
                    self.out.fillBranch("genHiggs2Phi", hadGenHs[1].phi)

    def _get_filler(self, obj):
        def filler(branch, value, default=0):
            self.out.fillBranch(branch, value if obj else default)
        return filler

    def fillFatJetInfo(self, event, fatjets):
        # hh system
        h1Jet = polarP4(fatjets[0],mass='regressed_massJMS')
        h2Jet = polarP4(None)
        if len(fatjets)>1:
            h2Jet = polarP4(fatjets[1],mass='regressed_massJMS')
            self.out.fillBranch("hh_pt", (h1Jet+h2Jet).Pt())
            self.out.fillBranch("hh_eta", (h1Jet+h2Jet).Eta())
            self.out.fillBranch("hh_phi", (h1Jet+h2Jet).Phi())
            self.out.fillBranch("hh_mass", (h1Jet+h2Jet).M())
            self.out.fillBranch("deltaEta_j1j2", abs(h1Jet.Eta() - h2Jet.Eta()))
            self.out.fillBranch("deltaPhi_j1j2", deltaPhi(fatjets[0], fatjets[1]))
            self.out.fillBranch("deltaR_j1j2", deltaR(fatjets[0], fatjets[1]))
            self.out.fillBranch("ptj2_over_ptj1", fatjets[1].pt/fatjets[0].pt)
            mj2overmj1 = -1 if fatjets[0].regressed_massJMS<=0 else fatjets[1].regressed_massJMS/fatjets[0].regressed_massJMS
            self.out.fillBranch("mj2_over_mj1", mj2overmj1)

            if self.isMC:
                h1Jet_JMS_Down = polarP4(fatjets[0],mass='regressed_mass_JMS_Down')
                h2Jet_JMS_Down = polarP4(fatjets[1],mass='regressed_mass_JMS_Down')
                h1Jet_JMS_Up = polarP4(fatjets[0],mass='regressed_mass_JMS_Up')
                h2Jet_JMS_Up = polarP4(fatjets[1],mass='regressed_mass_JMS_Up')

                h1Jet_JMR_Down = polarP4(fatjets[0],mass='regressed_mass_JMR_Down')
                h2Jet_JMR_Down = polarP4(fatjets[1],mass='regressed_mass_JMR_Down')
                h1Jet_JMR_Up = polarP4(fatjets[0],mass='regressed_mass_JMR_Up')
                h2Jet_JMR_Up = polarP4(fatjets[1],mass='regressed_mass_JMR_Up')
    
                self.out.fillBranch("hh_pt_JMS_Down", (h1Jet_JMS_Down+h2Jet_JMS_Down).Pt())
                self.out.fillBranch("hh_eta_JMS_Down", (h1Jet_JMS_Down+h2Jet_JMS_Down).Eta())
                self.out.fillBranch("hh_mass_JMS_Down", (h1Jet_JMS_Down+h2Jet_JMS_Down).M())
                self.out.fillBranch("hh_pt_JMS_Up", (h1Jet_JMS_Up+h2Jet_JMS_Up).Pt())
                self.out.fillBranch("hh_eta_JMS_Up", (h1Jet_JMS_Up+h2Jet_JMS_Up).Eta())
                self.out.fillBranch("hh_mass_JMS_Up", (h1Jet_JMS_Up+h2Jet_JMS_Up).M())
                
                self.out.fillBranch("hh_pt_JMR_Down", (h1Jet_JMR_Down+h2Jet_JMR_Down).Pt())
                self.out.fillBranch("hh_eta_JMR_Down", (h1Jet_JMR_Down+h2Jet_JMR_Down).Eta())
                self.out.fillBranch("hh_mass_JMR_Down", (h1Jet_JMR_Down+h2Jet_JMR_Down).M())
                self.out.fillBranch("hh_pt_JMR_Up", (h1Jet_JMR_Up+h2Jet_JMR_Up).Pt())
                self.out.fillBranch("hh_eta_JMR_Up", (h1Jet_JMR_Up+h2Jet_JMR_Up).Eta())
                self.out.fillBranch("hh_mass_JMR_Up", (h1Jet_JMR_Up+h2Jet_JMR_Up).M())

        for idx in ([1, 2]):
            prefix = 'fatJet%i' % idx
            if len(fatjets)<=idx-1: continue
            fj = fatjets[idx-1]
            fill_fj = self._get_filler(fj)
            fill_fj(prefix + "Pt", fj.pt)
            fill_fj(prefix + "Eta", fj.eta)
            fill_fj(prefix + "Phi", fj.phi)
            fill_fj(prefix + "Mass", fj.mass)
            fill_fj(prefix + "MassRegressed_UnCorrected", fj.regressed_mass)
            fill_fj(prefix + "MassSD_UnCorrected", fj.msoftdrop)
            fill_fj(prefix + "PNetXbb", fj.Xbb)
            fill_fj(prefix + "PNetQCDb", fj.ParticleNetMD_probQCDb)
            fill_fj(prefix + "PNetQCDbb", fj.ParticleNetMD_probQCDbb)
            fill_fj(prefix + "PNetQCDc", fj.ParticleNetMD_probQCDc)
            fill_fj(prefix + "PNetQCDcc", fj.ParticleNetMD_probQCDcc)
            fill_fj(prefix + "PNetQCDothers", fj.ParticleNetMD_probQCDothers)
            fill_fj(prefix + "Tau3OverTau2", fj.t32)
            
            # uncertainties
            if self.isMC:
                fill_fj(prefix + "MassSD", fj.msoftdrop_corr)
                fill_fj(prefix + "MassSD_JMS_Down", fj.msoftdrop_JMS_Down)
                fill_fj(prefix + "MassSD_JMS_Up",  fj.msoftdrop_JMS_Up)
                fill_fj(prefix + "MassSD_JMR_Down", fj.msoftdrop_JMR_Down)
                fill_fj(prefix + "MassSD_JMR_Up",  fj.msoftdrop_JMR_Up)

                fill_fj(prefix + "MassRegressed", fj.regressed_mass_corr)
                fill_fj(prefix + "MassRegressed_JMS_Down", fj.regressed_mass_JMS_Down)
                fill_fj(prefix + "MassRegressed_JMS_Up",   fj.regressed_mass_JMS_Up)
                fill_fj(prefix + "MassRegressed_JMR_Down", fj.regressed_mass_JMR_Down)
                fill_fj(prefix + "MassRegressed_JMR_Up", fj.regressed_mass_JMR_Up)
            else:
                fill_fj(prefix + "MassSD", fj.msoftdropJMS)
                fill_fj(prefix + "MassRegressed", fj.regressed_massJMS)
            
            # lepton variables
            hasMuon = True if (closest(fj, event.cleaningMuons)[1] < 1.0) else False
            hasElectron = True if (closest(fj, event.cleaningElectrons)[1] < 1.0) else False
            fill_fj(prefix + "HasMuon", hasMuon)
            fill_fj(prefix + "HasElectron", hasElectron)

            # b jets
            hasBJetCSVLoose = True if (closest(fj, event.bljets)[1] < 1.0) else False
            hasBJetCSVMedium = True if (closest(fj, event.bmjets)[1] < 1.0) else False
            hasBJetCSVTight = True if (closest(fj, event.btjets)[1] < 1.0) else False
            fill_fj(prefix + "HasBJetCSVLoose", hasBJetCSVLoose)
            fill_fj(prefix + "HasBJetCSVMedium", hasBJetCSVMedium)
            fill_fj(prefix + "HasBJetCSVTight", hasBJetCSVTight)

            nb_fj_opp_ = 0
            for j in event.bmjetsCSV:
                if abs(deltaPhi(j, fj)) > 2.5 and j.pt>25:
                    nb_fj_opp_ += 1
            hasBJetOpp = True if (nb_fj_opp_>0) else False
            fill_fj(prefix + "OppositeHemisphereHasBJet", hasBJetOpp)

            # hh variables
            ptovermsd = -1 if fj.msoftdropJMS<=0 else fj.pt/fj.msoftdropJMS
            ptovermregressed = -1 if fj.regressed_massJMS<=0 else fj.pt/fj.regressed_massJMS
            fill_fj(prefix + "PtOverMSD", ptovermsd)
            fill_fj(prefix + "PtOverMRegressed", ptovermregressed)
            fill_fj(prefix + "PtOverMHH", fj.pt/(h1Jet+h2Jet).M())

            if self.isMC:
                if len(fatjets)>1:
                    fill_fj(prefix + "PtOverMHH_JMS_Down", fj.pt/(h1Jet_JMS_Down+h2Jet_JMS_Down).M())
                    fill_fj(prefix + "PtOverMHH_JMS_Up", fj.pt/(h1Jet_JMS_Up+h2Jet_JMS_Up).M())
                    fill_fj(prefix + "PtOverMHH_JMR_Down", fj.pt/(h1Jet_JMR_Down+h2Jet_JMR_Down).M())
                    fill_fj(prefix + "PtOverMHH_JMR_Up", fj.pt/(h1Jet_JMR_Up+h2Jet_JMR_Up).M())

            # matching variables
            if self.isMC:
                # info of the closest genH
                fill_fj(prefix + "GenMatchIndex", fj.genHidx if fj.genHidx else -1)

    def fillFatJetInfoJME(self, event):
        if not self._allJME or not self.isMC: return
        for syst in self._jmeLabels:
            if syst == 'nominal': continue
            if len(event.fatjetsJME[syst]) < 2: return
            h1Jet = polarP4(event.fatjetsJME[syst][0],mass='regressed_massJMS')
            h2Jet = polarP4(event.fatjetsJME[syst][1],mass='regressed_massJMS')
            self.out.fillBranch("hh_pt" + "_" + syst, (h1Jet+h2Jet).Pt())
            self.out.fillBranch("hh_eta" + "_" + syst, (h1Jet+h2Jet).Eta())
            self.out.fillBranch("hh_mass" + "_" + syst, (h1Jet+h2Jet).M())

            for idx in ([1, 2]):
                prefix = 'fatJet%i' % idx
                fj = event.fatjetsJME[syst][idx - 1]
                fill_fj = self._get_filler(fj)
                fill_fj(prefix + "Pt" + "_" + syst, fj.pt)
                fill_fj(prefix + "PtOverMHH" + "_" + syst, fj.pt/(h1Jet+h2Jet).M())

    def fillJetInfo(self, event, jets):
        nBTaggedJets = 0
        for idx in ([1, 2, 3, 4]):
            j = jets[idx-1] if len(jets)>idx-1 else _NullObject()
            prefix = 'jet%i'%(idx)
            fillBranch = self._get_filler(j)
            fillBranch(prefix + "Pt", j.pt)
            fillBranch(prefix + "Eta", j.eta)
            fillBranch(prefix + "Phi", j.phi)
        self.out.fillBranch("nBTaggedJets", self.nBTaggedJets)

    def fillVBFFatJetInfo(self, event, fatjets):
        for idx in ([1, 2]):
            fj = fatjets[idx-1] if len(fatjets)>idx-1 else _NullObject()
            prefix = 'vbffatJet%i' % (idx)
            fillBranch = self._get_filler(fj)
            fillBranch(prefix + "Pt", fj.pt)
            fillBranch(prefix + "Eta", fj.eta)
            fillBranch(prefix + "Phi", fj.phi)
            fillBranch(prefix + "PNetXbb", fj.Xbb)
            
    def fillVBFJetInfo(self, event, jets):
        for idx in ([1, 2]):
            j = jets[idx-1]if len(jets)>idx-1 else _NullObject()
            prefix = 'vbfjet%i' % (idx)
            fillBranch = self._get_filler(j)
            fillBranch(prefix + "Pt", j.pt)
            fillBranch(prefix + "Eta", j.eta)
            fillBranch(prefix + "Phi", j.phi)
            fillBranch(prefix + "Mass", j.mass)

        isVBFtag = 0
        if len(jets)>1:
            Jet1 = polarP4(jets[0])
            Jet2 = polarP4(jets[1])
            isVBFtag = 0
            if((Jet1+Jet2).M() > 500. and abs(Jet1.Eta() - Jet2.Eta()) > 4): isVBFtag = 1
            self.out.fillBranch('dijetmass', (Jet1+Jet2).M())
        else:
            self.out.fillBranch('dijetmass', 0)
        self.out.fillBranch('isVBFtag', isVBFtag)
            
    def fillLeptonInfo(self, event, leptons):
        for idx in ([1, 2]):
            lep = leptons[idx-1]if len(leptons)>idx-1 else _NullObject()
            prefix = 'lep%i'%(idx)
            fillBranch = self._get_filler(lep)
            fillBranch(prefix + "Pt", lep.pt)
            fillBranch(prefix + "Eta", lep.eta)
            fillBranch(prefix + "Phi", lep.phi)
            fillBranch(prefix + "Id", lep.Id)
    
    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        
        # fill histograms
        event.gweight = 1
        if self.isMC:
            event.gweight = event.genWeight / abs(event.genWeight)

        # select leptons and correct jets
        self.selectLeptons(event)
        self.correctJetsAndMET(event)          
        
        # basic jet selection 
        probe_jets = [fj for fj in event.fatjets if fj.pt > 200]
        if self._opts['option'] == "10":
            probe_jets = [fj for fj in event.fatjets if (fj.pt > 200 and fj.t32<0.54)]
            if len(probe_jets) < 1:
                return False
        else:
            if len(probe_jets) < 2:
                return False

        # evaluate regression
        self.evalMassRegression(event, probe_jets)

        # apply selection
        passSel = False
        if self._opts['option'] == "5":
            if(probe_jets[0].pt > 250 and probe_jets[1].pt > 250 and ((probe_jets[0].msoftdropJMS>50 and probe_jets[1].msoftdropJMS>50) or (probe_jets[0].regressed_massJMS>50 and probe_jets[1].regressed_massJMS>50)) and probe_jets[0].Xbb>0.8): passSel = True
        elif self._opts['option'] == "10":
            if len(probe_jets) >= 2:
                if(probe_jets[0].pt > 250 and probe_jets[1].pt > 250): passSel = True
            if(probe_jets[0].pt > 250 and len(event.looseLeptons)>0): passSel = True
        if not passSel: return False

        # load gen history
        hadGenHs = self.loadGenHistory(event, probe_jets)

        # fill output branches
        self.fillBaseEventInfo(event, probe_jets, hadGenHs)
        self.fillFatJetInfo(event, probe_jets)
          
        # for ak4 jets we only fill the b-tagged medium jets
        self.fillJetInfo(event, event.bmjets)
        self.fillVBFFatJetInfo(event, event.vbffatjets)
        self.fillVBFJetInfo(event, event.vbfak4jets)
        self.fillLeptonInfo(event, event.looseLeptons)
        
        # for all jme systs
        if self._allJME and self.isMC and self._opts['option'] == "5":
            self.fillFatJetInfoJME(event)

        return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def hh4bProducerFromConfig():
    import yaml
    with open('hh4b_cfg.json') as f:
        cfg = yaml.safe_load(f)
        year = cfg['year']
        return hh4bProducer(**cfg)
