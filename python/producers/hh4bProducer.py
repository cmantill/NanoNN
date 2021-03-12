import os
import itertools
import ROOT
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

class METObject(Object):
    def p4(self):
        return polarP4(self, eta=None, mass=None)
        
class hh4bProducer(Module):
    
    def __init__(self, year):
        # need to implement args
        self.year = year
        self.jetType = 'ak8'
        self._jetConeSize = 0.8
        self._fj_name = 'FatJet'
        self._sj_name = 'SubJet'
        self._fj_gen_name = 'GenJetAK8'
        self._sj_gen_name = 'SubGenJetAK8'
        self._opts = {'run_mass_regression': True, 'mass_regression_versions': ['ak8V01a', 'ak8V01b', 'ak8V01c'],
                      'WRITE_CACHE_FILE': False}
        
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
        #self.jetmetCorr = JetMETCorrector(year=self.year, jetType="AK4PFchs", **self._jmeSysts)
        #self.fatjetCorr = JetMETCorrector(year=self.year, jetType="AK8PFPuppi", **self._jmeSysts)
        #self.subjetCorr = JetMETCorrector(year=self.year, jetType="AK4PFPuppi", **self._jmeSysts)
        
        #def beginJob(self):
        #if self._needsJMECorr:
        #      self.jetmetCorr.beginJob()
        #self.fatjetCorr.beginJob()
        #      self.subjetCorr.beginJob()
        
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
        
        # trigger variables
        self.out.branch("passTrigHad", "O")
        
        # weight variables
        self.out.branch("weight", "F")
        self.out.branch("genMTT", "F")
        self.out.branch("triggerEffWeight", "F")
        self.out.branch("triggerEff3DWeight", "F")
        self.out.branch("triggerEffMCWeight", "F")
        self.out.branch("triggerEffMC3DWeight", "F")
        self.out.branch("pileupWeight", "F")
        self.out.branch("pileupWeightUp", "F")
        self.out.branch("pileupWeightDown", "F")
        self.out.branch("totalWeight", "F")
        
        # event variables
        self.out.branch("met", "F")
        
        # fatjets
        for idx in ([1, 2]):
            prefix = 'fatJet%i' % idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "Mass", "F")
            self.out.branch(prefix + "MassSD", "F")
            self.out.branch(prefix + "MassRegressed", "F")
            self.out.branch(prefix + "MassSD_UnCorrected", "F")
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

            self.out.branch(prefix + "PtOverMHH", "F")
            self.out.branch(prefix + "PtOverMSD", "F")
            
        # dihiggs variables
        self.out.branch("hh_pt", "F")
        self.out.branch("hh_eta", "F")
        self.out.branch("hh_phi", "F")
        self.out.branch("hh_mass", "F")
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
            
        # matching variables
        if self.isMC:
            for idx in ([1, 2]):
                prefix = 'genHiggs%i'%idx
                self.out.branch(prefix + "Pt", "F")
                self.out.branch(prefix + "Eta", "F")
                self.out.branch(prefix + "Phi", "F")
            self.out.branch("genHH_pt", "F")
            self.out.branch("genHH_eta", "F")
            self.out.branch("genHH_phi", "F")
            self.out.branch("genHH_mass", "F")
               
    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        if self._opts['run_mass_regression'] and self._opts['WRITE_CACHE_FILE']:
            for p in self.pnMassRegressions:
                p.update_cache()
                
        # remove all h5 cache files
        if self._opts['run_mass_regression']:
            for f in os.listdir('.'):
                if f.endswith('.h5'):
                    os.remove(f)
                    
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
               
    def fillBaseEventInfo(self, event):
        self.out.fillBranch("met", event.met.pt)
        
    def _get_filler(self, obj):
        def filler(branch, value, default=0):
            self.out.fillBranch(branch, value if obj else default)
        return filler

    def selectLeptons(self, event):
        # do lepton selection
        event.looseLeptons = []  # used for lepton counting
        event.cleaningElectrons = []
        event.cleaningMuons = []
        
        electrons = Collection(event, "Electron")
        for el in electrons:
            if el.pt > 35 and abs(el.eta) <= 2.5 and el.miniPFRelIso_all <= 0.2 and el.cutBased:
                # and abs(el.dxy) < 0.05 and abs(el.dz) < 0.2 
                el.Id = el.charge * (11)
                event.looseLeptons.append(el)
            if el.pt > 30 and el.mvaFall17V2noIso_WP90:
                event.cleaningElectrons.append(el)
                
        muons = Collection(event, "Muon")
        for mu in muons:
            if mu.pt > 30 and abs(mu.eta) <= 2.4 and mu.tightId and mu.miniPFRelIso_all <= 0.2:
                #and abs(mu.dxy) < 0.05 and abs(mu.dz) < 0.2 
                mu.Id = mu.charge * (13)
                event.looseLeptons.append(mu)
            if mu.pt > 30 and mu.looseId:
                event.cleaningMuons.append(mu)

        event.looseLeptons.sort(key=lambda x: x.pt, reverse=True)

    def correctJetsAndMET(self, event):
        # correct Jets and MET
        event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
        event._allJets = Collection(event, "Jet")
        event.met = METObject(event, "METFixEE2017") if self.year == 2017 else METObject(event, "MET")
        event._allFatJets = Collection(event, self._fj_name)
        event.subjets = Collection(event, self._sj_name)  # do not sort subjets after updating!!
        
        # need to implement JetMET corrections
        # jet mass resolution smearing
        
        # link fatjet to subjets and recompute softdrop mass
        for idx, fj in enumerate(event._allFatJets):
            fj.idx = idx
            fj.is_qualified = True
            fj.subjets = get_subjets(fj, event.subjets, ('subJetIdx1', 'subJetIdx2'))
            fj.msoftdrop = sumP4(*fj.subjets).M()
            
        # sort fat jets
        event._vbfFatJets = sorted(event._allFatJets, key=lambda x: x.pt, reverse=True)  # sort by pt
        event._allFatJets = sorted(event._allFatJets, key=lambda x: (x.ParticleNetMD_probXbb/(1.0 - x.ParticleNetMD_probXcc - x.ParticleNetMD_probXqq)), reverse = True) # sort by PnXbb score 
        
        # select jets
        event.fatjets = [fj for fj in event._allFatJets if fj.pt > 200 and abs(fj.eta) < 2.4 and (fj.jetId & 2)]
        event.ak4jets = [j for j in event._allJets if j.pt > 25 and abs(j.eta) < 2.4 and (j.jetId & 4)]
        event.ht = sum([j.pt for j in event.ak4jets])
        event.vbffatjets = [fj for fj in event._vbfFatJets if fj.pt > 200 and abs(fj.eta) < 2.4 and (fj.jetId & 2) and closest(fj, event.looseLeptons)[1] >= 0.8]
        event.vbfak4jets = [j for j in  event._allJets if j.pt > 30 and abs(j.eta) < 2.5]

        # b-tag AK4 jet selection
        # TODO: do these jets need a kinematic selection?
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
        
    def evalMassRegression(self, event, jets):
        for j in jets:
            if self._opts['run_mass_regression']:
                outputs = [p.predict_with_cache(self.tagInfoMaker, event.idx, j.idx, j) for p in self.pnMassRegressions]
                j.regressed_mass = ensemble(outputs, np.median)['mass']
            else:
                j.regressed_mass = 0          

    def fillFatJetInfo(self, event, fatjets):
        for idx in ([1, 2]):
            prefix = 'fatJet%i' % idx
            fj = fatjets[idx - 1]
            
            self.out.fillBranch(prefix + "Pt", fj.pt)
            self.out.fillBranch(prefix + "Eta", fj.eta)
            self.out.fillBranch(prefix + "Phi", fj.phi)
            self.out.fillBranch(prefix + "Mass", fj.mass)
            self.out.fillBranch(prefix + "MassSD", fj.msoftdrop) #*jmsValues[0]
            self.out.fillBranch(prefix + "MassRegressed", fj.regressed_mass)
            self.out.fillBranch(prefix + "MassSD_UnCorrected", fj.msoftdrop)
            self.out.fillBranch(prefix + "PNetXbb", (fj.ParticleNetMD_probXbb/(1.0 - fj.ParticleNetMD_probXcc - fj.ParticleNetMD_probXqq)))
            self.out.fillBranch(prefix + "PNetQCDb", fj.ParticleNetMD_probQCDb)
            self.out.fillBranch(prefix + "PNetQCDbb", fj.ParticleNetMD_probQCDbb)
            self.out.fillBranch(prefix + "PNetQCDc", fj.ParticleNetMD_probQCDc)
            self.out.fillBranch(prefix + "PNetQCDcc", fj.ParticleNetMD_probQCDcc)
            self.out.fillBranch(prefix + "PNetQCDothers", fj.ParticleNetMD_probQCDothers)
            self.out.fillBranch(prefix + "Tau3OverTau2", fj.tau3/fj.tau2)
            
            # lepton variables
            hasMuon = True if (closest(fj, event.cleaningMuons)[1] < 1.0) else False
            hasElectron = True if (closest(fj, event.cleaningElectrons)[1] < 1.0) else False
            self.out.fillBranch(prefix + "HasMuon", hasMuon)
            self.out.fillBranch(prefix + "HasElectron", hasElectron)

            # b jets
            hasBJetCSVLoose = True if (closest(fj, event.bljets)[1] < 1.0) else False
            hasBJetCSVMedium = True if (closest(fj, event.bmjets)[1] < 1.0) else False
            hasBJetCSVTight = True if (closest(fj, event.btjets)[1] < 1.0) else False
            self.out.fillBranch(prefix + "HasBJetCSVLoose", hasBJetCSVLoose)
            self.out.fillBranch(prefix + "HasBJetCSVMedium", hasBJetCSVMedium)
            self.out.fillBranch(prefix + "HasBJetCSVTight", hasBJetCSVTight)

            nb_fj_opp_ = 0
            for j in event.bmjetsCSV:
                if abs(deltaPhi(j, fj)) > 2.5 and j.pt>25:
                    nb_fj_opp_ += 1
            hasBJetOpp = True if (nb_fj_opp_>0) else False
            self.out.fillBranch(prefix + "OppositeHemisphereHasBJet", hasBJetOpp)
            
            ptovermsd = -1 if fj.msoftdrop<=0 else fj.pt/fj.msoftdrop
            self.out.fillBranch(prefix + "PtOverMSD", ptovermsd)

            # matching variables
            if self.isMC:
                # info of the closest genH
                self.out.fillBranch(prefix + "GenMatchIndex", fj.genHidx if fj.genHidx else -1)

        # hh system
        h1Jet = polarP4(fatjets[0])
        h2Jet = polarP4(fatjets[1])
        self.out.fillBranch("hh_pt", (h1Jet+h2Jet).Pt())
        self.out.fillBranch("hh_eta", (h1Jet+h2Jet).Eta())
        self.out.fillBranch("hh_phi", (h1Jet+h2Jet).Phi())
        self.out.fillBranch("hh_mass", (h1Jet+h2Jet).M())
        for idx in ([1, 2]):
            fj = fatjets[idx - 1]
            self.out.fillBranch('fatJet%i'%idx + "PtOverMHH", fj.pt/(h1Jet+h2Jet).M())

        self.out.fillBranch("deltaEta_j1j2", abs(h1Jet.Eta() - h2Jet.Eta()))
        self.out.fillBranch("deltaPhi_j1j2", deltaPhi(fatjets[0], fatjets[1]))
        self.out.fillBranch("deltaR_j1j2", deltaR(fatjets[0], fatjets[1]))
        self.out.fillBranch("ptj2_over_ptj1", fatjets[1].pt/fatjets[0].pt)
        mj2overmj1 = -1 if fatjets[0].msoftdrop<=0 else fatjets[1].msoftdrop/fatjets[0].msoftdrop
        self.out.fillBranch("mj2_over_mj1", mj2overmj1)

    def fillJetInfo(self, event, jets):
        for idx in range(len(jets)):
            if idx>3: continue
            prefix = 'jet%i'%(idx+1)
            j = jets[idx]
            self.out.fillBranch(prefix + "Pt", j.pt)
            self.out.fillBranch(prefix + "Eta", j.eta)
            self.out.fillBranch(prefix + "Phi", j.phi)


    def fillVBFFatJetInfo(self, event, fatjets):
        for idx in ([1, 2]):
            prefix = 'vbffatJet%i' % idx
            fj = fatjets[idx - 1]
            self.out.fillBranch(prefix + "Pt", fj.pt)
            self.out.fillBranch(prefix + "Eta", fj.eta)
            self.out.fillBranch(prefix + "Phi", fj.phi)
            self.out.fillBranch(prefix + "PNetXbb", (fj.ParticleNetMD_probXbb/(1.0 - fj.ParticleNetMD_probXcc - fj.ParticleNetMD_probXqq)))

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        
        self.selectLeptons(event)
        self.correctJetsAndMET(event)          
        
        # here we make the jet selection 
        probe_jets = [fj for fj in event.fatjets if fj.pt > 200]
        if len(probe_jets) < 2:
            return False

        self.selectLeptons(event)
        self.loadGenHistory(event, probe_jets)
        self.evalMassRegression(event, probe_jets)

        # fill output branches
        self.fillBaseEventInfo(event)
        self.fillFatJetInfo(event, probe_jets)          
        self.fillJetInfo(event, event._allJets)
        self.fillVBFFatJetInfo(event, event.vbffatjets)

        return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def hh4bProducer_2016(): return hh4bProducer(year=2016)
def hh4bProducer_2017(): return hh4bProducer(year=2017)
def hh4bProducer_2018(): return hh4bProducer(year=2018)
