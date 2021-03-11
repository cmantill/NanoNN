import os
import uproot
import awkward
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import numpy as np
from collections import Counter

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from PhysicsTools.NanoNN.triggerHelper import passTrigger
from PhysicsTools.NanoNN.utils import deltaPhi, sumP4, deltaR, minValue, closest

class hh4bProducer(Module):

     def __init__(self, year, **kwargs):
          self.year = year
          self.jetType = 'ak8'
          self._opts = {'run_mass_regression': True, 'mass_regression_versions': ['V01c'],
                        'WRITE_CACHE_FILE': False}
          
          if self._opts['run_mass_regression']:
               from PhysicsTools.NanoNN.makeInputs import ParticleNetTagInfoMaker
               from PhysicsTools.NanoNN.runPrediction import ParticleNetJetTagsProducer
               self.tagInfoMaker = ParticleNetTagInfoMaker(
                    fatjet_branch='FatJet', pfcand_branch='PFCands', sv_branch='SV', jetR=0.8)
               prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data')
               self.pnMassRegressions = [ParticleNetJetTagsProducer(
                    '%s/MassRegression/%s/{version}/particle_net_regression.onnx' % (prefix, self.jetType),
                    '%s/MassRegression/%s/{version}/preprocess.json' % (prefix, self.jetType),
                    version=ver, cache_suffix='mass') for ver in self._opts['mass_regression_versions']]
                    os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetHWW/input/V01/preprocess.json'),
               )

          # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
          self.DeepCSV_WP_L = {2016: 0.2217, 2017: 0.1522, 2018: 0.1241}[self.year]
          self.DeepCSV_WP_M = {2016: 0.6321, 2017: 0.4941, 2018: 0.4184}[self.year]
          self.DeepCSV_WP_T = {2016: 0.8953, 2017: 0.8001, 2018: 0.7527}[self.year]

          self.DeepFlavB_WP_L = {2016: 0.2217, 2017: 0.1522, 2018: 0.1241}[self.year]
          self.DeepFlavB_WP_M = {2016: 0.6321, 2017: 0.4941, 2018: 0.4184}[self.year]
          self.DeepFlavB_WP_T = {2016: 0.8953, 2017: 0.8001, 2018: 0.7527}[self.year]

     def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
          self.isMC = bool(inputTree.GetBranch('genWeight'))
          self.isParticleNetV01 = bool(inputTree.GetBranch(self._fj_name + '_ParticleNetMD_probQCD'))

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
          self.out.branch("run", "I") 
          self.out.branch("lumi", "I")
          self.out.branch("event", "l")
          self.out.branch("npu", "F")
          self.out.branch("rho", "F")
          self.out.branch("njets", "I")
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
               self.out.branch(prefix + "DeepJetBTag", "F")
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
               self.out.branch("genLeptonId", "I")
               self.out.branch("genLeptonMotherId", "I")
               self.out.branch("genLeptonPt", "F")
               self.out.branch("genLeptonEta", "F")
               self.out.branch("genLeptonPhi", "F")

     def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
          if self._opts['run_mass_regression'] and self._opts['WRITE_CACHE_FILE']:
               for p in self.pnMassRegressions:
                    p.update_cache()
                    
          # remove all h5 cache files
          if self._opts['run_tagger'] or self._opts['run_mass_regression']:
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
               fj.genH, fj.dr_H = closest(fj, hadGenHs)
               fj.genZ, fj.dr_Z = closest(fj, hadGenZs)
               fj.genW, fj.dr_W = closest(fj, hadGenWs)
               fj.genT, fj.dr_T = closest(fj, hadGenTops)
               fj.genLepT, fj.dr_LepT = closest(fj, lepGenTops)
               
     def fillBaseEventInfo(self, event):
          self.out.fillBranch("met", event.met.pt)

     def _get_filler(self, obj):
          def filler(branch, value, default=0):
               self.out.fillBranch(branch, value if obj else default)
          return filler

     def correctJetsAndMET(self, event):
          # correct Jets and MET
          event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
          event._allJets = Collection(event, "Jet")
          event.met = METObject(event, "METFixEE2017") if self.year == 2017 else METObject(event, "MET")
          event._allFatJets = Collection(event, self._fj_name)
          event.subjets = Collection(event, self._sj_name)  # do not sort subjets after updating!!
          
          if self._needsJMECorr:
               rho = event.fixedGridRhoFastjetAll
               # correct AK4 jets and MET
               self.jetmetCorr.setSeed(rndSeed(event, event._allJets))
               self.jetmetCorr.correctJetAndMET(jets=event._allJets, lowPtJets=Collection(event, "CorrT1METJet"),
                                                met=event.met, rawMET=METObject(event, "RawMET"),
                                                defaultMET=METObject(event, "MET"),
                                                rho=rho, genjets=Collection(event, 'GenJet') if self.isMC else None,
                                                isMC=self.isMC, runNumber=event.run)
               event._allJets = sorted(event._allJets, key=lambda x: x.pt, reverse=True)  # sort by pt after updating
               
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

          # jet mass resolution smearing
          if self.isMC and self._jmeSysts['jmr']:
               raise NotImplementedError
               
          # link fatjet to subjets and recompute softdrop mass
          for idx, fj in enumerate(event._allFatJets):
               fj.idx = idx
               fj.is_qualified = True
               fj.subjets = get_subjets(fj, event.subjets, ('subJetIdx1', 'subJetIdx2'))
               fj.msoftdrop = sumP4(*fj.subjets).M()
          event._allFatJets = sorted(event._allFatJets, key=lambda x: x.pt, reverse=True)  # sort by pt

          # select lepton-cleaned jets
          event.fatjets = [fj for fj in event._allFatJets if fj.pt > 200 and abs(fj.eta) < 2.4 and (
               fj.jetId & 2) and closest(fj, event.looseLeptons)[1] >= self._jetConeSize]
          event.ak4jets = [j for j in event._allJets if j.pt > 25 and abs(j.eta) < 2.4 and (
               j.jetId & 4) and closest(j, event.looseLeptons)[1] >= 0.4]
          event.ht = sum([j.pt for j in event.ak4jets])

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

               # matching variables
               if self.isMC:
                    # info of the closest genH
                    
     def analyze(self, event):
          """process event, return True (go to next module) or False (fail, go to next event)"""

          # correct jets before making jet related selections
          self.correctJetsAndMET(event)          
          
          # here we make the jet selection (omitting PnXbb selection for now)
          probe_jets = [fj for idx,fj in enumerate(event.fatjets) if (fj.msoftdrop > 50 and fj.pt > 250)]
          if len(probe_jets) == 0:
               return False
          # select first two jets
          probe_jets = probe_jets[:1]

          self.loadGenHistory(event, probe_jets)
          self.evalMassRegression(event, probe_jets)

          # fill output branches
          self.fillBaseEventInfo(event)
          self.fillFatJetInfo(event, probe_jets)          

          # b-tag AK4 jet selection
          event.bljets = []
          event.bmjets = []
          event.btjets = []
          for j in event._allJets:
               if not (j.pt > 40.0 and abs(j.eta) < 2.5 and (j.jetId >= 4) and (j.puId >=2)):
                    continue
               if j.btagDeepFlavB > self.DeepFlavB_WP_L:
                    event.bljets.append(j)
               if j.btagDeepFlavB > self.DeepFlavB_WP_M:
                    event.bmjets.append(j)
               if j.btagDeepFlavB > self.DeepFlavB_WP_T:
                    event.btjets.append(j)

          
          # event level
          
          return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def hh4bProducer_2016(): return hh4bProducer(year=2016)
def hh4bProducer_2017(): return hh4bProducer(year=2017)
def hh4bProducer_2018(): return hh4bProducer(year=2018)
