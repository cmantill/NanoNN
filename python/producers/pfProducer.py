import os
import uproot
import itertools
import awkward
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import numpy as np
from collections import Counter

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoNN.helpers.utils import closest, sumP4, polarP4, configLogger, get_subjets, deltaPhi, deltaR

from PhysicsTools.NanoNN.helpers.makeInputs import ParticleNetTagInfoMaker
from PhysicsTools.NanoNN.helpers.runPrediction import ParticleNetJetTagsProducer

class pfProducer(Module):

     def __init__(self):
          self.nJets = 2
          self.tagInfoMaker = ParticleNetTagInfoMaker(fatjet_branch='FatJet', pfcand_branch='PFCands', sv_branch='SV', fatpfcand_branch='FatJetPFCands', jetR=0.8)
          self.pnTagger = ParticleNetJetTagsProducer(
               os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetHWW/input/pf/preprocess.json'),
          )
          self.n_pf = 70
          self.pf_names = ["pfcand_pt_log_nopuppi",
                           "pfcand_e_log_nopuppi",
                           "pfcand_etarel",
                           "pfcand_phirel",
                           "pfcand_pt",
                           "pfcand_e",
          ]
          self.jet_r = 0.8
          self.jet_r2 = 0.8 * 0.8

     def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree, entriesRange=None):
          self.fetch_step = 1000
          self.tagInfoMaker.init_file(inputFile, fetch_step=self.fetch_step)
          self.tagInfo = None

          self.out = wrappedOutputTree
          self.out.branch("fj_isQCD", "I", 1)
          self.out.branch("fj_isTop", "I", 1)
          self.out.branch("fj_isHiggs", "I", 1)
          self.out.branch("fj_hadronFlavour", "I", 1)
          self.out.branch("fj_partonFlavour", "I", 1)
          self.out.branch("fj_H_bb", "I", 1)
          self.out.branch("fj_H_WW_4q", "I", 1) 
          self.out.branch("fj_H_WW_elenuqq", "I", 1)
          self.out.branch("fj_H_WW_munuqq", "I", 1)
          self.out.branch("fj_H_WW_taunuqq", "I", 1)

          for key in self.pf_names:
               self.out.branch(key, "F", self.n_pf)

     def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
          pass

     def loadGenHistory(self, event, fatjets):
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

          def isDecay(gp,idecay):
               if len(gp.dauIdx) == 0:
                    raise ValueError('Particle has no daughters!')
               for idx in gp.dauIdx:
                    if abs(genparts[idx].pdgId) == idecay:
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
          bbGenHs = []
          wwGenHs = []
          WWGenHs = {'munuqq': {'H': [],'W': [], 'Wstar':[]},
                     'elenuqq': {'H': [],'W': [], 'Wstar':[]},
                     'taunuqq': {'H': [],'W': [], 'Wstar':[]},
                     'qqqq': {'H': [],'W': [], 'Wstar':[]},
                }

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
                         bbGenHs.append(gp)
                    elif isDecay(gp,24):
                         wwGenHs.append(gp)
                         ws = []
                         for idx in gp.dauIdx:
                              dau = genparts[idx]
                              if abs(dau.pdgId) == 24:
                                   genW = getFinal(dau)
                                   ws.append(genW)
                                   if len(ws)==2: break
                         if len(ws)==2:
                              if(ws[0].mass < ws[1].mass):
                                   gp.genWstar = ws[0]
                                   gp.genW = ws[1]
                              else:
                                   gp.genW = ws[0]
                                   gp.genWstar = ws[1]
                              key = None
                              if isHadronic(gp.genW) and isHadronic(gp.genWstar): key = 'qqqq'
                              elif ((isHadronic(gp.genW) and isDecay(gp.genWstar,11)) or (isHadronic(gp.genWstar) and isDecay(gp.genW,11))): key = "elenuqq"
                              elif ((isHadronic(gp.genW) and isDecay(gp.genWstar,13)) or (isHadronic(gp.genWstar) and isDecay(gp.genW,13))): key = "munuqq"
                              elif ((isHadronic(gp.genW) and isDecay(gp.genWstar,15)) or (isHadronic(gp.genWstar) and isDecay(gp.genW,15))): key = "taunuqq"
                    
                              if key:
                                   WWGenHs[key]['H'].append(gp)
                                   WWGenHs[key]['W'].append(gp.genW)
                                   WWGenHs[key]['Wstar'].append(gp.genWstar)

          for parton in itertools.chain(lepGenTops, hadGenTops):
               parton.daus = (parton.genB, genparts[parton.genW.dauIdx[0]], genparts[parton.genW.dauIdx[1]])
               parton.genW.daus = parton.daus[1:]
          for parton in itertools.chain(hadGenWs, hadGenZs, bbGenHs, wwGenHs):
               parton.daus = (genparts[parton.dauIdx[0]], genparts[parton.dauIdx[1]])
      
          for fj in fatjets:
               fj.genZ, fj.dr_Z, fj.genZidx = closest(fj, hadGenZs)
               fj.genW, fj.dr_W, fj.genWidx = closest(fj, hadGenWs)
               fj.genT, fj.dr_T, fj.genTidx = closest(fj, hadGenTops)
               fj.genLepT, fj.dr_LepT, fj.genLepTidx = closest(fj, lepGenTops)
               fj.genHbb, fj.dr_Hbb, fj.genHbbidx = closest(fj, bbGenHs)
               fj.genHww, fj.dr_Hww, fj.genHwwidx = closest(fj, wwGenHs)
               
               fj.genHWW_qqqq, fj.dr_HWW_qqqq, tmpidx = closest(fj, WWGenHs['qqqq']['H'])
               fj.genHWW_munuqq, fj.dr_HWW_munuqq, tmpidx = closest(fj, WWGenHs['munuqq']['H'])
               fj.genHWW_elenuqq, fj.dr_HWW_elenuqq, tmpidx = closest(fj, WWGenHs['elenuqq']['H'])
               fj.genHWW_taunuqq, fj.dr_HWW_taunuqq, tmpidx = closest(fj, WWGenHs['taunuqq']['H'])

               mindR = 99
               key=None
               if len(WWGenHs['qqqq']['W']) > 0 and fj.dr_HWW_qqqq < mindR:
                    key='qqqq'
                    mindR = fj.dr_HWW_qqqq 
               if len(WWGenHs['munuqq']['W']) > 0 and fj.dr_HWW_munuqq < mindR: 
                    key='munuqq'
                    mindR = fj.dr_HWW_munuqq
               if len(WWGenHs['elenuqq']['W']) > 0 and fj.dr_HWW_elenuqq < mindR:
                    key='elenuqq'
                    mindR = fj.dr_HWW_elenuqq
               if len(WWGenHs['taunuqq']['W']) > 0 and fj.dr_HWW_taunuqq < mindR: 
                    key='taunuqq'
                    mindR = fj.dr_HWW_taunuqq
               
               wwgenHsW = WWGenHs[key]['W'] if key else []
               wwgenHsWstar = WWGenHs[key]['Wstar'] if key else []
               fj.genHWW_W, fj.dr_HWW_W, tmpidx = closest(fj, wwgenHsW)
               fj.genHWW_Wstar, fj.dr_HWW_Wstar, tmpidx = closest(fj, wwgenHsWstar)

               fj.genJ, fj.dr_genJ, fj.genJidx = closest(fj, event._GenFatJets)
               if fj.genJ:
                    fj.partonFlavour = fj.genJ.partonFlavour
                    fj.hadronFlavour = fj.genJ.hadronFlavour
               else:
                    fj.partonFlavour = 0
                    fj.hadronFlavour = 0

     def analyze(self, event, ievent):
          event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
          event._allFatJets = Collection(event, "FatJet")
          event._GenFatJets = Collection(event, "GenJetAK8")
          if len(event._allFatJets)>0: 
               self.tagInfo = self.tagInfoMaker.load(event.idx, is_input=True, is_pfarr=False, is_masklow=True)
               if self.tagInfo is None: 
                    return False
               else: return True
          else:
               return False

     def fill(self, event, ievent):
          self.loadGenHistory(event, event._allFatJets)
          nevt = 0
          skip = -1
          for idx, fj in enumerate(event._allFatJets):
               if idx>1 : continue
               if (fj.pt <= 171):
                    skip = idx;
                    continue
               else:
                    if skip==-1: jidx = idx
                    else: jidx = skip
               fj.idx = jidx
               fj = event._allFatJets[idx]
               outputs = self.pnTagger.pad_one(self.tagInfo, event.idx-self.tagInfoMaker._uproot_start, jidx)
               if outputs:
                    self.out.fillBranch("fj_hadronFlavour", fj.hadronFlavour)
                    self.out.fillBranch("fj_partonFlavour", fj.partonFlavour)

                    isHiggs = self.tagInfo['_isHiggs']
                    isTop = self.tagInfo['_isTop']
                    if(isHiggs==0 and isTop==0): isQCD = 1
                    else: isQCD = 0
                    self.out.fillBranch("fj_isQCD", isQCD)
                    self.out.fillBranch("fj_isTop", isTop)
                    self.out.fillBranch("fj_isHiggs", isHiggs)

                    dr_HWW_W = fj.dr_HWW_W if fj.dr_HWW_W else 99
                    dR_HWW_Wstar = fj.dr_HWW_Wstar if fj.dr_HWW_Wstar else 99
                    self.out.fillBranch("fj_H_WW_4q", 1 if (fj.dr_HWW_qqqq < self.jet_r and dr_HWW_W < self.jet_r and dR_HWW_Wstar < self.jet_r) else 0)
                    self.out.fillBranch("fj_H_WW_elenuqq", 1 if (fj.dr_HWW_elenuqq < self.jet_r and dr_HWW_W < self.jet_r and dR_HWW_Wstar < self.jet_r) else 0)
                    self.out.fillBranch("fj_H_WW_munuqq", 1 if (fj.dr_HWW_munuqq < self.jet_r and dr_HWW_W < self.jet_r and dR_HWW_Wstar < self.jet_r) else 0)
                    self.out.fillBranch("fj_H_WW_taunuqq", 1 if (fj.dr_HWW_taunuqq < self.jet_r and dr_HWW_W < self.jet_r and dR_HWW_Wstar < self.jet_r) else 0)
                    self.out.fillBranch("fj_H_bb", 1 if (fj.dr_Hbb < self.jet_r) else 0)
                    #self.out.fillBranch("fj_dR_W", dr_HWW_W)
                    #self.out.fillBranch("fj_dR_Wstar", dR_HWW_Wstar)
                    #self.out.fillBranch("fj_dR_HWW_daus", max([deltaR(fj, dau) for dau in fj.genHww.daus]) if fj.genHww else 99)
                    #self.out.fillBranch("fj_dR_Hbb_daus", max([deltaR(fj, dau) for dau in fj.genHbb.daus]) if fj.genHbb else 99)

                    for key in self.pf_names:
                         self.out.fillBranch(key, outputs['pf_features'][key])

                    self.out.fill()
                    nevt+=1

          return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
inputProduder = lambda : InputProducer()
