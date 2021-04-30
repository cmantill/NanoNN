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

class InputProducer(Module):

     def __init__(self,jetType="AK8"):
          self.nJets = 2
          self.jetType = jetType
          self.jet_r = 0.8 if jetType=="AK8" else 1.5
          self.jet_r2 = self.jet_r * self.jet_r
          self.jetTag = "" if jetType=="AK8" else jetType
          fatpfcand_branch = "FatJetPFCands" if jetType=="AK8" else "JetPFCandsAK15"
          self.tagInfoMaker = ParticleNetTagInfoMaker(fatjet_branch='FatJet'+self.jetTag, pfcand_branch='PFCands', sv_branch='SV', fatpfcand_branch=fatpfcand_branch, jetR=self.jet_r)
          self.pnTagger = ParticleNetJetTagsProducer(
               os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetHWW/input/V01/preprocess.json'),
          )
          self.n_pf = self.pnTagger.prep_params['pf_features']['var_length']
          self.pf_names = self.pnTagger.prep_params['pf_features']['var_names']
          self.n_sv = self.pnTagger.prep_params['sv_features']['var_length']
          self.sv_names = self.pnTagger.prep_params['sv_features']['var_names']

     def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree, entriesRange=None):
          self.tagInfoLen = 0
          self.fetch_step = 1000
          self.tagInfoMaker.init_file(inputFile, fetch_step=self.fetch_step)
          self.tagInfo = None

          self.out = wrappedOutputTree
          self.out.branch("fj_idx", "I", 1)
          self.out.branch("fj_pt", "F", 1)
          self.out.branch("fj_eta", "F", 1)
          self.out.branch("fj_phi", "F", 1)
          self.out.branch("fj_mass", "F", 1)
          self.out.branch("fj_msoftdrop", "F", 1)
          self.out.branch("fj_lsf3", "F", 1)
          self.out.branch("fj_deepTagMD_H4qvsQCD", "F", 1)
          self.out.branch("fj_deepTag_HvsQCD", "F", 1)
          self.out.branch("fj_PN_H4qvsQCD", "F", 1)

          self.out.branch("fj_isQCD", "I", 1)
          self.out.branch("fj_isTop", "I", 1)
          self.out.branch("fj_H_WW_4q", "I", 1) 
          self.out.branch("fj_H_WW_elenuqq", "I", 1)
          self.out.branch("fj_H_WW_munuqq", "I", 1)
          self.out.branch("fj_H_WW_taunuqq", "I", 1)
          self.out.branch("fj_dR_W", "F", 1)
          self.out.branch("fj_dR_Wstar", "F", 1)
          self.out.branch("fj_dR_HWW_daus", "F", 1)
          self.out.branch("fj_dR_Hbb_daus", "F", 1)

          for key in self.pf_names:
               self.out.branch(key, "F", self.n_pf)
          for key in self.sv_names:
               self.out.branch(key, "F", self.n_sv)

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

               key=None
               if len(WWGenHs['qqqq']['W']) > 0: key='qqqq'
               elif len(WWGenHs['munuqq']['W']) > 0: key='munuqq'
               elif len(WWGenHs['elenuqq']['W']) > 0: key='elenuqq'
               elif len(WWGenHs['taunuqq']['W']) > 0: key='taunuqq'
               
               wwgenHsW = WWGenHs[key]['W'] if key else []
               wwgenHsWstar = WWGenHs[key]['Wstar'] if key else []
               fj.genHWW_W, fj.dr_HWW_W, tmpidx = closest(fj, wwgenHsW)
               fj.genHWW_Wstar, fj.dr_HWW_Wstar, tmpidx = closest(fj, wwgenHsWstar)

     def analyze(self, event, ievent):
          absolute_event_idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
          event._allFatJets = Collection(event, 'FatJet'+self.jetTag)
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

          self.loadGenHistory(event,event._allFatJets)

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
                    self.out.fillBranch("fj_idx", fj.idx)
                    self.out.fillBranch("fj_pt", fj.pt)
                    self.out.fillBranch("fj_eta", fj.eta)
                    self.out.fillBranch("fj_phi", fj.phi)
                    self.out.fillBranch("fj_mass", fj.mass)
                    self.out.fillBranch("fj_msoftdrop", fj.msoftdrop)
                    if self.jetType=="AK8":
                         self.out.fillBranch("fj_lsf3", fj.lsf3)
                         self.out.fillBranch("fj_deepTagMD_H4qvsQCD", fj.deepTagMD_H4qvsQCD)
                         self.out.fillBranch("fj_deepTag_HvsQCD", self.tagInfo["_jet_deepTagHvsQCD"][ieventTag][jidx])
                         self.out.fillBranch("fj_PN_H4qvsQCD", fj.particleNet_H4qvsQCD)
                    else:
                         self.out.fillBranch("fj_lsf3", -1000)
                         self.out.fillBranch("fj_deepTagMD_H4qvsQCD", -1000)
                         self.out.fillBranch("fj_deepTag_HvsQCD", -1000)
                         self.out.fillBranch("fj_PN_H4qvsQCD", fj.ParticleNet_probHqqqq/(fj.ParticleNet_probHqqqq+fj.ParticleNet_probQCDb+fj.ParticleNet_probQCDbb+fj.ParticleNet_probQCDc+fj.ParticleNet_probQCDcc+fj.ParticleNet_probQCDothers))

                    isHiggs = self.tagInfo['_isHiggs']
                    isTop = self.tagInfo['_isTop']
                    if(isHiggs==0 and isTop==0): isQCD = 1
                    else: isQCD = 0
                    self.out.fillBranch("fj_isQCD", isQCD)
                    self.out.fillBranch("fj_isTop", isTop)

                    self.out.fillBranch("fj_H_WW_4q", 1 if fj.dr_HWW_qqqq < self.jet_r else 0)
                    self.out.fillBranch("fj_H_WW_elenuqq", 1 if fj.dr_HWW_elenuqq < self.jet_r else 0)
                    self.out.fillBranch("fj_H_WW_munuqq", 1 if fj.dr_HWW_munuqq < self.jet_r else 0)
                    self.out.fillBranch("fj_H_WW_taunuqq", 1 if fj.dr_HWW_taunuqq < self.jet_r else 0)
                    self.out.fillBranch("fj_dR_W", fj.dr_HWW_W if fj.dr_HWW_W else 99)
                    self.out.fillBranch("fj_dR_Wstar", fj.dr_HWW_Wstar if fj.dr_HWW_Wstar else 99)
                    self.out.fillBranch("fj_dR_HWW_daus", max([deltaR(fj, dau) for dau in fj.genHww.daus]) if fj.genHww else 99)
                    self.out.fillBranch("fj_dR_Hbb_daus", max([deltaR(fj, dau) for dau in fj.genHbb.daus]) if fj.genHbb else 99)
                    # add whether W or W star are hadronic

                    for key in self.pf_names:
                         self.out.fillBranch(key, outputs['pf_features'][key])
                    for key in self.sv_names:
                         self.out.fillBranch(key, outputs['sv_features'][key])

                    self.out.fill()
                    nevt+=1

          return True

inputProducer_AK8 = lambda : InputProducer()
inputProducer_AK15 = lambda : InputProducer("AK15")
