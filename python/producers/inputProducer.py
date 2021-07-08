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
               os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetHWW/input/V03/preprocess.json'),
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
          self.out.branch("fj_PN_XbbvsQCD", "F", 1)
          self.out.branch("fj_genjetmsd", "F", 1)
          self.out.branch("fj_genjetmass", "F", 1)

          self.out.branch("fj_isQCDb", "I", 1)
          self.out.branch("fj_isQCDbb", "I", 1)
          self.out.branch("fj_isQCDc", "I", 1)
          self.out.branch("fj_isQCDcc", "I", 1)
          self.out.branch("fj_isQCDlep", "I", 1)
          self.out.branch("fj_isQCDothers", "I", 1)
          self.out.branch("fj_isTop", "I", 1)
          self.out.branch("fj_isToplep", "I", 1)
          self.out.branch("fj_isW", "I", 1)
          self.out.branch("fj_isWlep", "I", 1)
          self.out.branch("fj_H_bb", "I", 1)
          self.out.branch("fj_H_cc", "I", 1)
          self.out.branch("fj_H_qq", "I", 1)
          self.out.branch("fj_H_WW_4q", "I", 1) 
          self.out.branch("fj_H_WW_elenuqq", "I", 1)
          self.out.branch("fj_H_WW_munuqq", "I", 1)
          self.out.branch("fj_H_WW_taunuqq", "I", 1)
          self.out.branch("fj_H_tt_elenuhad", "I", 1)
          self.out.branch("fj_H_tt_munuhad", "I", 1)
          self.out.branch("fj_H_tt_hadhad", "I", 1)

          self.out.branch("fj_nProngs", "I", 1)
          self.out.branch("fj_genRes_mass", "F", 1)
          self.out.branch("fj_genRes_pt", "F", 1)
          
          self.out.branch("fj_dR_W", "F", 1)
          self.out.branch("fj_genW_pt", "F", 1)
          self.out.branch("fj_genW_eta", "F", 1)
          self.out.branch("fj_genW_phi", "F", 1)
          self.out.branch("fj_genW_mass", "F", 1)
          self.out.branch("fj_dR_Wstar", "F", 1)
          self.out.branch("fj_genWstar_pt", "F", 1)
          self.out.branch("fj_genWstar_eta", "F", 1)
          self.out.branch("fj_genWstar_phi", "F", 1)
          self.out.branch("fj_genWstar_mass", "F", 1)
          self.out.branch("fj_mindR_HWW_daus", "F", 1)
          self.out.branch("fj_maxdR_HWW_daus", "F", 1)
          self.out.branch("fj_maxdR_Hbb_daus", "F", 1)
          self.out.branch("fj_genW_decay", "F", 1)
          self.out.branch("fj_genWstar_decay", "F", 1)

          self.out.branch("fj_evt_met_covxx", "F", 1)
          self.out.branch("fj_evt_met_covxy", "F", 1)
          self.out.branch("fj_evt_met_covyy", "F", 1)
          self.out.branch("fj_evt_met_dphi", "F", 1)
          self.out.branch("fj_evt_met_pt", "F", 1)
          self.out.branch("fj_evt_met_sig", "F", 1)
          self.out.branch("fj_evt_pupmet_pt", "F", 1)
          self.out.branch("fj_evt_pupmet_dphi", "F", 1)

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

          def findLeptons(genparts):
               leptons = []
               for gp in genparts:
                    if abs(gp.pdgId)==11 or abs(gp.pdgId)==13:
                         leptons.append(gp)
               return leptons

          def getFinal(gp):
               for idx in gp.dauIdx:
                    dau = genparts[idx]
                    if dau.pdgId == gp.pdgId:
                         return getFinal(dau)
               return gp

          hadGenPartons = []
          lepGenBsPartons = []
          lepGenTops = []
          hadGenTops = []
          hadGenWs = []
          lepGenWs = []
          hadGenZs = []
          bbGenHs = []
          ccGenHs = []
          qqGenHs = []
          wwGenHs = []
          ttGenHs = []
          TTGenHs = {'munuhad': {'H': [],'tau0': [], 'tau1': []},
                     'elenuhad': {'H': [],'tau0': [], 'tau1': []},
                     'hadhad': {'H': [],'tau0': [], 'tau1': []},
                }
          WWGenHs = {'munuqq': {'H': [],'W': [], 'Wstar':[]},
                     'elenuqq': {'H': [],'W': [], 'Wstar':[]},
                     'taunuqq': {'H': [],'W': [], 'Wstar':[]},
                     'qqqq': {'H': [],'W': [], 'Wstar':[]},
                }


          tauvs = []
          for gp in genparts:
               if gp.statusFlags & (1 << 13) == 0:
                    continue

               if abs(gp.pdgId) == 21 or abs(gp.pdgId) < 6:
                    hadGenPartons.append(gp)
                    if abs(gp.pdgId) == 5:
                         for idx in gp.dauIdx:
                              dau = genparts[idx]
                              if abs(dau.pdgId) == 511 or abs(dau.pdgId) == 521 or abs(dau.pdgId)==523:
                                   genB = getFinal(dau)
                                   if isDecay(genB,11) or isDecay(genB,13):
                                        lepGenBsPartons.append(genB)
                                        
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
                    else:
                         lepGenWs.append(gp)

               elif abs(gp.pdgId) == 23:
                    if isHadronic(gp):
                         hadGenZs.append(gp)

               elif abs(gp.pdgId) == 25:
                    if isDecay(gp,5):
                         bbGenHs.append(gp)
                    if isDecay(gp,4):
                         ccGenHs.append(gp)
                    if isDecay(gp,3) or isDecay(gp,2) or isDecay(gp,1):
                         qqGenHs.append(gp)
                    elif isDecay(gp,15):
                         taus = []
                         for idx in gp.dauIdx:
                              dau = genparts[idx]
                              if abs(dau.pdgId) == 15:
                                   genTau = getFinal(dau)
                                   taus.append(genTau)
                                   if len(taus)==2: break
                         if len(taus)==2:
                              nEle=0; nMu=0;
                              for t in taus:
                                   tau = ROOT.TLorentzVector();
                                   tau.SetPtEtaPhiM(t.pt, t.eta, t.phi, t.mass)
                                   find_lep = False
                                   for idx in t.dauIdx:
                                        neutrino = ROOT.TLorentzVector()
                                        tdau = genparts[idx]
                                        if abs(tdau.pdgId) == 12:
                                             # subtract neutrino from dau
                                             neutrino.SetPtEtaPhiM(tdau.pt, tdau.eta, tdau.phi, tdau.mass)
                                             ndau = tau -neutrino
                                             nEle+=1
                                             tauvs.append(ndau)
                                             find_lep = True
                                             break
                                        if abs(tdau.pdgId) == 14:
                                             neutrino.SetPtEtaPhiM(tdau.pt, tdau.eta, tdau.phi, tdau.mass)
                                             ndau = tau -neutrino
                                             nMu+=1
                                             tauvs.append(ndau)
                                             find_lep = True
                                             break
                                   if not find_lep:
                                        tauvs.append(tau)

                              key = None
                              if nEle==1 and nMu==0: key = "elenuhad"
                              if nMu==1 and nEle==0: key = "munuhad"
                              if nEle==0 and nMu==0: key = "hadhad"

                              ttGenHs.append(gp)

                              if key:
                                   TTGenHs[key]['H'].append(gp)
                                   TTGenHs[key]['tau0'].append(tauvs[0])
                                   TTGenHs[key]['tau1'].append(tauvs[1])

                    elif isDecay(gp,24):
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
                                   wwGenHs.append(gp)
                                   WWGenHs[key]['H'].append(gp)
                                   WWGenHs[key]['W'].append(gp.genW)
                                   WWGenHs[key]['Wstar'].append(gp.genWstar)


          for parton in itertools.chain(lepGenTops, hadGenTops):
               parton.daus = (parton.genB, genparts[parton.genW.dauIdx[0]], genparts[parton.genW.dauIdx[1]])
               parton.genW.daus = parton.daus[1:]
          for parton in itertools.chain(hadGenWs, hadGenZs, bbGenHs, ccGenHs, qqGenHs):
               parton.daus = (genparts[parton.dauIdx[0]], genparts[parton.dauIdx[1]])
          for parton in itertools.chain(wwGenHs):               
               parton.daus = (genparts[parton.genW.dauIdx[0]], genparts[parton.genW.dauIdx[1]], 
                              genparts[parton.genWstar.dauIdx[0]], genparts[parton.genWstar.dauIdx[1]])


          isQCD=0
          if len(lepGenTops)==0 and len(hadGenTops)==0 and len(hadGenWs)==0 and len(hadGenZs)==0 and len(bbGenHs)==0 and len(ccGenHs)==0 and len(qqGenHs)==0 and len(wwGenHs)==0 and len(ttGenHs)==0 and len(lepGenWs):
               isQCD=1

          for fj in fatjets:
               fj.genZ, fj.dr_Z, fj.genZidx = closest(fj, hadGenZs)
               fj.genW, fj.dr_W, fj.genWidx = closest(fj, hadGenWs)
               fj.genLepW, fj.dr_LepW, fj.genLepWidx = closest(fj, lepGenWs)
               fj.genT, fj.dr_T, fj.genTidx = closest(fj, hadGenTops)
               fj.genLepT, fj.dr_LepT, fj.genLepTidx = closest(fj, lepGenTops)
               fj.genHbb, fj.dr_Hbb, fj.genHbbidx = closest(fj, bbGenHs)
               fj.genHcc, fj.dr_Hcc, fj.genHccidx = closest(fj, ccGenHs)
               fj.genHqq, fj.dr_Hqq, fj.genHqqidx = closest(fj, qqGenHs)
               fj.genHww, fj.dr_Hww, fj.genHwwidx = closest(fj, wwGenHs)
               fj.genHtt, fj.dr_Htt, fj.genHttidx = closest(fj, ttGenHs)

               fj.genHtt_munuhad, fj.dr_Htt_munuhad, tmpidx = closest(fj, TTGenHs['munuhad']['H'])
               fj.genHtt_elenuhad, fj.dr_Htt_elenuhad, tmpidx = closest(fj, TTGenHs['elenuhad']['H'])
               fj.genHtt_hadhad, fj.dr_Htt_hadhad, tmpids =  closest(fj, TTGenHs['hadhad']['H'])

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
               if fj.genHWW_W:
                    fj.genHWW_Wdecay = 1 if isHadronic(fj.genHWW_W) else 0
                    fj.genHWW_Wstardecay = 1 if isHadronic(fj.genHWW_Wstar) else 0
               else:
                    fj.genHWW_Wdecay = 0
                    fj.genHWW_Wstardecay = 0

               # count the number of prongs
               nProngs = 0
               daus = []
               if fj.genHbb and fj.dr_Hbb<self.jet_r: daus = fj.genHbb.daus
               elif fj.genHcc and fj.dr_Hcc<self.jet_r: daus = fj.genHcc.daus
               elif fj.genHqq and fj.dr_Hqq<self.jet_r: daus = fj.genHqq.daus
               elif fj.genHww and fj.dr_Hww<self.jet_r: daus = fj.genHww.daus
               elif fj.genT and fj.dr_T<self.jet_r: daus = fj.genT.daus
               elif fj.genLepT and fj.dr_LepT<self.jet_r: daus = fj.genLepT.daus
               elif fj.genW and fj.dr_W<self.jet_r: daus = fj.genW.daus
               elif fj.genLepW and fj.dr_LepW<self.jet_r: daus = fj.genLepW.daus
               elif fj.genZ and fj.dr_Z<self.jet_r: daus = fj.genW.daus
               for dau in daus: 
                    if deltaR(fj, dau)< self.jet_r: nProngs +=1
               if fj.genHtt and fj.dr_Htt<self.jet_r and len(tauvs)==2:
                    nProngs = 0
                    jetv  = ROOT.TLorentzVector()
                    jetv.SetPtEtaPhiM(fj.pt, fj.eta, fj.phi, fj.msoftdrop)
                    if tauvs[0].Pt() > 0. and tauvs[1].Pt() > 0.:
                         if jetv.DeltaR(tauvs[0]) < 0.8: nProngs += 1
                         if jetv.DeltaR(tauvs[1]) < 0.8: nProngs += 1
               fj.nProngs = nProngs

               # for QCD
               # count hadrons
               n_bHadrons = fj.nBHadrons
               n_cHadrons = fj.nCHadrons
               # match parton and leptons
               fj.genParton, fj.dr_Parton, fj.genPartonidx = closest(fj, hadGenPartons)
               fj.genLepB, fj.dr_LepB, fj.genLepBidx = closest(fj, lepGenBsPartons)
               leptons = findLeptons(genparts)
               fj.genLep, fj.dr_Lep, fj.genLepBidx = closest(fj, leptons)

               fj.isQCDbb = 0
               fj.isQCDb = 0
               fj.isQCDcc = 0
               fj.isQCDc = 0
               fj.isQCDlep = 0
               fj.isQCDothers = 0
               if isQCD and fj.dr_Parton < self.jet_r:
                    if fj.dr_LepB < self.jet_r or fj.dr_Lep < (self.jet_r)*0.5:
                         fj.isQCDlep = 1
                    else:
                         if n_bHadrons>=2: fj.isQCDbb = 1
                         elif n_bHadrons==1: fj.isQCDb = 1
                         elif n_cHadrons>=2: fj.isQCDcc = 1
                         elif n_cHadrons==1: fj.isQCDc = 1
                         else:
                              fj.isQCDothers = 1

     def analyze(self, event, ievent):
          event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
          event._allFatJets = Collection(event, 'FatJet'+self.jetTag)
          if len(event._allFatJets)>0: 
               self.tagInfo = self.tagInfoMaker.load(event.idx, is_input=True, is_pfarr=False, is_masklow=True)
               if self.tagInfo is None: 
                    return False
               else: return True
          else:
               return False

     def fill(self, event, ievent):

          met = Object(event, "MET")
          pupmet = Object(event, "PuppiMET")
          
          self.loadGenHistory(event,event._allFatJets)

          if self.jetType == "AK8":
               genjetlabel = "GenJetAK8"
               genjetsdlabel = "CustomGenJetAK8"
          else:
               genjetlabel = "GenJetAK15"
               genjetsdlabel = "CustomGenJetAK15"

          is_genjet = True
          try:
               genjets = Collection(event, genjetlabel)
               genjets_sd = Collection(event, genjetsdlabel)
          except:
               is_genjet = False

          nevt = 0
          skip = -1
          for idx, fj in enumerate(event._allFatJets):
               if (fj.pt <= 171):
                    skip = idx;
                    continue
               else:
                    if skip==-1: jidx = idx
                    else: jidx = skip
               fj.idx = jidx
               fj = event._allFatJets[idx]
               # print these lines for debug
               #print('event ',event.idx,' ievent ',ievent)
               #print('fj pt ',fj.pt, self.tagInfo['_jetp4'].pt[event.idx-self.tagInfoMaker._uproot_start])
               # here we use the event idx to create the taginfo table
               # the ievent index should correspond to the event idx in case there is no previous skimming (which we should avoid here)

               isHiggs = self.tagInfo['_isHiggs']
               # for Higgs samples, save the 2 leading jets, for others save the leading jet
               if isHiggs==1:
                    if idx>2: continue
               else:
                    if idx>1: continue

               outputs = self.pnTagger.pad_one(self.tagInfo, event.idx-self.tagInfoMaker._uproot_start, jidx)
               
               is_fillflag = False
               if fj.isQCDbb==1 or fj.isQCDb==1 or fj.isQCDcc==1 or fj.isQCDc==1 or fj.isQCDlep==1 or fj.isQCDothers ==1 or fj.nProngs>0:
                    is_fillflag = True

               if outputs and fj.pt>=200 and fj.msoftdrop>=0 and is_fillflag:
                    self.out.fillBranch("fj_idx", fj.idx)
                    self.out.fillBranch("fj_pt", fj.pt)
                    self.out.fillBranch("fj_eta", fj.eta)
                    self.out.fillBranch("fj_phi", fj.phi)
                    self.out.fillBranch("fj_mass", fj.mass)
                    self.out.fillBranch("fj_msoftdrop", fj.msoftdrop)
                    if self.jetType=="AK8":
                         self.out.fillBranch("fj_lsf3", fj.lsf3)
                         self.out.fillBranch("fj_deepTagMD_H4qvsQCD", fj.deepTagMD_H4qvsQCD)
                         self.out.fillBranch("fj_deepTag_HvsQCD", self.tagInfo["_jet_deepTagHvsQCD"][event.idx-self.tagInfoMaker._uproot_start][jidx])
                         self.out.fillBranch("fj_PN_H4qvsQCD", fj.particleNet_H4qvsQCD)
                         fj_pn_xbb = 0
                         if (1.0 - fj.particleNetMD_Xcc - fj.particleNetMD_Xqq)>0:
                              fj_pn_xbb = fj.particleNetMD_Xbb/(1.0 - fj.particleNetMD_Xcc - fj.particleNetMD_Xqq)
                         self.out.fillBranch("fj_PN_XbbvsQCD", fj_pn_xbb)
                    else:
                         self.out.fillBranch("fj_lsf3", -1000)
                         self.out.fillBranch("fj_deepTagMD_H4qvsQCD", -1000)
                         self.out.fillBranch("fj_deepTag_HvsQCD", -1000)
                         self.out.fillBranch("fj_PN_H4qvsQCD", fj.ParticleNet_probHqqqq/(fj.ParticleNet_probHqqqq+fj.ParticleNet_probQCDb+fj.ParticleNet_probQCDbb+fj.ParticleNet_probQCDc+fj.ParticleNet_probQCDcc+fj.ParticleNet_probQCDothers))
                         fj_pn_xbb = 0
                         if (1.0 - fj.ParticleNetMD_probXcc - fj.ParticleNetMD_probXqq)>0:
                              fj_pn_xbb = fj.ParticleNetMD_probXbb/(1.0 - fj.ParticleNetMD_probXcc - fj.ParticleNetMD_probXqq)
                         self.out.fillBranch("fj_PN_XbbvsQCD", fj_pn_xbb)

                    self.out.fillBranch("fj_isQCDb", fj.isQCDb)
                    self.out.fillBranch("fj_isQCDbb", fj.isQCDbb)
                    self.out.fillBranch("fj_isQCDc", fj.isQCDc)
                    self.out.fillBranch("fj_isQCDcc", fj.isQCDcc)
                    self.out.fillBranch("fj_isQCDlep", fj.isQCDlep)
                    self.out.fillBranch("fj_isQCDothers", fj.isQCDothers)

                    self.out.fillBranch("fj_isTop", 1 if fj.dr_T < self.jet_r else 0)
                    self.out.fillBranch("fj_isToplep", 1 if fj.dr_LepT < self.jet_r else 0)
                    self.out.fillBranch("fj_isW", 1 if (fj.dr_W < self.jet_r and fj.dr_T > self.jet_r and not fj.genHww) else 0)
                    self.out.fillBranch("fj_isWlep", 1 if (fj.dr_LepW < self.jet_r and fj.dr_LepT > self.jet_r and not fj.genHww) else 0)

                    self.out.fillBranch("fj_H_bb", 1 if (fj.dr_Hbb < self.jet_r and max([deltaR(fj, dau) for dau in fj.genHbb.daus])<1.0) else 0)
                    self.out.fillBranch("fj_H_cc", 1 if (fj.dr_Hcc < self.jet_r and max([deltaR(fj, dau) for dau in fj.genHcc.daus])<1.0) else 0)
                    self.out.fillBranch("fj_H_qq", 1 if (fj.dr_Hqq < self.jet_r and max([deltaR(fj, dau) for dau in fj.genHqq.daus])<1.0) else 0)
                    self.out.fillBranch("fj_maxdR_Hbb_daus", max([deltaR(fj, dau) for dau in fj.genHbb.daus]) if fj.genHbb else 99)

                    # tautau
                    self.out.fillBranch("fj_H_tt_munuhad", 1 if (fj.dr_Htt_munuhad < self.jet_r and fj.nProngs==2) else 0)
                    self.out.fillBranch("fj_H_tt_elenuhad", 1 if (fj.dr_Htt_elenuhad < self.jet_r and fj.nProngs==2) else 0)
                    self.out.fillBranch("fj_H_tt_hadhad", 1 if (fj.dr_Htt_hadhad < self.jet_r and fj.nProngs==2) else 0)

                    ## WW 
                    dr_HWW_W = fj.dr_HWW_W if fj.dr_HWW_W else 99
                    dR_HWW_Wstar = fj.dr_HWW_Wstar if fj.dr_HWW_Wstar else 99
                    self.out.fillBranch("fj_H_WW_4q", 1 if (fj.dr_HWW_qqqq < self.jet_r and dr_HWW_W < self.jet_r and dR_HWW_Wstar < self.jet_r) else 0)
                    self.out.fillBranch("fj_H_WW_elenuqq", 1 if (fj.dr_HWW_elenuqq < self.jet_r and dr_HWW_W < self.jet_r and dR_HWW_Wstar < self.jet_r) else 0)
                    self.out.fillBranch("fj_H_WW_munuqq", 1 if (fj.dr_HWW_munuqq < self.jet_r and dr_HWW_W < self.jet_r and dR_HWW_Wstar < self.jet_r) else 0)
                    self.out.fillBranch("fj_H_WW_taunuqq", 1 if (fj.dr_HWW_taunuqq < self.jet_r and dr_HWW_W < self.jet_r and dR_HWW_Wstar < self.jet_r) else 0)

                    # resonance mass
                    genRes_mass = -99
                    genRes_pt = -99
                    if fj.genHww: 
                         genRes_mass = fj.genHww.mass
                         genRes_pt = fj.genHww.pt
                    elif fj.genHbb:
                         genRes_mass = fj.genHbb.mass
                         genRes_pt = fj.genHbb.pt
                    elif fj.genHcc:
                         genRes_mass = fj.genHcc.mass
                         genRes_pt = fj.genHcc.pt
                    elif fj.genHqq:
                         genRes_mass = fj.genHqq.mass
                         genRes_pt = fj.genHqq.pt
                    elif fj.genHtt:
                         genRes_mass = fj.genHtt.mass
                         genRes_pt = fj.genHtt.pt
                    elif fj.genZ:
                         genRes_mass = fj.genZ.mass
                         genRes_pt = fj.genZ.pt
                    elif fj.genT:
                         genRes_mass = fj.genT.mass
                         genRes_pt = fj.genT.pt
                    elif fj.genLepT:
                         genRes_mass = fj.genLepT.mass
                         genRes_pt = fj.genLepT.pt
                    elif fj.genW:
                         genRes_mass = fj.genW.mass
                         genRes_pt = fj.genW.pt
                    elif fj.genLepW:
                         genRes_mass = fj.genLepW.mass
                         genRes_pt = fj.genLepW.pt
                    self.out.fillBranch("fj_genRes_mass", genRes_mass)
                    self.out.fillBranch("fj_genRes_pt", genRes_pt)

                    # dR of W, Wstar, and daus
                    self.out.fillBranch("fj_dR_W", dr_HWW_W)
                    self.out.fillBranch("fj_genW_pt", fj.genHWW_W.pt if fj.genHWW_W else -99)
                    self.out.fillBranch("fj_genW_eta", fj.genHWW_W.eta if fj.genHWW_W else -99)
                    self.out.fillBranch("fj_genW_phi", fj.genHWW_W.phi if fj.genHWW_W else -99)
                    self.out.fillBranch("fj_genW_mass", fj.genHWW_W.mass if fj.genHWW_W else -99)
                    self.out.fillBranch("fj_dR_Wstar", dR_HWW_Wstar)
                    self.out.fillBranch("fj_genWstar_pt", fj.genHWW_Wstar.pt if fj.genHWW_Wstar else -99)
                    self.out.fillBranch("fj_genWstar_eta", fj.genHWW_Wstar.eta if fj.genHWW_Wstar else -99)
                    self.out.fillBranch("fj_genWstar_phi", fj.genHWW_Wstar.phi if fj.genHWW_Wstar else -99)
                    self.out.fillBranch("fj_genWstar_mass", fj.genHWW_Wstar.mass if fj.genHWW_Wstar else -99)
                    # add distance to WW daughters
                    ptt = fj.genHWW_W.pt if fj.genHWW_W else -99
                    mm = fj.genHww.mass if fj.genHww else -99
                    self.out.fillBranch("fj_maxdR_HWW_daus", max([deltaR(fj, dau) for dau in fj.genHww.daus]) if(fj.genHww and fj.dr_Hww < self.jet_r) else 99)
                    self.out.fillBranch("fj_mindR_HWW_daus", min([deltaR(fj, dau) for dau in fj.genHww.daus]) if(fj.genHww and fj.dr_Hww < self.jet_r) else 99)

                    # add whether W or W star are hadronic
                    self.out.fillBranch("fj_genW_decay", fj.genHWW_Wdecay if fj.genHWW_W else -99)
                    self.out.fillBranch("fj_genWstar_decay", fj.genHWW_Wstardecay if fj.genHWW_Wstar else -99)

                    self.out.fillBranch("fj_nProngs", fj.nProngs)

                    for key in self.pf_names:
                         self.out.fillBranch(key, outputs['pf_features'][key])
                    for key in self.sv_names:
                         self.out.fillBranch(key, outputs['sv_features'][key])

                    # fill gen 
                    if is_genjet:
                         notfilled_genjet=False
                         if self.jetType == "AK8":
                              if fj.genJetAK8Idx>0:
                                   self.out.fillBranch("fj_genjetmass", genjets[fj.genJetAK8Idx].mass)
                              else:
                                   notfilled_genjet=True
                         if not self.jetType == "AK8" or notfilled_genjet:
                              fj.genJ, fj.dr_genJ, fj.genJidx = closest(fj, genjets_sd)
                              if fj.genJ:
                                   self.out.fillBranch("fj_genjetmass",fj.genJ.mass)
                              else:
                                   self.out.fillBranch("fj_genjetmass",-1)

                         # since there is no index - make sure is matched
                         fj.genJsd, fj.dr_genJsd, fj.genJsdidx = closest(fj, genjets_sd)
                         if fj.genJsd:
                              self.out.fillBranch("fj_genjetmsd", fj.genJsd.mass)
                         else:
                              self.out.fillBranch("fj_genjetmsd", -1)
                    else:
                         self.out.fillBranch("fj_genjetmass", 0)
                         self.out.fillBranch("fj_genjetmsd", 0)

                    # fill evt info
                    self.out.fillBranch("fj_evt_met_covxx", met.covXX)
                    self.out.fillBranch("fj_evt_met_covxy", met.covXY)
                    self.out.fillBranch("fj_evt_met_covyy", met.covYY)
                    self.out.fillBranch("fj_evt_met_dphi", signedDeltaPhi(met.phi,fj.phi))
                    self.out.fillBranch("fj_evt_met_pt", met.pt)
                    self.out.fillBranch("fj_evt_met_sig", met.significance)
                    self.out.fillBranch("fj_evt_pupmet_pt", pupmet.pt)
                    self.out.fillBranch("fj_evt_pupmet_dphi", signedDeltaPhi(pupmet.phi,fj.phi))

                    self.out.fill()
                    nevt+=1

          return True

def signedDeltaPhi(phi1, phi2):
    dPhi = phi1 - phi2
    if (dPhi < -np.pi):
        dPhi = 2 * np.pi + dPhi
    elif (dPhi > np.pi):
        dPhi = -2 * np.pi + dPhi
    return dPhi

inputProducer_AK8 = lambda : InputProducer()
inputProducer_AK15 = lambda : InputProducer("AK15")
