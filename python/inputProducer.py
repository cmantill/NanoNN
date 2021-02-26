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

from PhysicsTools.NanoNN.makeInputs import ParticleNetTagInfoMaker
from PhysicsTools.NanoNN.runPrediction import ParticleNetJetTagsProducer

from uproot_methods import TLorentzVectorArray

class InputProducer(Module):

     def __init__(self):
          self.nJets = 2
          self.tagInfoMaker = ParticleNetTagInfoMaker(fatjet_branch='FatJet', pfcand_branch='PFCands', sv_branch='SV', fatpfcand_branch='FatJetPFCands', jetR=0.8)
          self.pnTagger = ParticleNetJetTagsProducer(
               os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetHWW/input/V01/preprocess.json'),
          )
          self.n_pf = self.pnTagger.prep_params['pf_features']['var_length']
          self.pf_names = self.pnTagger.prep_params['pf_features']['var_names']
          self.n_sv = self.pnTagger.prep_params['sv_features']['var_length']
          self.sv_names = self.pnTagger.prep_params['sv_features']['var_names']          
          self.jet_r2 = 0.8 * 0.8
          
     def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree, entriesRange=None):
          self._uproot_tree = uproot.open(inputFile.GetName())['Events']

          self.out = wrappedOutputTree
          self.out.branch("fj_idx", "I", 1)
          self.out.branch("fj_pt", "F", 1)
          self.out.branch("fj_eta", "F", 1)
          self.out.branch("fj_phi", "F", 1)
          self.out.branch("fj_mass", "F", 1)
          self.out.branch("fj_lsf3", "F", 1)
          self.out.branch("fj_deepTagMD_H4qvsQCD", "F", 1)
          self.out.branch("fj_deepTag_HvsQCD", "F", 1)

          self.out.branch("fj_isQCD", "I", 1)
          self.out.branch("fj_isTop", "I", 1)
          self.out.branch("fj_H_WW_4q", "I", 1) 
          self.out.branch("fj_H_WW_elenuqq", "I", 1)
          self.out.branch("fj_H_WW_munuqq", "I", 1)
          self.out.branch("fj_nProngs", "I", 1)
          self.out.branch("fj_dR_W", "F", 1)
          self.out.branch("fj_dR_Wstar", "F", 1)

          for key in self.pf_names:
               self.out.branch(key, "F", self.n_pf)
          for key in self.sv_names:
               self.out.branch(key, "F", self.n_sv)

     def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
          pass

     def _get_gen(self, event):
          ''' 
          Loop over gen particles.
          Find Higgs boson, the W daughters and its daughters.
          @return TLorentzVectorArray of Higgs, W's and quarks in event.
          '''
          gen = {}
          genparts = Collection(event, "GenPart")

          # find the higgs
          gpt = []; geta = []; gphi = []; gmass = []; motheridx = [];
          for ig, gpart in enumerate(genparts):
               if (gpart.pdgId == 25 and gpart.status==22):
                    motheridx.append(ig)
                    gpt.append(gpart.pt); geta.append(gpart.eta); gphi.append(gpart.phi); gmass.append(gpart.mass)

          gen['Higgs'] = TLorentzVectorArray.from_ptetaphim(gpt, geta, gphi, gmass)
          
          for im,mom in enumerate(motheridx):
               # find the Ws
               gpt = []; geta = []; gphi = []; gmass = []; dauidx = [];
               for ig, gpart in enumerate(genparts):
                    if gpart.genPartIdxMother == mom and abs(gpart.pdgId)==24:
                         dauidx.append(ig)
                         gpt.append(gpart.pt); geta.append(gpart.eta); gphi.append(gpart.phi); gmass.append(gpart.mass)
                    if len(dauidx)>0:
                         if gpart.genPartIdxMother==dauidx[0] and gpart.pdgId==genparts[dauidx[0]].pdgId:
                              gpt[0] = gpart.pt; geta[0] = gpart.eta; gphi[0] = gpart.phi; gmass[0] = gpart.mass;
                    if len(dauidx)>1:
                         if gpart.genPartIdxMother==dauidx[1] and gpart.pdgId==genparts[dauidx[1]].pdgId:
                              gpt[1] = gpart.pt; geta[1] = gpart.eta; gphi[1] = gpart.phi; gmass[1] = gpart.mass;
               gen['Ws_%i'%im] =  TLorentzVectorArray.from_ptetaphim(gpt, geta, gphi, gmass)

               # find the next daughters
               def searchForWMom(thispart, partlist, stopids):
                    if thispart.genPartIdxMother in stopids:
                         return thispart.genPartIdxMother
                    elif thispart.genPartIdxMother >= 0:
                         return searchForWMom(partlist[thispart.genPartIdxMother], partlist, stopids)
                    else:
                         return -1

               nEle = 0; nMu=0;
               if len(dauidx)>1:
                    for ig, gpart in enumerate(genparts):
                         matchidx = searchForWMom(gpart, genparts, dauidx)
                         if matchidx==dauidx[0] and abs(gpart.pdgId) in [11,13,15]:
                              if abs(gpart.pdgId) == 11:
                                   nEle = nEle + 1
                              elif abs(gpart.pdgId) == 13:
                                   nMu = nMu + 1
                         if matchidx==dauidx[1] and abs(gpart.pdgId) in [11,13,15]:
                              if abs(gpart.pdgId) == 11:
                                   nEle = nEle + 1
                              elif abs(gpart.pdgId) == 13:
                                   nMu = nMu + 1
               #gen['dauW_%i'%im] = TLorentzVectorArray.from_ptetaphim(gpt, geta, gphi, gmass)

               print(nEle,nMu)
          return gen

     def analyze(self, event, ievent):
          absolute_event_idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
          event._allFatJets = Collection(event, "FatJet")
          if len(event._allFatJets)>0: 
               table = self._uproot_tree.arrays(['FatJet_pt','FatJet_eta', 'FatJet_phi', 'FatJet_mass', 
                                                 'FatJet_msoftdrop','FatJet_deepTag_H','FatJet_deepTag_QCD','FatJet_deepTag_QCDothers',
                                                 '*FatJetPFCands*', 'PFCands*', 'SV*',
                                                 'GenPart_*'],
                                                namedecode='utf-8', entrystart=absolute_event_idx, entrystop=absolute_event_idx+1) 
               self.tagInfo = self.tagInfoMaker.convert(table,True)
               #if self.tagInfo['_isHiggs']:
               #     self.gen = self._get_gen(event)
               return True
          else:
               return False

     def fill(self, event, ievent):
          absolute_event_idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
          for idx, fj in enumerate(event._allFatJets):
               if idx>1 : continue
               fj.idx = idx
               fj = event._allFatJets[idx]
               outputs = self.pnTagger.pad_one(self.tagInfo, absolute_event_idx, idx)
               if outputs:
                    self.out.fillBranch("fj_idx", fj.idx)
                    self.out.fillBranch("fj_pt", fj.pt)
                    self.out.fillBranch("fj_eta", fj.eta)
                    self.out.fillBranch("fj_phi", fj.phi)
                    self.out.fillBranch("fj_mass", fj.mass)
                    self.out.fillBranch("fj_lsf3", fj.lsf3)
                    self.out.fillBranch("fj_deepTagMD_H4qvsQCD", fj.deepTagMD_H4qvsQCD)
                    self.out.fillBranch("fj_deepTag_HvsQCD", self.tagInfo["_jet_deepTagHvsQCD"][0][idx])
                                                                                                                                                               
                    isHiggs = self.tagInfo['_isHiggs']
                    isTop = self.tagInfo['_isTop']
                    if isHiggs==0 and isTop==0: 
                         isQCD = 1
                    else: isQCD = 0
                    self.out.fillBranch("fj_isQCD", isQCD)
                    self.out.fillBranch("fj_isTop", isTop)

                    fj_H_WW_4q = 0
                    fj_H_WW_elenuqq = 0
                    fj_H_WW_munuqq = 0
                    fj_nProngs = 0
                    fj_dR_W = 0.
                    fj_dR_Wstar = 0.
                    if isHiggs:
                         #print( self.gen['Higgs'] )
                         print( self.tagInfo['_jetp4'][idx] )
                         #jet_cross_genH = self.tagInfo['_jetp4'].cross(self.gen['Higgs'], nested=True)
                         #match = jet_cross_genH.i0.delta_r2(jet_cross_genH.i1) < (0.8*0.8)
                         #print(match)



                    self.out.fillBranch("fj_H_WW_4q", fj_H_WW_4q)
                    self.out.fillBranch("fj_H_WW_elenuqq", fj_H_WW_elenuqq)
                    self.out.fillBranch("fj_H_WW_munuqq", fj_H_WW_munuqq)
                    self.out.fillBranch("fj_nProngs", fj_nProngs)
                    self.out.fillBranch("fj_dR_W", fj_dR_W)
                    self.out.fillBranch("fj_dR_Wstar", fj_dR_Wstar)

                    for key in self.pf_names:
                         self.out.fillBranch(key, outputs['pf_features'][key])
                    for key in self.sv_names:
                         self.out.fillBranch(key, outputs['sv_features'][key])

                    self.out.fill()
          return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
inputProduder = lambda : InputProducer()
