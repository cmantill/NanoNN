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

class InputProducer(Module):

     def __init__(self):
          self.nJets = 2
          self.tagInfoMaker = ParticleNetTagInfoMaker(fatjet_branch='FatJet', pfcand_branch='PFCands', sv_branch='SV', fatpfcand_branch='FatJetPFCands', jetR=0.8)
          self.pnTagger = ParticleNetJetTagsProducer(
               os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetHWW/input/V01/preprocess.json'),
          )
          self.n_pf = self.pnTagger.prep_params['pf_features']['var_length']
          self.pf_names = self.pnTagger.prep_params['pf_features']['var_names']
          
     def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree, entriesRange=None):
          self._uproot_tree = uproot.open(inputFile.GetName())['Events']

          self.out = wrappedOutputTree
          self.out.branch("fj_idx", "I", 1)
          self.out.branch("fj_pt", "F", 1)
          self.out.branch("fj_eta", "F", 1)
          self.out.branch("fj_phi", "F", 1)
          self.out.branch("fj_mass", "F", 1)
          '''
          self.out.branch("fj_H_WW_4q", "I", 1) 
          self.out.branch("fj_H_WW_elenuqq", "I", 1)
          self.out.branch("fj_H_WW_munuqq", "I", 1)
          self.out.branch("fj_nProngs", "I", 1)
          self.out.branch("fj_dR_W", "F", 1)
          self.out.branch("fj_dR_Wstar", "F", 1)

          for key in self.pf_names:
               self.out.branch(key, "F", self.n_pf)
          '''
     def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
          pass

     def _get_gen(self, event):
          ''' 
          Loop over gen particles.
          Find Higgs boson, the W daughters and its daughters.
          @return TLorentzVectorArray of Higgs, W's and quarks in event.
          '''
          genparts = Collection(event, "GenPart")

          # find the higgs
          gmass = 0; motheridx = None; gstatus = 0;
          for ig, gpart in enumerate(genparts):
               if (gpart.pdgId == 25 and gpart.status > gstatus):
                    gMass = gpart.mass
                    motheridx = ig
                    gstatus = gpart.status

          # find its daughters
          dauidx1 = None; dau1 = ROOT.TLorentzVector();
          dauidx2 = None; dau2 = ROOT.TLorentzVector();
          if type(motheridx) != None:
               for ig, gpart in enumerate(genparts):
                    if gpart.genPartIdxMother == motheridx and abs(gpart.pdgId)==24:
                         if dau1.Pt() == 0.: dau1.SetPtEtaPhiM(gpart.pt, gpart.eta, gpart.phi, gpart.mass); dauidx1 = ig
                         elif dau2.Pt() == 0.: dau2.SetPtEtaPhiM(gpart.pt, gpart.eta, gpart.phi, gpart.mass); dauidx2 = ig
                    if type(dauidx1) != None:
                         if gpart.genPartIdxMother==dauidx1 and gpart.pdgId==genparts[dauidx1].pdgId:
                              dau1.SetPtEtaPhiM(gpart.pt, gpart.eta, gpart.phi, gpart.mass); dauidx1 = ig
                    if type(dauidx2) != None:
                         if gpart.genPartIdxMother==dauidx2 and gpart.pdgId==genparts[dauidx2].pdgId:
                              dau2.SetPtEtaPhiM(gpart.pt, gpart.eta, gpart.phi, gpart.mass); dauidx2 = ig
                        
          # find the next daughters
          def searchForWMom(self, thispart, partlist, stopids):
               if thispart.genPartIdxMother in stopids:
                    return thispart.genPartIdxMother
               elif thispart.genPartIdxMother >= 0:
                    return self.searchForWMom(partlist[thispart.genPartIdxMother], partlist, stopids)
               else:
                    return -1
          nEle = 0; nMu=0;
          if type(dauidx1) != None and type(dauidx2) != None:
               for ig, gpart in enumerate(genparts):
                    matchidx = self.searchForWMom(gpart, genparts, [dauidx1, dauidx2])
                    if matchidx==dauidx1 and abs(gpart.pdgId) in [11,13,15]:
                         nDau1
                         if abs(gpart.pdgId) == 11:
                              nEle = nEle + 1
                         elif abs(gpart.pdgId) == 13:
                              nMu = nMu + 1
                    if matchidx==dauidx2 and abs(gpart.pdgId) in [11,13,15]:
                         if abs(gpart.pdgId) == 11:
                              nEle = nEle + 1
                         elif abs(gpart.pdgId) == 13:
                              nMu = nMu + 1
                            
     def analyze(self, event, ievent):
          absolute_event_idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
          event._allFatJets = Collection(event, "FatJet")
          if len(event._allFatJets)>0: 
               table = self._uproot_tree.arrays(['FatJet_pt','FatJet_eta', 'FatJet_phi', 'FatJet_mass', 
                                                 'FatJet_msoftdrop',
                                                 '*FatJetPFCands*', 'PFCands*', 'SV*'],
                                                namedecode='utf-8', entrystart=absolute_event_idx, entrystop=absolute_event_idx+1) 
               tagInfo = self.tagInfoMaker.convert(table)
               
               # flatten to njets and pad to nPF and nSV
               for idx, fj in enumerate(event._allFatJets):
                    fj.idx = idx
                    fj = event._allFatJets[idx]
                    outputs = self.pnTagger.pad_one(tagInfo, absolute_event_idx, idx)
                    
                    print(outputs)
                    
                    # fill branches
                    '''
                    if outputs:
                    self.out.fillbranch("fj_idx", fj.idx)
                    self.out.fillbranch("fj_pt", fj.pt)
                    self.out.fillbranch("fj_eta", fj.eta)
                    self.out.fillbranch("fj_phi", fj.phi)
                    self.out.fillbranch("fj_mass", fj.mass)
                    
                    #for key in self.pf_names:
                    #     self.out.fillbranch(key, data[key])
                    '''
                    
               return True
          else:
               return False

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
inputProduder = lambda : InputProducer()
