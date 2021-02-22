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

from PhysicsTools.NanoNN.runPrediction import ParticleNetJetTagsProducer
from PhysicsTools.NanoNN.nnHelper import convert_prob

class ParticleNetTagInfoMaker(object):
     def __init__(self, fatjet_branch, pfcand_branch, sv_branch, fatpfcand_branch, jetR=0.8):
          self.fatjet_branch = fatjet_branch
          self.pfcand_branch = pfcand_branch
          self.fatjetpf_branch = fatpfcand_branch
          self.sv_branch = sv_branch
          self.jet_r2 = jetR * jetR

     def _finalize_data(self, data):
          for k in data:
               data[k] = data[k].astype('float32')

     def _mask(self, arr):
          #array = arr[self.maskEntries][self.mask_jetpt]
          array = arr[self.mask_jetpt]
          return array

     def _make_pfcands(self,table):
          data = {}
          cand_parents = table[self.fatjetpf_branch + '_jetIdx']#[self.maskEntries]
          #c = Counter((cand_parents.offsets[:-1] + cand_parents).content)
          #jet_cand_counts = awkward.JaggedArray.fromcounts(self.jetp4.counts, [c[k] for k in sorted(c.keys())])
          jet_cand_idxs = table[self.fatjetpf_branch + '_pFCandsIdx']#[self.maskEntries]

          def pf(var_name):
               branch_name = self.pfcand_branch + '_' + var_name
               if var_name[:4] == 'btag':
                    branch_name = self.fatjetpf_branch + '_' + var_name
               cand_arr = table[branch_name]#[self.maskEntries]
               cand_arr = cand_arr[jet_cand_idxs]
               return awkward.JaggedArray.fromiter([awkward.JaggedArray.fromparents(idx, a) if len(idx) else [] for idx, a in zip(cand_parents, cand_arr)])
               #return jet_cand_counts.copy(content=awkward.JaggedArray.fromcounts(jet_cand_counts.content, cand_arr.content))

          data['pfcand_VTX_ass'] = pf('pvAssocQuality')
          data['pfcand_lostInnerHits'] = pf('lostInnerHits')
          data['pfcand_quality'] = pf('trkQuality')
          
          pdgId = pf('pdgId')
          charge = pf('charge')
          data['pfcand_isEl'] = np.abs(pdgId) == 11
          data['pfcand_isMu'] = np.abs(pdgId) == 13
          data['pfcand_isChargedHad'] = np.abs(pdgId) == 211
          data['pfcand_isGamma'] = np.abs(pdgId) == 22
          data['pfcand_isNeutralHad'] = np.abs(pdgId) == 130
          data['pfcand_charge'] = charge

          dz = pf('dz')
          dxy = pf('d0')
          data['pfcand_dz'] = dz
          data['pfcand_dxy'] = dxy
          data['pfcand_dzsig'] = dz / pf('dzErr')
          data['pfcand_dxysig'] = dxy / pf('d0Err')

          candp4 = TLorentzVectorArray.from_ptetaphim(pf('pt'), pf('eta'), pf('phi'), pf('mass'))
          data['pfcand_mask'] = charge.ones_like()
          data['pfcand_phirel'] = candp4.delta_phi(self.jetp4)
          data['pfcand_etarel'] = self.eta_sign * (candp4.eta - self.jetp4.eta)
          data['pfcand_abseta'] = np.abs(candp4.eta)
          
          data['pfcand_pt_log_nopuppi'] = np.log(candp4.pt)
          data['pfcand_e_log_nopuppi'] = np.log(candp4.energy)
          
          chi2 = pf('trkChi2')
          chi2.content[chi2.content == -1] = 999
          data['pfcand_normchi2'] = np.floor(chi2)
          data['pfcand_btagEtaRel'] = pf('btagEtaRel')
          data['pfcand_btagPtRatio'] = pf('btagPtRatio')
          data['pfcand_btagPParRatio'] = pf('btagPParRatio')
          data['pfcand_btagSip3dVal'] = pf('btagSip3dVal')
          data['pfcand_btagSip3dSig'] = pf('btagSip3dSig')
          data['pfcand_btagJetDistVal'] = pf('btagJetDistVal')
          
          self._finalize_data(data)
          self.data.update(data)

     def _make_sv(self, table):
          data = {}
          all_svp4 = TLorentzVectorArray.from_ptetaphim(
               table[self.sv_branch + '_pt'],#[self.maskEntries],
               table[self.sv_branch + '_eta'],#[self.maskEntries],
               table[self.sv_branch + '_phi'],#[self.maskEntries],
               table[self.sv_branch + '_mass']#[self.maskEntries],
          )
          
          jet_cross_sv = self.jetp4.cross(all_svp4, nested=True)
          match = jet_cross_sv.i0.delta_r2(jet_cross_sv.i1) < self.jet_r2

          def sv(var_name):
              sv_arr = table[self.sv_branch + '_' + var_name]#[self.maskEntries]
              return self.jetp4.eta.cross(sv_arr, nested=True).unzip()[1][match]
              
          svp4 = TLorentzVectorArray.from_ptetaphim(sv('pt'), sv('eta'), sv('phi'), sv('mass'))
          data['sv_phirel'] = svp4.delta_phi(self.jetp4)
          data['sv_etarel'] = self.eta_sign * (svp4.eta - self.jetp4.eta)
          data['sv_abseta'] = np.abs(svp4.eta)
          data['sv_mass'] = svp4.mass
          data['sv_pt_log'] = np.log(svp4.pt)
          data['sv_mask'] = data['sv_pt_log'].ones_like()
          
          data['sv_ntracks'] = sv('ntracks')
          data['sv_normchi2'] = sv('chi2')
          data['sv_dxy'] = sv('dxy')
          data['sv_dxysig'] = sv('dxySig')
          data['sv_d3d'] = sv('dlen')
          data['sv_d3dsig'] = sv('dlenSig')
          data['sv_costhetasvpv'] = -np.cos(sv('pAngle'))
          
          dxysig = sv('dxySig')
          dxysig.content[~np.isfinite(dxysig.content)] = 0
          pos = dxysig.argsort()
          for k in data:
               data[k] = data[k][pos]
               
          self._finalize_data(data)
          self.data.update(data)
          
     def convert(self,table,entriesRange=None):
          self.data = {}
          if entriesRange:
               self.maskEntries = entriesRange
               self.mask_jetpt = (table[self.fatjet_branch + '_pt'][self.maskEntries] > 170)
               self.jetp4 = TLorentzVectorArray.from_ptetaphim(
                    self._mask(table[self.fatjet_branch + '_pt']),
                    self._mask(table[self.fatjet_branch + '_eta']),
                    self._mask(table[self.fatjet_branch + '_phi']),
                    self._mask(table[self.fatjet_branch + '_mass']),
               )
          else:
               self.mask_jetpt = (table[self.fatjet_branch + '_pt'] > 170)
               self.jetp4 = TLorentzVectorArray.from_ptetaphim(
                    self._mask(table[self.fatjet_branch + '_pt']),
                    self._mask(table[self.fatjet_branch + '_eta']),
                    self._mask(table[self.fatjet_branch + '_phi']),
                    self._mask(table[self.fatjet_branch + '_mass']),
               )
          self.eta_sign = self.jetp4.eta.ones_like()
          self.eta_sign[self.jetp4.eta <= 0] = -1
          
          self._make_pfcands(table)
          self._make_sv(table)
          self.data['_jetp4'] = self.jetp4
          return self.data

class InferenceProducer(Module):

     def __init__(self):
          self.tagInfoMaker = ParticleNetTagInfoMaker(fatjet_branch='FatJet', pfcand_branch='PFCands', sv_branch='SV', fatpfcand_branch='FatJetPFCands', jetR=0.8)
          self.pnTaggerMD = ParticleNetJetTagsProducer(
               os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetAK8/MD-2prong/V00/ParticleNetMD.onnx'),
               os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/ParticleNetAK8/MD-2prong/V00/preprocess.json'),
          )
     """
     def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree, entriesRange=None):
          self.out = wrappedOutputTree
          self.out.branch("FatJet_pnXqqVsQCD", "F", lenVar="nFatJet")
          
          uproot_tree = uproot.open(inputFile.GetName())['Events']
          table = uproot_tree.arrays(['FatJet_pt','FatJet_eta', 'FatJet_phi', 'FatJet_mass', '*FatJetPFCands*', 'PFCands*', 'SV*'],
                                     namedecode='utf-8')
          self._tagInfo = self.tagInfoMaker.convert(table, entriesRange)
          outputs = self.pnTaggerMD.predict(self._tagInfo)

          self._pn_XqqVsQCD = convert_prob(outputs, ['Xqq'], prefix='prob')

     def analyze(self, event, ievent):
          #process event, return True (go to next module) or False (fail, go to next event)

          # uncomment these lines for consistency check
          # fatjets = Collection(event, "FatJet")
          # for idx,jet in enumerate(fatjets):
          #      p4 = self._data['_jetp4'][entry_idx][idx]
          #      print('pt,eta,phi', (jet.pt, jet.eta, jet.phi), (p4.pt, p4.eta, p4.phi))    

          self.out.fillBranch("FatJet_pnXqqVsQCD", self._pn_XqqVsQCD[ievent])
          return True
     """

     def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree, entriesRange=None):
          self.out = wrappedOutputTree
          self.out.branch("FatJet_pnXqqVsQCD", "F", lenVar="nFatJet")
          self._uproot_tree = uproot.open(inputFile.GetName())['Events']

     def analyze(self, event, ievent):
          absolute_event_idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
          jets = Collection(event, "FatJet")
          if len(jets)>0:
               table = self._uproot_tree.arrays(['FatJet_pt','FatJet_eta', 'FatJet_phi', 'FatJet_mass', '*FatJetPFCands*', 'PFCands*', 'SV*'],
                                                namedecode='utf-8', entrystart=absolute_event_idx, entrystop=absolute_event_idx+1) 
               tagInfo = self.tagInfoMaker.convert(table)
               outputs = self.pnTaggerMD.predict(tagInfo)
               pn_XqqVsQCD = convert_prob(outputs, ['Xqq'], prefix='prob')
               self.out.fillBranch("FatJet_pnXqqVsQCD", pn_XqqVsQCD[0])
          return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
inferenceProducer = lambda : InferenceProducer()
