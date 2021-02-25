import os
import uproot
import awkward
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import numpy as np
from uproot_methods import TLorentzVectorArray

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

     def _get_array(self, table, arr):
          mask = ((table[self.fatjet_branch + '_pt'] > 300) & (table[self.fatjet_branch + '_msoftdrop'] > 0)).any()
          return table[arr][mask]

     def _make_pfcands(self,table):
          data = {}
          cand_parents = self._get_array(table,self.fatjetpf_branch + '_jetIdx')
          jet_cand_idxs = self._get_array(table,self.fatjetpf_branch + '_pFCandsIdx')

          def pf(var_name):
               branch_name = self.pfcand_branch + '_' + var_name
               if var_name[:4] == 'btag':
                    branch_name = self.fatjetpf_branch + '_' + var_name
               cand_arr = self._get_array(table,branch_name)
               cand_arr = cand_arr[jet_cand_idxs]
               return awkward.JaggedArray.fromiter([awkward.JaggedArray.fromparents(idx, a) if len(idx) else [] for idx, a in zip(cand_parents, cand_arr)])

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
               self._get_array(table,self.sv_branch + '_pt'),
               self._get_array(table,self.sv_branch + '_eta'),
               self._get_array(table,self.sv_branch + '_phi'),
               self._get_array(table,self.sv_branch + '_mass'),
          )
          
          jet_cross_sv = self.jetp4.cross(all_svp4, nested=True)
          match = jet_cross_sv.i0.delta_r2(jet_cross_sv.i1) < self.jet_r2

          def sv(var_name):
              sv_arr =  self._get_array(table,self.sv_branch + '_' + var_name)
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
          
     def _make_jet(self, table):
          from PhysicsTools.NanoNN.nnHelper import convert_prob
          data = {}
          deepTag = {}
          deepTag['probH'] = self._get_array(table, self.fatjet_branch + '_deepTag_H')
          deepTag['probQCD'] = self._get_array(table, self.fatjet_branch + '_deepTag_QCD')
          deepTag['probQCDothers'] = self._get_array(table, self.fatjet_branch + '_deepTag_QCDothers')
          data['_jet_deepTagHvsQCD'] = convert_prob(deepTag, ['H'], prefix='prob', bkgs=['QCD','QCDothers'])
          data['_jet_deepTagMD_H4qvsQCD'] = self._get_array(table, self.fatjet_branch + '_deepTagMD_H4qvsQCD')
          data['_jet_lsf3'] =  self._get_array(table, self.fatjet_branch + '_lsf3')

          self._finalize_data(data)
          self.data.update(data)

     def convert(self,table,is_input=False):
          self.data = {}
          self.jetp4 = TLorentzVectorArray.from_ptetaphim(
               self._get_array(table, self.fatjet_branch + '_pt'),
               self._get_array(table, self.fatjet_branch + '_eta'),
               self._get_array(table, self.fatjet_branch + '_phi'),
               self._get_array(table, self.fatjet_branch + '_mass'),
          )
          self.eta_sign = self.jetp4.eta.ones_like()
          self.eta_sign[self.jetp4.eta <= 0] = -1
          
          self._make_pfcands(table)
          self._make_sv(table)
          self.data['_jetp4'] = self.jetp4

          if is_input:
               self._make_jet(table)

          return self.data
