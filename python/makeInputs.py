import os
import uproot
import awkward as ak
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import numpy as np
from uproot_methods import TLorentzVectorArray
from collections import Counter
import itertools 

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

     def _get_array(self, table, arr, maskjet=False, maskpf=False):
          if maskpf:
               return table[arr][self.mask_id]
          elif maskjet:
               return table[arr][self.mask_jet][(self.mask_jet).any()]
          else:
               return table[arr][(self.mask_jet).any()]

     def _make_pfcands(self,table):
          data = {}

          cand_parents = self._get_array(table,self.fatjetpf_branch + '_jetIdx', False, True)
          jet_cand_idxs = self._get_array(table,self.fatjetpf_branch + '_pFCandsIdx', False, True)

          c = Counter((cand_parents.offsets[:-1] + cand_parents).content)
          jet_cand_counts = ak.JaggedArray.fromcounts(self.jetp4.counts, [c[k] for k in sorted(c.keys())])

          def pf(var_name):
               branch_name = self.pfcand_branch + '_' + var_name
               if var_name[:4] == 'btag':
                    branch_name = self.fatjetpf_branch + '_' + var_name
               cand_arr = self._get_array(table,branch_name,False,True)
               out = jet_cand_counts.copy(
                    content=ak.JaggedArray.fromcounts(jet_cand_counts.content, cand_arr.content))
               return out

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

          self._finalize_data(data)

          data['_isHiggs'] = np.any(np.abs(self._get_array(table,'GenPart_pdgId')==25)[0]).astype('int')
          data['_isTop'] = np.any(np.abs(self._get_array(table,'GenPart_pdgId')==6)[0]).astype('int')

          self.data['_jet_H_WW_4q'] = self.data['_jetp4'].pt.zeros_like().astype('int')
          self.data['_jet_H_WW_elenuqq'] = self.data['_jetp4'].pt.zeros_like().astype('int')
          self.data['_jet_H_WW_munuqq'] = self.data['_jetp4'].pt.zeros_like().astype('int')
          self.data['_jet_nProngs'] = self.data['_jetp4'].pt.zeros_like().astype('int')
          self.data['_jet_dR_W'] = self.data['_jetp4'].pt.zeros_like()
          self.data['_jet_dR_Wstar'] = self.data['_jetp4'].pt.zeros_like()

          if data['_isHiggs'] > 0:
               gen = {'mom': self._get_array(table, 'GenPart_genPartIdxMother'),
                      'id': np.abs(self._get_array(table, 'GenPart_pdgId')),
                      'pt': self._get_array(table, 'GenPart_pt'),
                      'eta': self._get_array(table, 'GenPart_eta'),
                      'phi': self._get_array(table, 'GenPart_phi'),
                      'mass': self._get_array(table, 'GenPart_mass'),
                      'status': self._get_array(table, 'GenPart_status'),
                 }
                    
               def getBosons(gen,idp=25):
                    mask = (gen['id'] == idp) & (gen['status']==22)
                    idx = ak.JaggedArray.fromiter([np.where(mask[index]) for index in range(len(gen['id']))])
                    vs = TLorentzVectorArray.from_ptetaphim(
                         gen['pt'][mask],
                         gen['eta'][mask],
                         gen['phi'][mask],
                         gen['mass'][mask],
                    )
                    return idx.flatten(),vs
                    
               def getFinal(gen,mom,index,pdgId=24):
                    mask = np.isin(gen['mom'][index],mom)
                    maskW = (gen['id'][index] == pdgId) & mask
                    if maskW.any():
                         idxs = np.where(maskW)[0]
                         maskd = np.isin(gen['mom'][index],idxs)
                         if ((gen['id'][index] == pdgId) & maskd).any():
                              return getFinal(gen,idxs,index,pdgId)
                         else:
                              return np.where(maskW)[0]
                    else:
                         return mom


               def getWdaus(gen,mom,index):
                    idxW = getFinal(gen,mom,index)
                    maskd = np.isin(gen['mom'][index],idxW) 
                    pdgid = gen['id'][index][maskd]

                    masks = {'ele': (pdgid==11) | (pdgid==12),
                             'mu': (pdgid==13) | (pdgid==14),
                             'tau': (pdgid==15) | (pdgid==16),
                             'q': (pdgid<=5)
                        }

                    def matchIndex(genP,index):
                         if len(genP)>0:
                              matchP = []
                              for j in self.data['_jetp4'][index]:
                                   matchP.append([j.delta_r2(g) < self.jet_r2 for g in genP])
                              return matchP
                         else:
                              return np.zeros(len(self.data['_jetp4'][index].pt), dtype=int)

                    matchD = matchIndex(
                         TLorentzVectorArray.from_ptetaphim(
                              gen['pt'][index][maskd],
                              gen['eta'][index][maskd],
                              gen['phi'][index][maskd],
                              gen['mass'][index][maskd],
                         ), index
                    )
                    match = {}
                    for key,im in masks.items():
                         matchdau = matchIndex(
                              TLorentzVectorArray.from_ptetaphim(
                                   gen['pt'][index][maskd][im],
                                   gen['eta'][index][maskd][im],
                                   gen['phi'][index][maskd][im],
                                   gen['mass'][index][maskd][im]
                              ), index
                         )
                         try:
                              match[key] = np.any(matchdau,axis=1)
                         except:
                              match[key] = matchdau>0

                    try:
                         nProngs = np.sum(matchD,axis=1)
                    except:
                         nProngs = matchD
                    jetHWWqq = {}
                    for key in ['mu','tau','ele','q']:
                         try: 
                              jetHWWqq[key] = (match[key] & match['q']).astype('int')
                         except:
                              jetHWWqq[key] = (match[key] & match['q'])

                    if((gen['id'][index][maskd] <=5).any()):
                         for ij in range(len(jetHWWqq['q'])):
                              if(jetHWWqq['mu'][ij]==0 and jetHWWqq['ele'][ij]==0 and jetHWWqq['tau'][ij]==0 and jetHWWqq['q'][ij]==1):
                                   jetHWWqq['q'][ij] = 1
                              else:
                                   jetHWWqq['q'][ij] = 0
                         
                    return nProngs,jetHWWqq

               def getWs(gen,mom):
                    genW = {}
                    for key in ['W','Won','Woff']:
                         genW[key] = {'pt': [], 'eta': [], 'phi': [], 'mass': [], 'idx':[]}

                    genW['W']['nprong'] = []
                    genW['W']['munuqq'] = []
                    genW['W']['elenuqq'] = []
                    genW['W']['taunuqq'] = []
                    genW['W']['qqqq'] = []

                    for index in range(len(gen['mom'])):
                         igenW = {}
                         for key in genW.keys():
                              igenW[key] = {'pt': [], 'eta': [], 'phi': [], 'mass': [], 'idx':[]}

                         nprongs_W = []
                         jetHWWmunuqq_W = []
                         jetHWWelenuqq_W = []
                         jetHWWtaunuqq_W = []
                         jetHWWqqqq_W = []

                         for midx in range(len(mom[index])):
                              idxH = getFinal(gen,mom[index][midx],index,25)
                              mask = (gen['id'][index] == 24) & np.isin(gen['mom'][index],idxH)
                              msort = gen['mass'][index][mask].argsort() # sort Ws by mass

                              igenW['W']['idx'].append(list(np.where(mask.flatten()))[0])
                              for prop in igenW['W'].keys():
                                   if prop=='idx': continue
                                   igenW['W'][prop].append(gen[prop][index][mask][msort])

                              if len(gen['pt'][index][mask][msort])>0:
                                   for prop in igenW['Won'].keys():
                                        if prop=='idx': continue
                                        igenW['Won'][prop].append(gen[prop][index][mask][msort][0])
                                   if len(gen['pt'][index][mask][msort])>1:
                                        for prop in igenW['Woff'].keys():
                                             if prop=='idx': continue
                                             igenW['Woff'][prop].append(gen[prop][index][mask][msort][1])

                              nprongs,jetHWWqq = getWdaus(gen,np.where(mask.flatten())[0],index)
                              nprongs_W.append(nprongs)
                              jetHWWmunuqq_W.append(jetHWWqq['mu'])
                              jetHWWelenuqq_W.append(jetHWWqq['ele'])
                              jetHWWtaunuqq_W.append(jetHWWqq['tau'])
                              jetHWWqqqq_W.append(jetHWWqq['q'])

                         nprongs_W = np.array(nprongs_W)

                         def pad_with_zeros(A):
                              out = np.zeros(len(self.data['_jetp4'][index].pt),dtype=int)
                              r_= len(A)
                              if(len(A)>len(self.data['_jetp4'][index].pt)):
                                   out = A[0:len(self.data['_jetp4'][index].pt)]
                              else:
                                   out[0:r_] = A
                              return out
                              
                         npro = pad_with_zeros(nprongs_W[nprongs_W.nonzero()])
                         genW['W']['nprong'].append(npro)

                         def compress(arr_W):
                              arr = [0]
                              for ik,k in enumerate(arr_W):
                                   if ik==0:
                                        arr = arr_W[ik]
                                   else:
                                        arr |= arr_W[ik]
                              return arr
                         genW['W']['munuqq'].append(pad_with_zeros(compress(jetHWWmunuqq_W)))
                         genW['W']['elenuqq'].append(pad_with_zeros(compress(jetHWWelenuqq_W)))
                         genW['W']['taunuqq'].append(pad_with_zeros(compress(jetHWWtaunuqq_W)))
                         genW['W']['qqqq'].append(pad_with_zeros(compress(jetHWWqqqq_W)))

                         for key,gl in genW.items():
                              for prop in ['pt','eta','phi','mass','idx']:
                                   gl[prop].append(igenW[key][prop])

                    genWs = {}
                    for key,w in genW.items():
                         genWs[key] = TLorentzVectorArray.from_ptetaphim(
                              ak.JaggedArray.fromiter(genW[key]['pt']),
                              ak.JaggedArray.fromiter(genW[key]['eta']),
                              ak.JaggedArray.fromiter(genW[key]['phi']),
                              ak.JaggedArray.fromiter(genW[key]['mass']),
                         )

                    nprongs = ak.JaggedArray.fromiter(genW['W']['nprong'])
                    munuqq = ak.JaggedArray.fromiter(genW['W']['munuqq'])
                    elenuqq = ak.JaggedArray.fromiter(genW['W']['elenuqq'])
                    taunuqq = ak.JaggedArray.fromiter(genW['W']['taunuqq'])
                    qqqq = ak.JaggedArray.fromiter(genW['W']['qqqq'])
                    return nprongs,munuqq,elenuqq,taunuqq,qqqq,genWs['Won'],genWs['Woff']

               # finding Higgs
               genHidx,genH = getBosons(gen)
               jet_cross_genH = genH.cross(self.data['_jetp4'], nested=True)
               matchH = jet_cross_genH.i0.delta_r2(jet_cross_genH.i1) < self.jet_r2
               genHmatch = ak.JaggedArray.fromiter(genHidx[matchH.any()])

               # only get Ws from Hs that are matched to a jet
               nProngs, munuqq,elenuqq,taunuqq,qqqq, genW0, genW1 = getWs(gen,genHmatch)
               jet_cross_genW0 =  self.data['_jetp4'].cross(genW0)
               self.data['_jet_dR_W'] = jet_cross_genW0.i0.delta_r2(jet_cross_genW0.i1).pad(2).fillna(0)
               jet_cross_genW1 =  self.data['_jetp4'].cross(genW1)
               self.data['_jet_dR_Wstar'] = jet_cross_genW1.i0.delta_r2(jet_cross_genW1.i1).pad(2).fillna(0)
               self.data['_jet_nProngs'] = nProngs
               self.data['_jet_H_WW_munuqq'] = munuqq
               self.data['_jet_H_WW_elenuqq'] = elenuqq
               self.data['_jet_H_WW_4q'] = qqqq

          self.data.update(data)

     def convert(self,table,is_input=False):
          self.data = {}
          self.mask_jet = (table[self.fatjet_branch + '_pt'] > 300.) & (table[self.fatjet_branch + '_msoftdrop'] > 20.)
          nj = self.mask_jet.astype('int').sum()
          self.mask_id = (ak.JaggedArray.fromiter([np.isin(table[self.fatjetpf_branch + '_jetIdx'][index],list(range(0,nj[index]))) for index in range(len(table[self.fatjetpf_branch + '_jetIdx']))]))

          self.jetp4 = TLorentzVectorArray.from_ptetaphim(
               self._get_array(table, self.fatjet_branch + '_pt', True),
               self._get_array(table,self.fatjet_branch + '_eta', True),
               self._get_array(table,self.fatjet_branch + '_phi', True),
               self._get_array(table,self.fatjet_branch + '_mass', True),
          )
          self.eta_sign = self.jetp4.eta.ones_like()
          self.eta_sign[self.jetp4.eta <= 0] = -1

          if(self.mask_jet.any().any()):
               self._make_pfcands(table)
               self._make_sv(table)
               self.data['_jetp4'] = self.jetp4

               if is_input:
                    self._make_jet(table)

               return self.data
          else:
               return None

     def init_file(self, inputFile, fetch_step=1000):
          self._uproot_basketcache = uproot.cache.ThreadSafeArrayCache('200MB')
          self._uproot_keycache = uproot.cache.ThreadSafeArrayCache('10MB')
          self._uproot_tree = uproot.open(inputFile.GetName())['Events']
          self._uproot_fetch_step = fetch_step
          self._uproot_start = 0
          self._uproot_stop = 0
          self._taginfo = None

     def load(self, event_idx, tag_info_len, is_input=False):
          if event_idx >= self._uproot_stop:
               self._uproot_start = event_idx
               self._uproot_stop = self._uproot_start + self._uproot_fetch_step
               table = self._uproot_tree.arrays(['FatJet_pt','FatJet_eta', 'FatJet_phi', 'FatJet_mass',
                                                 'FatJet_msoftdrop','FatJet_deepTag_H','FatJet_deepTag_QCD','FatJet_deepTag_QCDothers',
                                                 '*FatJetPFCands*', 'PFCands*', 'SV*',
                                                 'GenPart_*'],
                                                namedecode='utf-8',
                                                entrystart=self._uproot_start, entrystop=self._uproot_stop,
                                                basketcache=self._uproot_basketcache, keycache=self._uproot_keycache,
                                           )
               self._taginfo = self.convert(table,is_input)
               tag_info_len += len(self._taginfo['_jetp4'].pt)

          return self._taginfo, tag_info_len
