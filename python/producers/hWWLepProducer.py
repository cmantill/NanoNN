import os
import uproot
import itertools
import awkward
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import numpy as np
from collections import Counter
from uproot_methods import TLorentzVectorArray

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from PhysicsTools.NanoNN.helpers.utils import closest, sumP4, polarP4, configLogger, get_subjets, deltaPhi, deltaR
from PhysicsTools.NanoNN.helpers.runTF import INJetTagsProducer

from keras.models import load_model
import tensorflow as tf

def signedDeltaPhi(phi1, phi2):
    dPhi = phi1 - phi2
    if (dPhi < -np.pi):
        dPhi = 2 * np.pi + dPhi
    elif (dPhi > np.pi):
        dPhi = -2 * np.pi + dPhi
    return dPhi

class hWWLepProducer(Module):
    
    def __init__(self):
        self.jet_r = 0.8 
        
        prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data')
        self.tagger_channels = ['hadel','hadmu']
        self.inTaggers = [INJetTagsProducer('%s/INtautau/v4p1/preprocess.json'%prefix,
                                              '%s/INtautau/v4p1/IN_{version}_v4p1_on_TTbar_WJets_fillFactor=1_5_200GeV_ohe_take_1_model.h5'%prefix,version=channel) for channel in self.tagger_channels]
        
        self.n_pf = self.inTaggers[0].prep_params['pf_features']['var_length']
        self.n_sv = self.inTaggers[0].prep_params['sv_features']['var_length']
        
    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree, entriesRange=None):
        self.out = wrappedOutputTree
        self.out.branch("fj_pt", "F", 1)
        self.out.branch("fj_eta", "F", 1)
        self.out.branch("fj_phi", "F", 1)
        self.out.branch("fj_mass", "F", 1)
        self.out.branch("fj_msoftdrop", "F", 1)
        self.out.branch("fj_lsf3", "F", 1)
        self.out.branch("fj_IN_hadel_v4p1", "F", 1)
        self.out.branch("fj_IN_hadmu_v4p1", "F", 1)
        self.out.branch("fj_mindPhi", "I", 1)
        self.out.branch("fj_metdPhi", "F", 1)

        self.out.branch("fj_isQCD", "I", 1)
        self.out.branch("fj_isTop", "I", 1)
        self.out.branch("fj_isTopLep", "I", 1)
        self.out.branch("fj_isW", "I", 1)
        self.out.branch("fj_isWLep", "I", 1)
        
        self.out.branch("fj_H_WW_elenuqq", "I", 1)
        self.out.branch("fj_H_WW_munuqq", "I", 1)
        self.out.branch("fj_H_WW_taunuqq", "I", 1)
        self.out.branch("fj_genH_mass", "F", 1)
        self.out.branch("fj_genH_pt", "F", 1)
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
        self.out.branch("fj_genW_decay", "F", 1)
        self.out.branch("fj_genWstar_decay", "F", 1)
        self.out.branch("fj_nProngs", "I", 1)
          
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
        lepGenWs = []
        hadGenZs = []
        bbGenHs = []
        ccGenHs = []
        qqGenHs = []
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
                

            nProngs = 0
            daus = []
            if fj.genHbb and fj.dr_Hbb<self.jet_r: daus = fj.genHbb.daus
            elif fj.genHcc and fj.dr_Hcc<self.jet_r: daus = fj.genHcc.daus
            elif fj.genHqq and fj.dr_Hqq<self.jet_r: daus = fj.genHqq.daus
            elif fj.genHww and fj.dr_Hww<self.jet_r: daus = fj.genHww.daus
            for dau in daus:
                if deltaR(fj, dau)< self.jet_r: nProngs +=1
            fj.nProngs = nProngs

            fj.isHiggs = 0
            if fj.dr_Hww<self.jet_r or fj.dr_Hbb < self.jet_r or fj.dr_Hcc < self.jet_r or fj.dr_Hqq < self.jet_r:
                fj.isHiggs = 1

            fj.isTop = 0
            if fj.dr_T < self.jet_r: fj.isTop = 1

            fj.isTopLep = 0
            if fj.dr_LepT < self.jet_r: fj.isTopLep = 1

            fj.isW = 0
            if fj.dr_W < self.jet_r and fj.isHiggs == 0 and fj.isTop==0 and fj.isTopLep==0: 
                if not fj.genHbb and not fj.genHcc and not fj.genHqq and not fj.genHww and not fj.genLepT and not fj.genT:
                    fj.isW =1

            fj.isWLep = 0
            if fj.dr_LepW < self.jet_r and fj.isHiggs == 0 and fj.isTop==0 and fj.isTopLep==0: 
                if not fj.genHbb and not fj.genHcc and not fj.genHqq and not fj.genHww and not fj.genLepT and not fj.genT and not fj.genW:
                    fj.isWLep = 1

            fj.isQCD = 0
            if(fj.isHiggs==0 and fj.isTop==0 and fj.isW==0 and fj.isWLep==0 and fj.isTopLep==0):
                if not fj.genHbb and not fj.genHcc and not fj.genHqq and not fj.genHww and not fj.genLepT and not fj.genT and not fj.genZ and not fj.genW and not fj.genLepW:
                    fj.isQCD = 1

    def analyze(self, event, ievent):
         event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
         return True

    def fill(self, event, ievent):
        jets = Collection(event, "FatJet")
        svs = Collection(event, "SV")
        met = Object(event, "MET")
        pfcands = Collection(event, "PFCands")
        fjpfcands = Collection(event, "FatJetPFCands")

        # load gen history
        self.loadGenHistory(event, jets)
        
        print(event.idx)

        # avoid phi cut for now
        jet_idx = -1
        min_dphi = 999.
        for ij, jet in enumerate(jets):
            if (jet.pt < 200.): continue
            this_dphi = abs(signedDeltaPhi(met.phi, jet.phi))
            if (this_dphi < min_dphi):
                min_dphi = this_dphi
                jet_idx = ij

        for idx, fj in enumerate(jets):
            if (fj.pt < 200.): continue
            # this part relies on nConstituents
            # if (idx < jet_idx):
            #     pf_idx = pf_idx + fj.nConstituents
            #     continue
            # elif (idx > jet_idx):
            #     continue
            # if fj.nConstituents < 1: continue

            self.out.fillBranch("fj_pt", fj.pt)
            self.out.fillBranch("fj_eta", fj.eta)
            self.out.fillBranch("fj_phi", fj.phi)
            self.out.fillBranch("fj_mass", fj.mass)
            self.out.fillBranch("fj_msoftdrop", fj.msoftdrop)
            self.out.fillBranch("fj_lsf3", fj.lsf3)

            if idx == jet_idx:
                self.out.fillBranch("fj_mindPhi", 1)
            else:
                self.out.fillBranch("fj_mindPhi", 0)
                
            self.out.fillBranch("fj_metdPhi",  abs(signedDeltaPhi(met.phi, fj.phi)))
            
            # WW gen info
            dr_HWW_W = fj.dr_HWW_W if fj.dr_HWW_W else 99
            dR_HWW_Wstar = fj.dr_HWW_Wstar if fj.dr_HWW_Wstar else 99
            self.out.fillBranch("fj_H_WW_elenuqq", 1 if (fj.dr_HWW_elenuqq < self.jet_r and dr_HWW_W < self.jet_r and dR_HWW_Wstar < self.jet_r) else 0)
            self.out.fillBranch("fj_H_WW_munuqq", 1 if (fj.dr_HWW_munuqq < self.jet_r and dr_HWW_W < self.jet_r and dR_HWW_Wstar < self.jet_r) else 0)
            self.out.fillBranch("fj_H_WW_taunuqq", 1 if (fj.dr_HWW_taunuqq < self.jet_r and dr_HWW_W < self.jet_r and dR_HWW_Wstar < self.jet_r) else 0)
            
            genH_mass = -99
            genH_pt = -99
            if fj.genHww:
                genH_mass = fj.genHww.mass
                genH_pt = fj.genHww.pt
            elif fj.genHbb:
                genH_mass = fj.genHbb.mass
                genH_pt = fj.genHbb.pt
            elif fj.genHcc:
                genH_mass = fj.genHcc.mass
                genH_pt = fj.genHcc.pt
            elif fj.genHqq:
                genH_mass = fj.genHqq.mass
                genH_pt = fj.genHqq.pt
            self.out.fillBranch("fj_genH_mass", genH_mass)
            self.out.fillBranch("fj_genH_pt", genH_pt)
            
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
            
            ptt = fj.genHWW_W.pt if fj.genHWW_W else -99
            mm = fj.genHww.mass if fj.genHww else -99
            self.out.fillBranch("fj_maxdR_HWW_daus", max([deltaR(fj, dau) for dau in fj.genHww.daus]) if(fj.genHww and fj.dr_Hww < self.jet_r) else 99)
            self.out.fillBranch("fj_mindR_HWW_daus", min([deltaR(fj, dau) for dau in fj.genHww.daus]) if(fj.genHww and fj.dr_Hww < self.jet_r) else 99)
            
            self.out.fillBranch("fj_genW_decay", fj.genHWW_Wdecay if fj.genHWW_W else -99)
            self.out.fillBranch("fj_genWstar_decay", fj.genHWW_Wstardecay if fj.genHWW_Wstar else -99)
            self.out.fillBranch("fj_nProngs", fj.nProngs)
        
            self.out.fillBranch("fj_isQCD", fj.isQCD)
            self.out.fillBranch("fj_isTop", fj.isTop)
            self.out.fillBranch("fj_isTopLep", fj.isTopLep)
            self.out.fillBranch("fj_isW", fj.isW)
            self.out.fillBranch("fj_isWLep", fj.isWLep)

            jetv = ROOT.TLorentzVector()
            jetv.SetPtEtaPhiM(jet.pt, jet.eta, jet.phi, jet.mass)
            
            # fill sv information
            svpt = np.zeros(self.n_sv, dtype=np.float16)
            svdlen = np.zeros(self.n_sv, dtype=np.float16)
            svdlenSig = np.zeros(self.n_sv, dtype=np.float16)
            svdxy = np.zeros(self.n_sv, dtype=np.float16)
            svdxySig = np.zeros(self.n_sv, dtype=np.float16)
            svchi2 = np.zeros(self.n_sv, dtype=np.float16)
            svpAngle = np.zeros(self.n_sv, dtype=np.float16)
            svx = np.zeros(self.n_sv, dtype=np.float16)
            svy = np.zeros(self.n_sv, dtype=np.float16)
            svz = np.zeros(self.n_sv, dtype=np.float16)
            svmass = np.zeros(self.n_sv, dtype=np.float16)
            svphi = np.zeros(self.n_sv, dtype=np.float16)
            sveta = np.zeros(self.n_sv, dtype=np.float16)
            svv = ROOT.TLorentzVector()
            arrIdx = 0
            for isv, sv in enumerate(svs):
                if arrIdx == self.n_sv: break
                svv.SetPtEtaPhiM(sv.pt, sv.eta, sv.phi, sv.mass)
                if jetv.DeltaR(svv) < 0.8:
                    svpt[arrIdx] = sv.pt / fj.pt
                    svdlen[arrIdx] = sv.dlen
                    svdlenSig[arrIdx] = sv.dlenSig
                    svdxy[arrIdx] = sv.dxy
                    svdxySig[arrIdx] = sv.dxySig
                    svchi2[arrIdx] = sv.chi2
                    svpAngle[arrIdx] = sv.pAngle
                    svx[arrIdx] = sv.x
                    svy[arrIdx] = sv.y
                    svz[arrIdx] = sv.z
                    sveta[arrIdx] = sv.eta - fj.eta
                    svphi[arrIdx] = signedDeltaPhi(sv.phi, fj.phi)
                    svmass[arrIdx] = sv.mass
                    arrIdx += 1
                    
            # fill pf information
            # candrange = range(pf_idx, pf_idx + jet.nConstituents)
            candrange = [fjpf.pFCandsIdx for fjpf in fjpfcands if fjpf.jetIdx == idx]
            print('candrange',candrange)
            pfpt = np.zeros(self.n_pf, dtype=np.float16)
            pfeta = np.zeros(self.n_pf, dtype=np.float16)
            pfphi = np.zeros(self.n_pf, dtype=np.float16)
            pftrk = np.zeros(self.n_pf, dtype=np.float16)
            pfpup = np.zeros(self.n_pf, dtype=np.float16)
            pfpupnolep = np.zeros(self.n_pf, dtype=np.float16)
            pfq = np.zeros(self.n_pf, dtype=np.float16)
            pfid = np.zeros(self.n_pf, dtype=np.float16)
            pfdz = np.zeros(self.n_pf, dtype=np.float16)
            pfdxy = np.zeros(self.n_pf, dtype=np.float16)
            pfdxyerr = np.zeros(self.n_pf, dtype=np.float16)
            arrIdx = 0
            for ip, part in enumerate(pfcands):
                if ip not in candrange: continue
                if arrIdx == self.n_pf: break
                pfpt[arrIdx] = part.pt / fj.pt
                pfeta[arrIdx] = part.eta - fj.eta
                pfphi[arrIdx] = signedDeltaPhi(part.phi, fj.phi)
                pfpup[arrIdx] = part.puppiWeight
                pfpupnolep[arrIdx] = part.puppiWeightNoLep
                pfq[arrIdx] = part.charge
                pfid[arrIdx] = part.pdgId
                pfdz[arrIdx] = part.dz
                pfdxy[arrIdx] = part.d0
                pfdxyerr[arrIdx] = part.d0Err
                pftrk[arrIdx] = part.trkChi2
                arrIdx += 1

            pfData = np.vstack([pfpt, pfeta, pfphi, pfq, pfdz, pfdxy, pfdxyerr, pfpup, pfpupnolep, pfid])
            pfData = np.transpose(pfData)
            pfData = np.expand_dims(pfData,axis=0)
            svData = np.vstack([svdlen,svdlenSig, svdxy, svdxySig, svchi2, svpAngle, svx, svy, svz, svpt, svmass, sveta, svphi])
            svData = np.transpose(svData)
            svData = np.expand_dims(svData, axis=0)
            
            idconv = {211.:1, 13.:2,  22.:3,  11.:4, 130.:5, 1.:6, 2.:7, 3.:8, 4.:9,
                      5.:10, -211.:1, -13.:2,
                      -11.:4, -1.:-6, -2.:7, -3.:8, -4.:9, -5.:10, 0.:0}
            pfData[:,:,-1] = np.vectorize(idconv.__getitem__)(pfData[:,:,-1])          
            
            idlist = np.abs(pfData[:,:,-1]).astype(int)
            pfData = np.concatenate([pfData[:,:,:-1],np.eye(11)[idlist]],axis=-1)#relies on number of IDs being 11, be careful
            
            IN_hadel_v4p1 = float(self.inTaggers[0].model.predict([pfData, svData]))
            IN_hadmu_v4p1 = float(self.inTaggers[1].model.predict([pfData, svData]))

            self.out.fillBranch("fj_IN_hadel_v4p1", IN_hadel_v4p1)
            self.out.fillBranch("fj_IN_hadmu_v4p1", IN_hadmu_v4p1)

            self.out.fill()

        return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
hwwLepProduder = lambda : hwwLepProducer()
