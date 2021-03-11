import os
import numpy as np
import awkward
import onnxruntime
import json
import traceback
import logging

def _pad(a, min_length, max_length, value=0, dtype='float32'):
    if len(a) > max_length:
        return a[:max_length].astype(dtype)
    elif len(a) < min_length:
        x = (np.ones(min_length) * value).astype(dtype)
        x[:len(a)] = a.astype(dtype)
        return x
    else:
        return a.astype('float32')

def configLogger(name, loglevel=logging.INFO, filename=None):
    # define a Handler which writes INFO messages or higher to the sys.stderr
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    console = logging.StreamHandler()
    console.setLevel(loglevel)
    console.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(console)
    if filename:
        logfile = logging.FileHandler(filename)
        logfile.setLevel(loglevel)
        logfile.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s'))
        logger.addHandler(logfile)

logger = logging.getLogger('NanoNN')
configLogger('NanoNN', loglevel=logging.INFO)

class ParticleNetJetTagsProducer(object):

    def __init__(self, preprocess_path, model_path=None, debug=False):
        self.debug = debug
        with open(preprocess_path) as fp:
            self.prep_params = json.load(fp)
        if model_path:
            logger.info('Loading model %s' % model_path) 
            self.sess = onnxruntime.InferenceSession(model_path)

    def _preprocess(self, taginfo, eval_flags=None):
        data = {}
        counts = None
        for group_name in self.prep_params['input_names']:
            data[group_name] = []
            info = self.prep_params[group_name]
            for var in info['var_names']:
                a = taginfo[var]
                if eval_flags is not None:
                    a = a[eval_flags]
                if counts is None:
                    counts = a.counts
                else:
                    assert(np.array_equal(counts, a.counts))
                a = (a - info['var_infos'][var]['median']) * info['var_infos'][var]['norm_factor']
                a = a.flatten().pad(info['var_length'], clip=True).fillna(0).regular()
                a = np.clip(a, info['var_infos'][var].get('lower_bound', -5), info['var_infos'][var].get('upper_bound', 5))
                if self.debug:
                    print(var, a)
                data[group_name].append(a.astype('float32'))
            data[group_name] = np.nan_to_num(np.stack(data[group_name], axis=1))
        return data, counts

    def pad_one(self, taginfo, ievent, jet_idx):
        data = {}
        for group_name in self.prep_params['input_names']:
            data[group_name] = {}
            info = self.prep_params[group_name]
            for var in info['var_names']:
                a = taginfo[var][ievent][jet_idx]
                a = _pad(a, min_length=info['var_length'], max_length=info['var_length'])
                data[group_name][var] = a.astype('float32')
        return data

    def predict(self, taginfo, eval_flags=None):
        data, counts = self._preprocess(taginfo, eval_flags)
        preds = self.sess.run([], data)[0]
        outputs = {flav:awkward.JaggedArray.fromcounts(counts, preds[:, i]) for i, flav in enumerate(self.prep_params['output_names'])}
        return outputs

    def predict_one(self, taginfo, event_idx, jet_idx):
        data = {}
        for group_name in self.prep_params['input_names']:
            data[group_name] = []
            info = self.prep_params[group_name]
            for var in info['var_names']:
                a = taginfo[var][event_idx][jet_idx]
                a = (a - info['var_infos'][var]['median']) * info['var_infos'][var]['norm_factor']
                a = np.clip(a, info['var_infos'][var].get('lower_bound', -5), info['var_infos'][var].get('upper_bound', 5))
                try:
                    a = _pad(a, min_length=info['min_length'], max_length=info['max_length'])
                except KeyError:
                    a = _pad(a, min_length=info['var_length'], max_length=info['var_length'])
                if self.debug:
                    print(var, a)
                data[group_name].append(a.astype('float32'))
            data[group_name] = np.nan_to_num(np.expand_dims(np.stack(data[group_name], axis=0), 0))
        preds = self.sess.run([], data)[0]
        outputs = {flav:preds[0, i] for i, flav in enumerate(self.prep_params['output_names'])}
        return outputs

if __name__ == '__main__':
    import time
    import uproot
    import argparse
    parser = argparse.ArgumentParser('TEST')
    parser.add_argument('-i', '--input')
    parser.add_argument('-m', '--model')
    parser.add_argument('-p', '--preprocess')
    parser.add_argument('--make_baseline', action='store_true')
    args = parser.parse_args()

    from PhysicsTools.NanoNN.makeInputs import ParticleNetTagInfoMaker
    from PhysicsTools.NanoNN.nnHelper import convert_prob

    fatjet_name = 'FatJet'
    p = ParticleNetTagInfoMaker(fatjet_branch='FatJet', pfcand_branch='PFCands', sv_branch='SV', fatpfcand_branch='FatJetPFCands', jetR=0.8)
    tree = uproot.open(args.input)['Events']
    table = tree.arrays(['FatJet_pt','FatJet_eta', 'FatJet_phi', 'FatJet_mass', 'FatJet_msoftdrop', '*FatJetPFCands*', 'PFCands*', 'SV*',
                         'FatJet_particleNetMD_Xbb'],
                        namedecode='utf-8', entrystart=1, entrystop=2)
    start = time.time()
    taginfo = p.convert(table)
    diff = time.time() - start
    print('--- Convert inputs: %f s total, %f s per jet ---' % (diff, diff / taginfo['pfcand_mask'].counts.sum()))
    # jetmass = tree.array('FatJet_msoftdrop')
    # eval_flags = (jetmass > 50) * (jetmass < 200)
    # jetmass = jetmass[eval_flags]
    eval_flags = None

    start = time.time()
    nn = ParticleNetJetTagsProducer(args.preprocess, args.model)
    diff = time.time() - start
    print('--- Setup model: %f s total' % (diff,))

    start = time.time()
    outputs = nn.predict(taginfo, eval_flags)
    diff = time.time() - start
    print('--- Run prediction: %f s total, %f s per jet ---' % (diff, diff / outputs['probQCDbb'].counts.sum()))
    # print(outputs)
    # for k in outputs:
    #  print(k, outputs[k].content.mean())

    if fatjet_name + '_particleNetMD_Xbb' in table:
        print('Compare w/ stored values')
        print('Stored values:\n ...', table[fatjet_name + '_particleNetMD_Xbb'][:5])
        print('Computed values:\n ...', outputs['probXbb'][:5])
        print('Diff (50%, 95%, 99%, 100%) = ',
              np.percentile(np.abs(outputs['probXbb'] - table[fatjet_name + '_particleNetMD_Xbb']).content, [50, 95, 99, 100])
              )

    # assert(np.array_equal(jetmass.counts, outputs['probQCDbb'].counts))
    alloutputs = awkward.JaggedArray.zip(outputs)
    if args.make_baseline:
        with open('baseline.awkd', 'wb') as fout:
            awkward.save(fout, alloutputs)
    else:
        if os.path.exists('baseline.awkd'):
            with open('baseline.awkd', 'rb') as fin:
                baseline = awkward.load(fin)
            print("Comparison to baseline:", (alloutputs == baseline).all().all())
