# these functions help to detect mispronunciations using editops
#
#

# standard libs
import argparse
# third-party libs
import Levenshtein as lev
# local libs
from speech.utils.data_helpers import path_to_id
from speech.utils.io import read_data_json


def main(hypo_path:str, tgt_path:str, eval_phn_path:str):
    """This function will aim to detect mispronunciations of the `target_phn`
    in the predictions in `hypo_path` when compared with the reference in `phn_path`

    Args:
        hypo_path: path to model predictions
        tgt_path: path to phones the speaker should have said
        eval_phn_path: path to one-hot encoded labels of evaluation phonemes

    Notes:
        hypo_path file is formatted as:
            ay ih t (None-0)
            ao r dh ah t ay m (None-7)
            ay l iy v d uw (None-6)
           
        tgt_path file is formatted as: 
            ay iy t
            p r ih t iy sh
            dh ah jh ih m

        eval_phn_path:
            id              l       r       dh      p       v
            00F931A9-6EA9-4233-85B4-94015A257352    1       0       0       0       1       0
            012C1AC5-13E0-4337-B6CC-BFD58A12A8BC    1       1       0       0       0       0
            054C13A4-9499-453F-90A0-950DA50C4576    1       0       1       0       0       0

    """
    hypo_dict = {}
    with open(hypo_path, 'r') as hypo_f:
        for line in hypo_f:
            line = line.strip().split()
            phones = line[:-1]
            # line last element has format '(None-1)'
            hypo_id = int(line[-1].split('-')[1].replace(')', ''))
            
            hypo_dict[hypo_id] = phones

    # create mapping from record_id to hypo numerical ordering
    tsv_path = tgt_path.replace(".phn", ".tsv")
    id_to_order = {}
    with open(tsv_path, 'r') as tsv_f:
        _ = next(tsv_f)
        for i, line in enumerate(tsv_f):
            sub_path = line.strip().split('\t', maxsplit=1)[0]
            id_to_order[path_to_id(sub_path)] = i
    
    ord_to_eval_phns = read_eval_file(eval_phn_path, id_to_order)

    with open(tgt_path, 'r') as phn_f:
        for i, line in enumerate(phn_f):
            ref_phns = line.strip().split()
            hyp_phns = hypo_dict[i]
            
            edit_ops = get_editops(hyp_phns, ref_phns)
            try:
                rec_id, has_mispro, eval_phns = ord_to_eval_phns[i]
            except KeyError as e:
                print(f"Key error at index: {i} with line: {line}")
                raise e
            for eval_phn in eval_phns:
                print(f"record id: {rec_id}")
                print(f"evaluation phone: {eval_phn}")
                print(f"has mispro: {bool(has_mispro)}")
                print_editops(edit_ops, hyp_phns, ref_phns)
                mispro_detected = check_mispro(edit_ops, hyp_phns, ref_phns, eval_phn)
                print(f"mispro detected?: {mispro_detected}")
                print(f"detector is correct?: {has_mispro == mispro_detected}")           
 
                print('\n\n')


def assess_from_json(eval_phn_path, ds_json_path):
    ds_preds = read_data_json(ds_json_path)
    
    rec_to_eval_phns = read_eval_file(eval_phn_path)
    
    for xmpl in ds_preds:
        ref_phns = xmpl['label']
        hyp_phns = xmpl['prediction']
        
        edit_ops = get_editops(hyp_phns, ref_phns)
        rec_id = path_to_id(xmpl['filename'])
        rec_id, has_mispro, eval_phns = rec_to_eval_phns[rec_id]
        for eval_phn in eval_phns:
            print(f"record id: {rec_id}")
            print(f"evaluation phone: {eval_phn}")
            print(f"has mispro: {bool(has_mispro)}")
            print_editops(edit_ops, hyp_phns, ref_phns)
            mispro_detected = check_mispro(edit_ops, hyp_phns, ref_phns, eval_phn)
            print(f"mispro detected?: {mispro_detected}")
            print(f"detector is correct?: {has_mispro == mispro_detected}")           

            print('\n\n')


def read_eval_file(eval_phn_path:str, id_to_order:dict=None)->dict:
    """Reads the eval-phn file that contains information on the mispronunciations
    for each record and returns that information as a mapping from record to phonemes.

    Args:
        eval_phn_path: path to eval file
        id_to_order: mapping from record_id to the ordering. used for w2v files
    Returns:
        dict: mapping record_id or order to target phonemes information
    """
    with open(eval_phn_path, 'r') as lbl_f:
        header = next(lbl_f).strip().split()
        phn_hdr = header[2:]

        rec_to_eval_phns = {}        
        for line in lbl_f:
            line =  line.strip().split('\t')
            rec_id, has_mispro, row_lbl = line[0], int(line[1]), list(map(int, line[2:]))
            eval_phns = [phn_hdr[i] for i, one_h in enumerate(row_lbl) if one_h ==1]
            key = id_to_order[rec_id] if id_to_order else rec_id
            rec_to_eval_phns[key] = (rec_id, has_mispro, eval_phns)
    
    return rec_to_eval_phns


def check_mispro(edit_ops, hyp_phns, ref_phns, target_phn):

    hyp_phns, ref_phns = balance_phn_lengths(edit_ops, hyp_phns, ref_phns)
    
    mispro_detected = False
    for op, spos, dpos in edit_ops:
        if target_phn in ref_phns[dpos]:
            # don't include delete operations when assessing mispro
            if op == 'delete':
                continue
            else:
                # if target_phn is in both the hypo and tgt
                # handles cases where `r` is replaced by `er`, which is not a mispro
                if target_phn in hyp_phns[spos] and target_phn in ref_phns[dpos]:
                    continue
                else:
                    mispro_detected = True
   
    return mispro_detected 


def balance_phn_lengths(edit_ops, s_phns, d_phns):
    """lengths the source_phones or dest_phones if the indices in editops are
    greater than the lengths of the respective phoneme lists"""

    for _, spos, dpos in edit_ops:
        if spos > len(s_phns)-1:
            s_phns += ['blank'] * (spos - (len(s_phns)-1))
        if dpos > len(d_phns)-1:
            d_phns += ['blank'] * (dpos - (len(d_phns)-1))

    return s_phns, d_phns


def get_editops(hyp_phns, ref_phns):
    phn_super_set = set(hyp_phns + ref_phns)
    p2c = {ph:chr(65+i) for i, ph in enumerate(sorted(phn_super_set))}
    c2p = {chr(65+i):ph for i, ph in enumerate(sorted(phn_super_set))}
    hyp_chars = "".join([p2c[ph] for ph in hyp_phns])
    ref_chars = "".join([p2c[ph] for ph in ref_phns])

    return lev.editops(hyp_chars, ref_chars)

    
def print_editops(edit_ops, hyp_phns, ref_phns):
    print(f"hypos: {hyp_phns}")
    print(f"tgts: {ref_phns}")
    hyp_phns, ref_phns = balance_phn_lengths(edit_ops, hyp_phns, ref_phns)
    
    for op, spos, dpos in edit_ops:
        try:
            print(
                '{:7}   s[{}] --> d[{}] {!r:>8} --> {!r}'.\
                format(op, spos, dpos, hyp_phns[spos], ref_phns[dpos])
            )
        except IndexError as e:
            print("Index Error")
            print(op, spos, dpos, hyp_phns, ref_phns)
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--action", help="determines what function to call"
    )   
    parser.add_argument(
        "--hypo-path", help="path to w2v predictions"
    )
    parser.add_argument(
        "--json-path", help="path to json prediction for deepspeech model"
    )
    parser.add_argument(
        "--phn-path", help="path to w2v predictions"
    )
    parser.add_argument(
        "--eval-phn-path", type=str, help="path to one-hot encoding for evaluation phonemes by utterance id"
    )  
    args = parser.parse_args()
    if args.action == "": 
        main(args.hypo_path, args.phn_path, args.eval_phn_path)
    elif args.action == "assess-from-json":
        assess_from_json(args.eval_phn_path, args.json_path)
