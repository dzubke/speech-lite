# standard library
import argparse
# project libraries
from speech.utils.data_helpers import (
    combine_lexicon_helper, create_master_lexicon, lexicon_to_dict
)


def combine_lexicons(
    lex_1_path:str, 
    lex_2_path:str,
    lex_1_name:str,
    lex_2_name:str,
    save_path:str
    )->None:

    lex_1_dict = lexicon_to_dict(lex_1_path, corpus_name=lex_1_name)
    lex_2_dict = lexicon_to_dict(lex_2_path, corpus_name=lex_2_name)
    
    combined_dict, diff_labels = combine_lexicon_helper(lex_1_dict, lex_2_dict)
    
    print(f"words with different phoneme labels are: \n {diff_labels}")                                                         
    print(f"number of words with different labels: {len(diff_labels)}")

    with open(save_path, 'w') as fid:
        for word, phones in sorted(combined_dict.items()):
            fid.write(f"{word} {' '.join(phones)}\n")
    

def make_master_lexicon(cmu_lex_fn:str, ted_lex_fn:str, libsp_lex_fn:str, lex_out_fn:str)->None:
    """
    This function creates a pronunciation dictionary for common voice
    from the pronunciation dict of cmudict, tedlium and librispeech.
    Arguments:
        cmu_lex_fn - str: filename to cmudict lexicon
        ted_lex_fn - str: filename to tedlium lexicon
        libsp_lex_fn - str: filename to librispeech lexicon
        lex_out_fn - str: filename where lexicon dictionary will be saved
    Returns:
        None
    """
    cmu_dict = lexicon_to_dict(cmu_lex_fn, corpus_name="cmudict")
    ted_dict = lexicon_to_dict(ted_lex_fn, corpus_name="tedlium")
    libsp_dict = lexicon_to_dict(libsp_lex_fn, corpus_name="librispeech")

    master_dict = create_master_lexicon(cmu_dict, ted_dict, libsp_dict, out_path=lex_out_fn)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Preprocess librispeech dataset.")
    parser.add_argument(
        "--data-paths", nargs='+', help="The path to the cmudict pronuciation dictionary."
    )
    parser.add_argument(
        "--action", type=str, help="Describes the function to be called."
    )
    parser.add_argument(
        "--corpus-names", nargs='+', help="Name of the lexicon corpora."
    )
    parser.add_argument(
        "--save-path", type=str, help="The path where the output will be saved."
    )
    args = parser.parse_args()
    
    if args.action == "create-master-lexicon":
        make_master_lexicon(*args.data_paths, args.save_path)
    elif args.action == "combine-lexicons":
        combine_lexicons(*args.data_paths, *args.corpus_names, args.save_path) 
