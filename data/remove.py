# standard libs
import argparse
import csv
from datetime import date
from functools import partial
import glob
import math
from multiprocessing import Array, Pool
import os
import random
# third-party libs
import tqdm
# project libs


def remove_files(data_dir:str, file_ext:str, percent_removed:float):
    """
    Removes a random percentage of files in a directory with a given file extension.
    Args:
        data_dir: Directory where files will be removed
        file_ext: File extension of the files to be removed. Files without that extension will be ignored. 
        percent_removed: Decimal percent of files with given extension that will bre removed.
    Output:
        None
    """
    
    raise NotImplementedError("need to add removing tsv path to this function.")
    NUM_PROC = 200

    # create the file paths
    pattern = "*." + file_ext
    regex = os.path.join(data_dir, pattern)    
    file_paths = glob.glob(regex)
    
    # randomize the files
    random.shuffle(file_paths)

    # create the subset to remove
    assert 0 < percent_removed < 1.0
    num_to_remove = int(len(file_paths) * percent_removed)
    files_to_remove = file_paths[:num_to_remove]

    # remove the files
    # single process approach
    #for file_path in files_to_remove:
    #    os.remove(file_path)
    
    # mutlit-process approach
    pool = Pool(processes=NUM_PROC)
    pool.imap_unordered(os.remove, files_to_remove, chunksize=100)
    pool.close()
    pool.join()


def only_trim_tsv(data_dir:str, audio_ext:str, tsv_path:str)->None:
    """
    This function reads an existing tsv file and if the audio file in each line
    exists, that line is added to a new tsv. Otherwise, the audio file is not added to 
    the next tsv file.
    Args:       
        tsv_path (str): path to existing tsv file
    Returns:
        None, new tsv_file is written
    """

    NUM_PROC = 20
    chunk_size = 50000  # the number of elements fed into the pool
    
    filename, tsv_ext = os.path.splitext(tsv_path)
    new_tsv_path = filename + "_trim_" + str(date.today()) + tsv_ext

    with open(tsv_path, 'r') as tsv_file:
        tsv_list = list(csv.reader(tsv_file, delimiter='\t'))

    #removing the header
    header, tsv_list = tsv_list[0], tsv_list[1:]
    
    # initialize the new tsv file        
    with open(new_tsv_path, 'w') as new_tsv_file:
        new_tsv_file.write('\t'.join(header))
        new_tsv_file.write('\n')
    
    pool_fn = partial(
        _check_write_row,
        new_tsv_path = new_tsv_path,
        data_dir = data_dir,
        audio_ext = audio_ext
    )
   
    # breaking the audio_path list into chunks to see progress
    iterations = math.ceil(len(tsv_list) / chunk_size)
    for chunk_idx in tqdm.tqdm(range(iterations)):
    
        pool = Pool(processes=NUM_PROC)
        pool.map(
            pool_fn, tsv_list[chunk_idx * chunk_size: (chunk_idx+1) * chunk_size]
        )
        pool.close()
        pool.join()


def _check_write_row(tsv_row:list,
             new_tsv_path:str,
             data_dir:str, 
             audio_ext:str):
    """
    Multiprocessing function writes the tsv_row to the new_tsv_path    
    if audio_path (first element of tsv_row) exists.
    """
    audio_path = os.path.join(data_dir, tsv_row[0] + os.extsep + audio_ext)
    if os.path.exists(audio_path):
        with open(new_tsv_path, 'a') as new_tsv:
            new_tsv.write('\t'.join(tsv_row))
            new_tsv.write('\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Removes a random percentage of files in a directory with a given file extension."
    )
    parser.add_argument(
        "--data-dir", type=str, help="Directory where files will be removed"
    )
    parser.add_argument(
        "--ext", type=str, help="File extension of the files to be removed."
    )
    parser.add_argument(
        "--percent-removed", type=float, 
        help="Decimal percent of files with given extension that will be removed."
    )
    parser.add_argument(
        "--tsv-path", type=str,
        help="Path to the tsv file providing the information on each sample."
    )
    parser.add_argument(
        "--only-trim-tsv", default=False, action="store_true",
        help="If True, the tsv files will be removed from the tsv-file"
    )

    args = parser.parse_args()

    if args.only_trim_tsv:
        print("only removing non-existing audio files from tsv")
        only_trim_tsv(args.data_dir, args.ext, args.tsv_path)


#        """
#        This function reads an existing tsv file and if the audio file in each line
#        exists, that line is added to a new tsv. Otherwise, the audio file is not added to 
#        the next tsv file.
#        Args:       
#            tsv_path (str): path to existing tsv file
#        Returns:
#            None, new tsv_file is written
#        Notes:
#            the trim-only-tsv section is all in the main execution because the the pool_fn check_write_row 
#            uses a shared list tsv_list that needs to be defined prior to the check_write_row function's compilation. 
#            It was not possible to include all this inside its own function because the check_write_row function 
#            needed to be defined outside the local function scope. 
#        """
#        
#        data_dir = args.data_dir
#        audio_ext = args.ext
#        tsv_path = args.tsv_path
#
#        NUM_PROC = 20
#        filename, tsv_ext = os.path.splitext(tsv_path)
#        new_tsv_path = filename + "_trim_" + str(date.today()) + tsv_ext
#
#        #global tsv_list
#        with open(tsv_path, 'r') as tsv_file:
#            tsv_list = list(csv.reader(tsv_file, delimiter='\t'))
#
#        #removing the header
#        header, tsv_list = tsv_list[0], tsv_list[1:]
#        
#        # initialize the new tsv file        
#        with open(new_tsv_path, 'w') as new_tsv_file:
#            new_tsv_file.write('\t'.join(header))
#            new_tsv_file.write('\n')
#            
#        def _check_write_row(tsv_row:list,
#                     new_tsv_path:str,
#                     data_dir:str, 
#                     audio_ext:str):
#                     #tsv_array:list = tsv_list):
#            """
#            Multiprocessing function writes a row in contents_dict to the new_tsv_path    
#            if audio_path exists.
#            """
#            #audio_path = os.path.join(data_dir, tsv_list[audio_idx][0] + os.extsep + audio_ext)
#            audio_path = os.path.join(data_dir, tsv_row[0] + os.extsep + audio_ext)
#            if os.path.exists(audio_path):
#                #record_id = os.path.basename(os.path.splitext(audio_path)[0])
#                with open(new_tsv_path, 'a') as new_tsv:
#                    new_tsv.write('\t'.join(tsv_row))
#                    new_tsv.write('\n')
#
#        pool_fn = partial(
#            _check_write_row,
#            new_tsv_path = new_tsv_path,
#            data_dir = data_dir,
#            audio_ext = audio_ext
#        )
#       
#        # breaking the audio_path list into chunks to see progress
#        chunk_size = 10000 
#        iterations = math.ceil(len(tsv_list) / chunk_size)
#        for chunk_idx in tqdm.tqdm(range(iterations)):
#        
#            pool = Pool(processes=NUM_PROC)
#            pool.map(
#                pool_fn, tsv_list[chunk_idx * chunk_size: (chunk_idx+1) * chunk_size]
#            )
#            pool.close()
#            pool.join()
        
    else:
        print("removing files and trimming tsv")
        remove_random(args.data_dir, args.ext, args.percent_removed, args.tsv_path)
