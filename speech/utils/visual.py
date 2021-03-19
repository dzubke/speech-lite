# This file contains a variety of functions that help visualize data like plots, tables, print stats.
# Author: Dustion Zubke
# Copyright: Speak Labs 2021

# standard libs
# third-party libs
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from prettytable import PrettyTable



def plot_eval_1(mispro_scores:list, sent_scores:list, save_path:str)->None:
    """Creats a line chart of the mispronuncation and whole-sentence scores from evaluation 1
    """
    


def plot_count(ax, count_dict:dict, label:str):
    """
    """
    ax.plot(range(len(count_dict.values())), sorted(list(count_dict.values()), reverse=True))
    ax.set_title(label)
    ax.set_xlabel(f"unique {label}")
    ax.set_ylabel(f"utterance per {label}")
    ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));
    ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));
    plt.tight_layout()


def print_stats(count_dict:dict)->None:
    """Prints stats of the count_dict like the mean, std, max, min, and number of unique keys.
    
    Used in assess_speak_train of assess.py

    Args:
        count_dict
    """
    values = list(count_dict.values())
    mean = round(np.mean(values), 2)
    std = round(np.std(values), 2)
    max_val = round(max(values), 2)
    min_val = round(min(values), 2)
    print(f"mean: {mean}, std: {std}, max: {max_val}, min: {min_val}, total_unique: {len(count_dict)}")
    print(f"sample of 5 values: {list(count_dict.keys())[0:5]}")


def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and 
    also appropriately turns 4000 into 4K (no zero after the decimal).
    taken from: https://dfrieds.com/data-visualizations/how-format-large-tick-values.html
    """
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val
    
    return str(new_tick_format)


def print_symmetric_table(values_dict:dict, row_name:str, title:str)->None:
    """Prints a table of values in  2-d dict with identical inner and outer keys

    Args:
        values_dict (Dict[str, Dict[str, float]]): 2-d dictionary with identical keys on the two levels
        row_name (str): name of the rows
        title (str): title of the table
    """
    table = PrettyTable(title=title)
    sorted_keys = sorted(values_dict.keys())
    table.add_column(row_name, sorted_keys)
    for data_name in sorted_keys:
        table.add_column(data_name, [values_dict[data_name][key] for key in sorted_keys])
    print(table)


def print_nonsym_table(values_dict:dict, row_name:str, title:str)->None:                
    """Prints a prety table from a 2-d dict that has different inner and outer keys (not-symmetric)
    Args: 
        values_dict (Dict[str, Dict[str, float]]): 2-d dict with different keys on inner and outer levels 
        row_name (str): name of the rows 
        title (str): title of the table 
    """ 
    single_row_name = list(values_dict.keys())[0] 
    sorted_inner_keys = sorted(values_dict[single_row_name].keys()) 
    column_names = [row_name] + sorted_inner_keys 
    table = PrettyTable(title=title, field_names=column_names)                             
                                    
    for row_name in values_dict: 
        table.add_row([row_name] + [values_dict[row_name][key] for key in sorted_inner_keys]) 
    print(table)