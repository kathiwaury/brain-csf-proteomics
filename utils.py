# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:16:14 2022

@author: Katharina Waury (k.waury@vu.nl)
"""

import numpy as np
import pandas as pd

from Bio.SeqUtils.ProtParam import ProteinAnalysis


def keep_first_uniprot(string):
    if "," in string:
        uniprots = string.split(",")
        uniprot1 = uniprots[0]
    else:
        uniprot1 = string
    
    return uniprot1


def get_brain_expression(string):
    # check if expression for multiple tissues is provided
    if ";" in string:
        tissues = string.split(";")
        for t in tissues:
            # keep only information on brain expression
            if "brain" in t:
                brain_string = t 
    else:
        brain_string = string
    
    # extract expression value from string
    _, exp = brain_string.split(" ")
    exp = float(exp)

    return exp


def protein_analysis(df, seq_col):

    PA = ProteinAnalysis(df[seq_col])

    # molecular weight
    df["Molecular weight"] = PA.molecular_weight()

    # amino acid proportions
    amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    aa_dict = PA.get_amino_acids_percent()
    for aa in amino_acids:
        df[aa] = aa_dict[aa]

    # isoelectric point
    df["Isoelectric point"] = PA.isoelectric_point()

    # instability index
    df["Instability index"] = PA.instability_index()

    # hydrophobicity
    df["Polar"] = df[["R", "K", "E", "D", "Q", "N"]].sum()
    df["Neutral"] = df[["G", "A", "S", "T", "P", "H", "Y"]].sum()
    df["Hydrophobic"] = df[["C", "L", "V", "I", "M", "F", "W"]].sum()

    # normalized van der Waals volume
    df["Volume_small"] = df[["G", "A", "S", "T", "P", "D"]].sum()
    df["Volume_medium"] = df[["N", "V", "E", "Q", "I", "L"]].sum()
    df["Volume_large"] = df[["M", "H", "K", "F", "R", "Y", "W"]].sum()

    # polarity
    df["Polarity_low"] = df[["L", "I", "F", "W", "C", "M", "V", "Y"]].sum()
    df["Polarity_medium"] = df[["P", "A", "T", "G", "S"]].sum()
    df["Polarity_large"] = df[["H", "Q", "R", "K", "N", "E", "D"]].sum()

    # polarizability
    df["Polarizability_low"] = df[["G", "A", "S", "D", "T"]].sum()
    df["Polarizability_medium"] = df[["C", "P", "N", "V", "E", "Q", "I", "L"]].sum()
    df["Polarizability_large"] = df[["K", "M", "H", "F", "R", "Y", "W"]].sum()

    # charge
    df["Charge_positive"] = df[["K", "R"]].sum()
    df["Charge_neutral"] = df[["A", "N", "C", "Q", "G", "H", "I", "L", "M", "F", "P", "S", "T", "W", "Y", "V"]].sum()
    df["Charge_negative"] = df[["D", "E"]].sum()

    # solvent accessibility
    df["Buried"] = df[["A", "L", "F", "C", "G", "I", "V",  "W"]].sum()
    df["Exposed"] = df[["P", "K", "Q", "E", "N", "D"]].sum()
    df["Intermediate"] = df[["M", "P", "S", "T", "H", "Y"]].sum()

    # # secondary structure
    # df["Helix"] = df[["E", "A", "L", "M", "Q", "K", "R", "H"]].sum()
    # df["Strand"] = df[["V", "I", "Y", "C", "W", "F", "T"]].sum()
    # df["Coil"] = df[["G", "N", "P", "S", "D"]].sum()

    return df


def derive_global_features(df_features, df_nsp):
    """
    Takes feature dataframe and NSP dataframe.
    Calculates global features from NSP residue-based features for the relevant protein and adds the values to 
    the feature dataframe in new columns.
    Outputs the updated feature dataframe with additional columns.
    """     
    
    # subset NSP dataframe
    uniprot = df_features["Uniprot"]
    df_nsp_protein = df_nsp[df_nsp["id"] == uniprot]
    
    # disorder
    df_features["Disorder_NSP"] = np.mean(df_nsp_protein["disorder"])     
    
    # secondary structure
    sec_str = {"C":"Coil_NSP", "H":"Helix_NSP", "E":"Sheet_NSP"}
    for i in sec_str.keys():
        column_name = sec_str[i]
        try:
            # add percentage of secondar structure as feature
            df_features[column_name] = df_nsp_protein["q3"].value_counts(normalize=True)[i]
        except KeyError:
            # if secondary structure type not present, add 0
            df_features[column_name] = 0
    
    return df_features


def get_uniprot(string):
    try:
        _, uniprot, _ = string.split("|")
    except:
        _, uniprot, _ = string.split("_", maxsplit=2)  
    return uniprot


def get_value(string):
    _, value = string.split("=")
    return float(value)


def increase_stringency_CSF(df_features, csf_df, i):
    
    stringent_csf = csf_df[csf_df["#Studies"]>=i]["Uniprot"]
    remove_csf = set(df_features[df_features["CSF"] == 1]["Uniprot"]) - set(stringent_csf)
    df_stringent = df_features.drop(df_features[(df_features["CSF"] == 1) & (df_features["Uniprot"].isin(remove_csf))].index)
    
    print("Number of CSF proteins to be removed:", len(remove_csf))
    print("Number of CSF proteins left:", len(df_stringent[(df_stringent["CSF"] == 1)]))   
    
    return df_stringent
