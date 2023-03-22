# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:16:14 2022

@author: Katharina Waury (k.waury@vu.nl)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import seaborn as sns

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# define continuous variables
cont = ['Length', 'Molecular weight', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 
        'V', 'W', 'Y', 'Isoelectric point', 'Instability index', 'Solubility', 'Disorder_NSP', 'Helix_NSP', 'Coil_NSP', 
        'Sheet_NSP', 'ExpAA', 'First60ExpAA', 'PredHel']


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
            # add percentage of secondary structure as feature
            df_features[column_name] = df_nsp_protein["q3"].value_counts(normalize=True)[i]
        except KeyError:
            # if secondary structure type not present, add 0
            df_features[column_name] = 0
    
    return df_features


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


def get_uniprot(string):
    try:
        _, uniprot, _ = string.split("|")
    except:
        _, uniprot, _ = string.split("_", maxsplit=2)  
       
    return uniprot


def get_value(string):
    _, value = string.split("=")
    return float(value)


def increase_stringency_brain(feature_df, brain_set):

    df_stringent = feature_df[feature_df["Uniprot"].isin(brain_set)]
    print("Number of brain proteins to be removed:", len(df_features) - len(df_stringent))
    print("Number of CSF proteins left:", len(df_stringent[(df_stringent["CSF"] == 1)]))   
    print("Number of non-CSF proteins left:", len(df_stringent[(df_stringent["CSF"] == -1)])) 
    
    return df_stringent


def increase_stringency_CSF(df_features, csf_df, i):
    
    stringent_csf = csf_df[csf_df["#Studies"]>=i]["Uniprot"]
    remove_csf = set(df_features[df_features["CSF"] == 1]["Uniprot"]) - set(stringent_csf)
    df_stringent = df_features.drop(df_features[(df_features["CSF"] == 1) & (df_features["Uniprot"].isin(remove_csf))].index)
    
    print("Number of CSF proteins to be removed:", len(remove_csf))
    print("Number of CSF proteins left:", len(df_stringent[(df_stringent["CSF"] == 1)]))   
    
    return df_stringent


def keep_first_uniprot(string, delim=","):
    if delim in string:
        uniprot, _ = string.split(delim, maxsplit=1)
    else:
        uniprot = string
    
    return uniprot


def preprocess(X, y, random_state=0):
    
    # preprocessing 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)  
    X_train_scal, X_test_scal, scaler = scale_data(X_train, X_test, scaler=StandardScaler(), scaled=cont)
    X_train_bal, y_train_bal = RandomUnderSampler(random_state=0).fit_resample(X_train_scal, y_train)
    
    return X_train_bal, X_test_scal, y_train_bal, y_test, scaler


def print_p_val(p_val):
    
    if p_val < 0.0001:
        return "< 0.0001"
    else:
        return "%.4f" % p_val


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

    return df


def scale_data(X_train, X_test, scaler=StandardScaler(), scaled=cont):
    """
    Default is Standard Scaler (standardization) on all variables.
    """
    
    if scaled == "all":
        # scale all variables
        X_train_scal = scaler.fit_transform(X_train)
        X_test_scal = scaler.transform(X_test)

    else:
        # scale continuous variables
        X_train_scal = X_train.copy()
        X_train_scal[scaled] = scaler.fit_transform(X_train_scal[scaled])
        X_test_scal = X_test.copy()
        X_test_scal[scaled] = scaler.transform(X_test_scal[scaled])

    return X_train_scal, X_test_scal, scaler