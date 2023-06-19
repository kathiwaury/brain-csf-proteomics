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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# define global variables

cont_variables = ["Length", "Molecular weight", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V",
                  "W", "Y", "Isoelectric point", "Instability index", "Disorder_NSP", "Helix_NSP", "Coil_NSP", "Sheet_NSP", "PredHel"]

col_dict = {"Disorder_NSP":"Disorder", 
            "Helix_NSP":"Helix", 
            "Coil_NSP":"Coil",
            "Sheet_NSP":"Sheet",
            "ExpAA":"TM residues", 
            "First60ExpAA":"First 60 TM residues", 
            "PredHel":"TM region",
            "PredHel_binary":"TM region (binary)",
            "Cell_membrane":"Cell membrane", 
            "Endoplasmic_reticulum":"Endoplasmic reticulum", 
            "Mitochondrion":"Mitochondrion",
            "Golgi_apparatus":"Golgi apparatus",                          
            "NetNGlyc":"NetNGlyc",
            "GlycoMine_N":"N-linked Glycosylation (GlycoMine)", 
            "GlycoMine_O":"O-linked Glycosylation (GlycoMine)", 
            "GlycoMine_C":"C-linked Glycosylation (GlycoMine)",
            "PS00022":"EGF1",
            "PS01186":"EGF2",
            "PS00232":"Cadherin-1",
            "PS00237":"G-protein receptor F1",
            "PS00027":"Homeobox",
            "PS00028":"Zinc Finger C2H2",
            "DNA_binding": "DNA binding",
            "RNA_binding": "RNA binding",
            "Ectodomain_shedding":"Ectodomain shedding protein"}


def derive_global_features(df_features, df_nsp):
    """
    Takes feature dataframe and NetSurfP-2.0 results dataframe as input.
    Calculates protein-level features from NetSurfP-2.0 residue-based features for the relevant protein and adds the values to 
    the feature dataframe with additional columns for disorder, coil, helix and sheet content.
    Outputs the updated feature dataframe.
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
    """
    Takes expression information provided by the Human Protein Atlas.
    Filter for information on brain expression leves; extracts and returns expression value as string.
    """         
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


def get_uniprot(string, delim="|"):
    """
    Takes string that contaisn Uniprot ID as input. Splits string to extract and return Uniprot ID as string.
    """ 
    try:
        _, uniprot, _ = string.split(sep=delim, maxsplit=2)
    except ValueError:
        _, uniprot = string.split(sep=delim)  
       
    return uniprot


def get_value(string):
    """
    Extracts results value from TMHMM results file and returns it as a float.
    """ 
    _, value = string.split("=")
    return float(value)


def increase_stringency_CSF(df_features, csf_df, i):
    """
    Takes feature dataframe, CSF study dataframe and integer detemrining the minimum number of studies as input.
    Filters feature dataframe for CSF proteins that are found in minimum number of studies. Prints number of CSF proteins to be removed and 
    to be contained. Returns updated feature dataframe. 
    """     
    stringent_csf = csf_df[csf_df["#Studies"]>=i]["Uniprot"]
    remove_csf = set(df_features[df_features["CSF"] == 1]["Uniprot"]) - set(stringent_csf)
    df_stringent = df_features.drop(df_features[(df_features["CSF"] == 1) & (df_features["Uniprot"].isin(remove_csf))].index)
    
    print("Number of CSF proteins to be removed:", len(remove_csf))
    print("Number of CSF proteins left:", len(df_stringent[(df_stringent["CSF"] == 1)]))   
    
    return df_stringent


def keep_first_uniprot(string, delim=","):
    """
    Takes string with several Uniprot IDs as input. Splits string; retains and returns first Uniprot ID as a string.
    """ 
    if delim in string:
        uniprot, _ = string.split(delim, maxsplit=1)
    else:
        uniprot = string
    
    return uniprot


def preprocess(X, y, random_state=0):
    """
    Takes variables and target as input.
    Peforms standard preprocessing including 80/20 train-test split of data. Scales continuous variables using Robust scaler and     
    undersamples majority class to gain balanced classes in training data.
    Returns variabes and targets of train and test sets and the fitted scaler.
    """     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)  
    X_train_scal, X_test_scal, scaler = scale_data(X_train, X_test, scaler=RobustScaler(), cont=cont_variables)
    X_train_bal, y_train_bal = RandomUnderSampler(random_state=0).fit_resample(X_train_scal, y_train)
    
    return X_train_bal, X_test_scal, y_train_bal, y_test, scaler


def print_p_val(p_val):
    """
    Prints P-values in comprehensible manner.
    """     
    if p_val < 0.0001:
        return "< 0.0001"
    else:
        return "%.4f" % p_val


def protein_analysis(df, seq_col):
    """
    Takes feature dataframe and name of sequence column as input.
    Performs analysis of protein seqeuence using the BioPython SeqUtils module. Adds columns for molecular weights, amino acid proportions, 
    iso electric point and instability index.
    Returns updated dataframe.
    """ 
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


def remove_isoform_label(uniprot):
    """
    Takes Uniprot string as input.
    Removes isoform labels and returns Uniprot ID as string.
    """         
    if "-" in uniprot:
        uniprot, _ = uniprot.split("-")
    
    return uniprot


def scale_data(X_train, X_test, cont, scaler=RobustScaler()):
    """
    Takes variables of train and test sets, scaler type and list of continuous variables as input.
    Performs scaling on all continuous variables and returns scales train and test variables and fitted scaler.
    """
    X_train_scal = X_train.copy()
    X_train_scal[cont] = scaler.fit_transform(X_train_scal[cont])
    X_test_scal = X_test.copy()
    X_test_scal[cont] = scaler.transform(X_test_scal[cont])

    return X_train_scal, X_test_scal, scaler