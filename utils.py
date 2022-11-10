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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# define continuous variables
cont = ['Length', 'Molecular weight', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 
        'V', 'W', 'Y', 'Isoelectric point', 'Instability index', 'Solubility', 'Disorder_NSP', 'Helix_NSP', 'Turn_NSP', 
        'Sheet_NSP', 'ExpAA', 'First60ExpAA', 'PredHel']

def keep_first_uniprot(string):
    if "," in string:
        uniprots = string.split(",")
        uniprot1 = uniprots[0]
    else:
        uniprot1 = string
    
    return uniprot1


def get_uniprot(string):
    try:
        _, uniprot, _ = string.split("|")
    except:
        _, uniprot, _ = string.split("_", maxsplit=2)  
    return uniprot


def get_value(string):
    _, value = string.split("=")
    return float(value)


def print_p_val(p_val):
    
    if p_val < 0.0001:
        return "< 0.0001"
    else:
        return "%.4f" % p_val


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
            # add percentage of secondary structure as feature
            df_features[column_name] = df_nsp_protein["q3"].value_counts(normalize=True)[i]
        except KeyError:
            # if secondary structure type not present, add 0
            df_features[column_name] = 0
    
    return df_features


def increase_stringency_CSF(df_features, csf_df, i):
    
    stringent_csf = csf_df[csf_df["#Studies"]>=i]["Uniprot"]
    remove_csf = set(df_features[df_features["CSF"] == 1]["Uniprot"]) - set(stringent_csf)
    df_stringent = df_features.drop(df_features[(df_features["CSF"] == 1) & (df_features["Uniprot"].isin(remove_csf))].index)
    
    print("Number of CSF proteins to be removed:", len(remove_csf))
    print("Number of CSF proteins left:", len(df_stringent[(df_stringent["CSF"] == 1)]))   
    
    return df_stringent


def increase_stringency_brain(feature_df, brain_set):

    df_stringent = feature_df[feature_df["Uniprot"].isin(brain_set)]
    print("Number of brain proteins to be removed:", len(df_features) - len(df_stringent))
    print("Number of CSF proteins left:", len(df_stringent[(df_stringent["CSF"] == 1)]))   
    print("Number of non-CSF proteins left:", len(df_stringent[(df_stringent["CSF"] == -1)])) 
    
    return df_stringent


def accuracy_stringent(df, n, model, save_model=True):
    """
    """
    
    # define explanatory and response variables
    X = (df.drop(["Uniprot", "Sequence", "CSF"], axis=1))
    y = (df["CSF"])
    
    bac = []
    auc = []
    coefs = []
    
    for i in range(10):
        
        X_train_bal, X_test_scal, y_train_bal, y_test, scaler = preprocess(X, y, random_state=i)
        
        # train model
        if model == "LogisticClassifier_L1":
            clf = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=10000, dual=False, 
                random_state=0).fit(X_train_bal, y_train_bal)
        elif model == "LogisticClassifier_L2":
            clf = LogisticRegression(penalty="l2", C=0.1, max_iter=1000, dual=False, random_state=0).fit(X_train_bal, y_train_bal)     
            
        y_pred = clf.predict(X_test_scal)
        
        bac.append(balanced_accuracy_score(y_test, y_pred))
        auc.append(roc_auc_score(y_test, clf.decision_function(X_test_scal)))
        
        # extract coefficients
        coef = pd.Series(index=X_train_bal.columns, data=clf.coef_[0], name=i)
        coef.sort_values(ascending=False, key=abs, inplace=True)
        coefs.append(coef)
        
    coef_final_model = pd.Series(index=X_train_bal.columns, data=clf.coef_[0], name=n)
    coef_final_model.sort_values(ascending=False, key=abs, inplace=True)
    
    # merge all coefficient series into one dataframe
    coefs_df = pd.merge(coefs[0], coefs[1], left_index=True, right_index=True)
    for i in range(2, len(coefs)):
        coefs_df = pd.merge(coefs_df, coefs[i], left_index=True, right_index=True)  
    coefs_df["Mean"] = coefs_df.mean(axis=1)
    coefs_df["Standard deviation"] = coefs_df.std(axis=1)
    
    if (save_model == True) & (n in [1,2,3]):
        with open(os.getcwd() + "/Models/" + model + "_" + str(n) + "plus.pkl", "wb") as f:  
                pickle.dump(clf, f)

    return bac, auc, coefs, coef_final_model


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


def preprocess(X, y, random_state=0):
    
    # preprocessing 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)  
    X_train_scal, X_test_scal, scaler = scale_data(X_train, X_test, scaler=StandardScaler(), scaled=cont)
    X_train_bal, y_train_bal = RandomUnderSampler(random_state=0).fit_resample(X_train_scal, y_train)
    
    return X_train_bal, X_test_scal, y_train_bal, y_test, scaler
