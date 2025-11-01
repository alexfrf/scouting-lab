# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 01:39:23 2025

@author: aleex
"""

import pandas as pd
from sqlalchemy import create_engine
import json
import streamlit as st
import os

def get_secret(key, subkey):
    """
    Obtiene un secreto desde:
    1️⃣ st.secrets (cuando se ejecuta en Streamlit Cloud)
    2️⃣ variables de entorno (Render, Docker, etc.)
    """
    try:
        return st.secrets[key][subkey]
    except (KeyError, AttributeError):
        env_key = f"{key.upper()}__{subkey.upper()}"
        return os.environ.get(env_key)

def get_conn():
    user = get_secret("db_watford", "user")
    password = get_secret("db_watford", "password")
    host = get_secret("db_watford", "host")
    port = get_secret("db_watford", "port")
    database = get_secret("db_watford", "database")

    engine = create_engine(
        f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    )
    return engine

def clean_df(df):
    for i in df.columns:
        if "name" not in i.lower() and "id" not in i.lower():
            df[i] = df[i].apply(lambda x: x.replace(",",".") if isinstance(x, str) else x)
            df[i] = df[i].fillna(0)
    df = df.apply(pd.to_numeric, errors='ignore')
    for i in df.columns:
        if "top" in i:
            df[i]=df[i].fillna(0)
    return df