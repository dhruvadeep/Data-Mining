# GiG

from typing import Any

import duckdb
import pandas as pd


def get_fec_dataset(filename: str = "resources/fec_2012_contribution_subset.csv") -> pd.DataFrame:
    # read the csv file into a Pandas data frame

    fec_all = pd.read_csv(filename, low_memory=False)
    # The date is in the format 20-JUN-11
    fec_all["contb_receipt_dt"] = pd.to_datetime(fec_all["contb_receipt_dt"], format="%d-%b-%y")

    # ignore the refunds
    # Get the subset of dataset where contribution amount is positive
    fec_all = fec_all[fec_all.contb_receipt_amt > 0]

    # fec_all contains details about all presidential candidates.
    # fec contains the details about contributions to Barack Obama and Mitt Romney only
    # for the rest of the tasks, unless explicitly specified, work on the fec data frame.
    fec = fec_all[fec_all.cand_nm.isin(["Obama, Barack", "Romney, Mitt"])]

    # Make the original dataset as None so that it will be garbage collected
    fec_all = None

    return fec


def tot_amount_candidate_pandas(fec_df: pd.DataFrame, name: str) -> float:
    return 0.0


def tot_amount_state_pandas(fec_df: pd.DataFrame, state: str) -> float:
    return 0.0


def tot_amount_job_pandas(fec_df: pd.DataFrame, candidate: str, company: str, job: str) -> float:
    return 0.0


def tot_contributions_for_cand_pandas(fec_df: pd.DataFrame, candidate: str) -> pd.Series:
    return pd.Series()


def top_10_state_pandas(fec_df: pd.DataFrame, candidate: str) -> pd.Series:
    return pd.Series()


def discretization_pandas(fec_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()


def load_fec_data_to_duckdb(
    filename: str = "resources/fec_2012_contribution_subset.csv",
) -> duckdb.DuckDBPyConnection:
    return duckdb.DuckDBPyConnection()


def query_fec_data(con: duckdb.DuckDBPyConnection, query: str) -> list[Any]:
    return []


t32b1_query = """
"""

t32b2_query = """
"""


t32c1_query = """
"""
