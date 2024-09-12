# GiG

from typing import Any

import duckdb
import pandas as pd
from pandas import Interval

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
    return float(fec_df[fec_df["cand_nm"] == name]["contb_receipt_amt"].sum())
    # print(temp, "Here is the answer....")


def tot_amount_state_pandas(fec_df: pd.DataFrame, state: str) -> float:
    return float(fec_df[fec_df["contbr_st"] == state]["contb_receipt_amt"].sum())
    # print(temp, "Here is the answer....")
    # return temp
    # return 0.0


def tot_amount_job_pandas(fec_df: pd.DataFrame, candidate: str, company: str, job: str) -> float:
    # Correct way to filter with multiple conditions
    temp = fec_df[
        (fec_df["cand_nm"] == candidate) &
        (fec_df["contbr_employer"].str.contains(company)) &
        (fec_df["contbr_occupation"].str.contains(job))
    ]
    return temp["contb_receipt_amt"].sum()



def tot_contributions_for_cand_pandas(fec_df: pd.DataFrame, candidate: str) -> pd.Series:
    return fec_df[
        (fec_df["cand_nm"] == candidate)
    ].groupby(by="contbr_st")["contbr_nm"].count()


def top_10_state_pandas(fec_df: pd.DataFrame, candidate: str) -> pd.Series:
    return tot_contributions_for_cand_pandas(
        fec_df, candidate
        ).sort_values(
            ascending=False
            ).head(10)


def discretization_pandas(fec_df: pd.DataFrame) -> pd.DataFrame:
    bins = [0] +  [10**i for i in range(8)]
    labels = []
    for i in range(1, len(bins)):
        ptr_1 = bins[i - 1]
        ptr_2 = bins[i]
        temp = Interval(left=ptr_1, right=ptr_2, closed="right")
        labels.append(temp)
    fec_df["amount_bucket"] = pd.cut(
        fec_df["contb_receipt_amt"],
        bins=bins,
        labels=labels,
        right=True
    )
    temp = fec_df.groupby(
            ["cand_nm", "amount_bucket"],
            observed=False
        )
    return temp["contb_receipt_amt"].sum().unstack(0)

def load_fec_data_to_duckdb(
    filename: str = "resources/fec_2012_contribution_subset.csv",
) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    conn.execute(f"""
    CREATE TABLE fec_table AS SELECT * FROM read_csv('{filename}')
    """)
    return conn


def query_fec_data(con: duckdb.DuckDBPyConnection, query: str) -> list[Any]:
    return con.execute(query).fetchall()


t32b1_query = """
SELECT SUM(contb_receipt_amt) AS total_amount
FROM fec_table
WHERE contb_receipt_amt > 0
AND (cand_nm = 'Obama, Barack'
    OR
    cand_nm = 'Romney, Mitt'
    )
GROUP BY cand_nm
ORDER BY cand_nm
"""

t32b2_query = """
SELECT contbr_st, COUNT(DISTINCT contbr_nm) AS num_contributors
FROM fec_table
WHERE cand_nm = 'Obama, Barack' AND contb_receipt_amt > 0
GROUP BY contbr_st
ORDER BY num_contributors DESC
LIMIT 10
"""


t32c1_query = """
"""
