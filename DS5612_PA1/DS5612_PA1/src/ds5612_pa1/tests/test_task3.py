# GiG


import duckdb
import pandas as pd
import pytest
from pandas import Interval

from ds5612_pa1 import t3_tasks


# Pytest fixtures are a cool feature. Read them up.
# We are using a fixture with module scope.
# This ensures that the function is only called once per module loading
@pytest.fixture(scope="module", name="fec_df")
def fixture_fec_df() -> pd.DataFrame:
    # We are explicitly passing the fully qualified name here
    # because rye test runs from project root folder
    return t3_tasks.get_fec_dataset("src/ds5612_pa1/resources/fec_2012_contribution_subset.csv")


@pytest.fixture(scope="module", name="duckdb_con")
def fixture_duckdb_con() -> duckdb.DuckDBPyConnection:
    return t3_tasks.load_fec_data_to_duckdb(
        "src/ds5612_pa1/resources/fec_2012_contribution_subset.csv"
    )


@pytest.mark.t31a
def test_basic_filtering_candidate_pandas(fec_df: pd.DataFrame) -> None:
    assert t3_tasks.tot_amount_candidate_pandas(fec_df, "Obama, Barack") == 135877427.24
    assert t3_tasks.tot_amount_candidate_pandas(fec_df, "Romney, Mitt") == 88335907.53


@pytest.mark.t31a
def test_basic_filtering_state_pandas(fec_df: pd.DataFrame) -> None:
    assert t3_tasks.tot_amount_state_pandas(fec_df, "CA") == 35062620.839999996
    assert t3_tasks.tot_amount_state_pandas(fec_df, "NY") == 24836131.14
    assert t3_tasks.tot_amount_state_pandas(fec_df, "TX") == 12792822.129999999


@pytest.mark.t31a
def test_basic_filtering_job_pandas(fec_df: pd.DataFrame) -> None:
    assert t3_tasks.tot_amount_job_pandas(fec_df, "Obama, Barack", "GOOGLE", "ENGINEER") == 87212.4
    assert t3_tasks.tot_amount_job_pandas(fec_df, "Romney, Mitt", "GOOGLE", "ENGINEER") == 2850.0


@pytest.mark.t31b
def test_tot_contr_per_state(fec_df: pd.DataFrame) -> None:
    expected_output = {
        "AA": 74,
        "AB": 4,
        "AE": 395,
        "AK": 2036,
        "AL": 3854,
        "AP": 158,
        "AR": 1747,
        "AS": 31,
        "AZ": 10509,
        "CA": 100182,
        "CO": 12289,
        "CT": 9977,
        "DC": 11491,
        "DE": 1782,
        "FL": 29797,
        "FM": 3,
        "GA": 13856,
        "GU": 69,
        "HI": 3973,
        "IA": 3947,
        "ID": 1624,
        "IL": 33240,
        "IN": 6887,
        "KS": 3087,
        "KY": 3627,
        "LA": 3281,
        "MA": 24864,
        "MD": 22552,
        "ME": 3781,
        "MI": 14907,
        "MN": 9266,
        "MO": 6819,
        "MP": 34,
        "MS": 1499,
        "MT": 2003,
        "NC": 14054,
        "ND": 576,
        "NE": 1750,
        "NH": 3154,
        "NJ": 16605,
        "NM": 5344,
        "NV": 3698,
        "NY": 50383,
        "OH": 11999,
        "OK": 3478,
        "ON": 9,
        "OR": 9418,
        "PA": 19280,
        "PR": 472,
        "QU": 1,
        "RI": 2038,
        "SC": 4228,
        "SD": 713,
        "TN": 6534,
        "TX": 32292,
        "UT": 2790,
        "VA": 21451,
        "VI": 415,
        "VT": 3563,
        "WA": 20783,
        "WI": 8050,
        "WV": 1330,
        "WY": 1055,
        "ZZ": 15,
    }

    output = t3_tasks.tot_contributions_for_cand_pandas(fec_df, "Obama, Barack")
    assert output.to_dict() == expected_output


@pytest.mark.t31b
def test_top_10_state(fec_df: pd.DataFrame) -> None:
    expected_output = {
        "CA": 100182,
        "NY": 50383,
        "IL": 33240,
        "TX": 32292,
        "FL": 29797,
        "MA": 24864,
        "MD": 22552,
        "VA": 21451,
        "WA": 20783,
        "PA": 19280,
    }
    output = t3_tasks.top_10_state_pandas(fec_df, "Obama, Barack")
    assert output.to_dict() == expected_output


@pytest.mark.t31c
def test_discretization(fec_df: pd.DataFrame) -> None:
    expected_output = {
        "Obama, Barack": {
            Interval(0, 1, closed="right"): 318.24,
            Interval(1, 10, closed="right"): 337267.62,
            Interval(10, 100, closed="right"): 20288981.41,
            Interval(100, 1000, closed="right"): 54798531.46,
            Interval(1000, 10000, closed="right"): 51753705.67,
            Interval(10000, 100000, closed="right"): 59100.0,
            Interval(100000, 1000000, closed="right"): 1490683.08,
            Interval(1000000, 10000000, closed="right"): 7148839.76,
        },
        "Romney, Mitt": {
            Interval(0, 1, closed="right"): 77.0,
            Interval(1, 10, closed="right"): 29819.66,
            Interval(10, 100, closed="right"): 1987783.76,
            Interval(100, 1000, closed="right"): 22363381.69,
            Interval(1000, 10000, closed="right"): 63942145.42,
            Interval(10000, 100000, closed="right"): 12700.0,
            Interval(100000, 1000000, closed="right"): 0.0,
            Interval(1000000, 10000000, closed="right"): 0.0,
        },
    }

    output = t3_tasks.discretization_pandas(fec_df)
    assert output.to_dict() == expected_output


@pytest.mark.t32a
def test_duckdb_con_basic(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    num_rows_res = duckdb_con.sql("SELECT COUNT(*) as count FROM fec_table").fetchall()
    assert num_rows_res[0][0] == 1001731


@pytest.mark.t32b
def test_duckdb_queries(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    result = t3_tasks.query_fec_data(duckdb_con, t3_tasks.t32b1_query)
    assert result[0][0] == 135877427.2400002
    assert result[1][0] == 88335907.53000028

    expected_result = [
        (100182, "CA"),
        (50383, "NY"),
        (33240, "IL"),
        (32292, "TX"),
        (29797, "FL"),
        (24864, "MA"),
        (22552, "MD"),
        (21451, "VA"),
        (20783, "WA"),
        (19280, "PA"),
    ]
    result = t3_tasks.query_fec_data(duckdb_con, t3_tasks.t32b2_query)
    assert result == expected_result


@pytest.mark.t32c
def test_duckdb_queries_over_pandas(fec_df: pd.DataFrame) -> None:
    assert fec_df is not None
    result = duckdb.query(t3_tasks.t32c1_query).fetchall()
    expected_result = [
        (100182, "CA"),
        (50383, "NY"),
        (33240, "IL"),
        (32292, "TX"),
        (29797, "FL"),
        (24864, "MA"),
        (22552, "MD"),
        (21451, "VA"),
        (20783, "WA"),
        (19280, "PA"),
    ]

    assert result == expected_result
