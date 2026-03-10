from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "ssc_data"

TABLE_METADATA = [
    {
        "filename": "enrollement.csv",
        "domain": "cohort metadata",
        "description": "Enrollment metadata, site information, and the cohort-level group label.",
        "likely_implementation_level": "metadata",
    },
    {
        "filename": "demographic.csv",
        "domain": "demographics",
        "description": "Age, sex/gender, education, living situation, and social background variables.",
        "likely_implementation_level": "Level 1",
    },
    {
        "filename": "clinical.csv",
        "domain": "clinical history",
        "description": "Diagnosis history, disease milestones, and clinician-entered clinical context.",
        "likely_implementation_level": "Level 3",
    },
    {
        "filename": "epidemiological.csv",
        "domain": "exposures and comorbidity",
        "description": "Environmental exposures, medical history, admissions, and epidemiological factors.",
        "likely_implementation_level": "Level 1",
    },
    {
        "filename": "medication.csv",
        "domain": "medication",
        "description": "Levodopa and related medication use, dosing, and treatment response context.",
        "likely_implementation_level": "Level 3",
    },
    {
        "filename": "moca.csv",
        "domain": "cognition",
        "description": "Montreal Cognitive Assessment items and total cognitive screening score.",
        "likely_implementation_level": "Level 3",
    },
    {
        "filename": "moca_1.csv",
        "domain": "cognition",
        "description": "Alternative MoCA extract with the same baseline key structure.",
        "likely_implementation_level": "Level 3",
    },
    {
        "filename": "moca_2.csv",
        "domain": "cognition",
        "description": "Second MoCA extract for the same assessment family.",
        "likely_implementation_level": "Level 3",
    },
    {
        "filename": "mds-updrs.csv",
        "domain": "motor and non-motor severity",
        "description": "MDS-UPDRS items spanning cognition, hallucinations, mood, ADLs, and motor severity.",
        "likely_implementation_level": "Level 3",
    },
    {
        "filename": "mds-updrs-1.csv",
        "domain": "motor and non-motor severity",
        "description": "Related MDS-UPDRS export with overlapping clinical severity fields.",
        "likely_implementation_level": "Level 3",
    },
    {
        "filename": "updrs_1.csv",
        "domain": "motor severity",
        "description": "Legacy UPDRS motor symptom item responses.",
        "likely_implementation_level": "Level 3",
    },
    {
        "filename": "timed_up_go.csv",
        "domain": "functional mobility",
        "description": "Timed Up and Go performance measurements and mobility timing.",
        "likely_implementation_level": "Level 3",
    },
    {
        "filename": "schwab_&_england.csv",
        "domain": "activities of daily living",
        "description": "Schwab and England functional independence score.",
        "likely_implementation_level": "Level 3",
    },
    {
        "filename": "scopa.csv",
        "domain": "autonomic symptoms",
        "description": "SCOPA-AUT style autonomic symptom questionnaire responses.",
        "likely_implementation_level": "Level 2",
    },
    {
        "filename": "pdq_39.csv",
        "domain": "quality of life",
        "description": "PDQ-39 quality-of-life questionnaire items.",
        "likely_implementation_level": "Level 2",
    },
    {
        "filename": "pdq_8.csv",
        "domain": "quality of life",
        "description": "PDQ-8 short-form quality-of-life questionnaire items.",
        "likely_implementation_level": "Level 2",
    },
    {
        "filename": "fatigue_severity_scale.csv",
        "domain": "fatigue",
        "description": "Fatigue Severity Scale item responses.",
        "likely_implementation_level": "Level 2",
    },
    {
        "filename": "parkinson_severity_scale.csv",
        "domain": "anxiety and severity",
        "description": "Parkinson Anxiety Scale style symptom burden items.",
        "likely_implementation_level": "Level 2",
    },
    {
        "filename": "apathy_scale.csv",
        "domain": "apathy",
        "description": "Apathy scale questionnaire items.",
        "likely_implementation_level": "Level 2",
    },
    {
        "filename": "bai.csv",
        "domain": "anxiety",
        "description": "Beck Anxiety Inventory item responses.",
        "likely_implementation_level": "Level 2",
    },
    {
        "filename": "bdii.csv",
        "domain": "depression",
        "description": "Beck Depression Inventory-II item responses.",
        "likely_implementation_level": "Level 2",
    },
    {
        "filename": "ehi.csv",
        "domain": "handedness",
        "description": "Edinburgh Handedness Inventory item responses.",
        "likely_implementation_level": "Level 1",
    },
    {
        "filename": "mbic.csv",
        "domain": "behavior and motivation",
        "description": "Behavioral inventory items related to apathy and motivation.",
        "likely_implementation_level": "Level 2",
    },
    {
        "filename": "mbic_capri.csv",
        "domain": "behavior and motivation",
        "description": "CaPRI-flavored MBI-C export with overlapping behavior items.",
        "likely_implementation_level": "Level 2",
    },
    {
        "filename": "neuropsychological.csv",
        "domain": "neuropsychology",
        "description": "Neuropsychological battery scores and z-scores across cognitive domains.",
        "likely_implementation_level": "Level 4",
    },
    {
        "filename": "neuropsychological_capri.csv",
        "domain": "neuropsychology",
        "description": "Alternative neuropsychological export with overlapping measures.",
        "likely_implementation_level": "Level 4",
    },
    {
        "filename": "neuropsychological_v02.csv",
        "domain": "neuropsychology",
        "description": "Expanded neuropsychology extract with evaluation-specific metadata.",
        "likely_implementation_level": "Level 4",
    },
]

DEFAULT_FEATURE_TABLES = [
    "demographic.csv",
    "clinical.csv",
    "epidemiological.csv",
    "medication.csv",
    "moca.csv",
    "mds-updrs.csv",
    "apathy_scale.csv",
    "bai.csv",
    "bdii.csv",
    "fatigue_severity_scale.csv",
    "pdq_8.csv",
    "pdq_39.csv",
    "scopa.csv",
    "timed_up_go.csv",
    "parkinson_severity_scale.csv",
    "schwab_&_england.csv",
    "ehi.csv",
]

METADATA_SUBSTRINGS = (
    "event name",
    "questionnaire completed",
    "assessment completed",
    "tests completed",
    "how is",
    "how was",
    "who ",
    "filled out",
    "filled",
    "complete?",
    "correlation",
    "date ",
    "date of",
    "language",
    "version",
    "external id",
    "lara id",
    "linked ",
    "question for the coordinator",
    "question for the participant",
    "administered by",
)

TARGET_GROUP_MAP = {
    "PD (Parkinson's Disease)/(Maladie de Parkinson)": "PD",
    "AP (Atypical Parkinsonism)/(Parkinsonisme Atypique)": "AP",
    "Healthy control/Contrôle": "HC",
}

AP_SUBTYPE_MAP = {
    "...Progressive Supranuclear Palsy (PSP)/paralysie supranucléaire progressive (PSP)": "PSP",
    "...Multiple System Atrophy (MSA)/Atrophie multisystématisée (AMS)": "MSA",
    "...Corticobasal Syndrome (CBS)/Dégénérescence cortico-basale (DCB)": "CBS",
    "...Dementia with Lewy Bodies (DLB)/Démence à corps de Lewy (DCL)": "DLB",
    "...Essential Tremor (ET)/Tremblements essentiel (TE)": "ET",
    "...REM Sleep Behaviour Disorder (RBD)/trouble du comportement en sommeil paradoxal (TCSP)": "RBD",
    "...Other/Autre": "OTHER",
    "...Not Determined/Non déterminé": "UNDETERMINED",
}

LEAKAGE_PATTERNS = (
    "clinical__determined_diagnosis",
    "clinical__1_was_the_patient_diagnosed_with_parkinson_s_disease",
    "clinical__1a_if_no_or_uncertain_is_the_diagnosis",
    "clinical__1b_if_applicable_what_is_the_level_of_certainty_of_the_diagnosis",
    "clinical__1c_how_was_the_diagnosis_confirmed_for_c_opn",
    "clinical__2_what_is_the_date_of_diagnosis",
    "clinical__what_is_the_age_at_diagnosis",
    "clinical__ubc_and_toh_only_what_is_the_age_at_diagnosis",
)


@dataclass
class PreparedDataset:
    X: pd.DataFrame
    y: pd.Series
    master: pd.DataFrame
    target_name: str
    feature_summary: pd.DataFrame


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def normalize_column_name(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized[:80]


def _table_prefix(filename: str) -> str:
    return filename.replace(".csv", "").replace("-", "_").replace("&", "and")


def build_unique_feature_names(columns: Sequence[str], filename: str) -> dict[str, str]:
    prefix = _table_prefix(filename)
    rename_map: dict[str, str] = {}
    seen: dict[str, int] = {}

    for column in columns:
        if column == "Project key":
            continue

        base_name = f"{prefix}__{normalize_column_name(column)}"
        seen[base_name] = seen.get(base_name, 0) + 1

        if seen[base_name] == 1:
            rename_map[column] = base_name
        else:
            rename_map[column] = f"{base_name}__dup{seen[base_name]}"

    return rename_map


def get_available_tables(data_dir: Path = DATA_DIR) -> list[str]:
    return sorted(path.name for path in data_dir.glob("*.csv"))


def repository_table_catalog(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    available = set(get_available_tables(data_dir))
    catalog = pd.DataFrame(TABLE_METADATA)
    catalog["available"] = catalog["filename"].isin(available)
    return catalog.sort_values(["available", "filename"], ascending=[False, True]).reset_index(drop=True)


def audit_normalized_column_collisions(
    data_dir: Path = DATA_DIR,
    tables: Sequence[str] | None = None,
) -> pd.DataFrame:
    collision_records: list[dict[str, object]] = []

    for filename in list(tables or get_available_tables(data_dir)):
        path = data_dir / filename
        if not path.exists():
            continue

        header_df = _read_csv(path, nrows=0)
        grouped: dict[str, list[str]] = {}
        for column in header_df.columns:
            if column in {"Project key", "Event Name"}:
                continue
            normalized = normalize_column_name(column)
            grouped.setdefault(normalized, []).append(column)

        for normalized_name, raw_columns in grouped.items():
            if len(raw_columns) > 1:
                collision_records.append(
                    {
                        "filename": filename,
                        "normalized_name": normalized_name,
                        "raw_columns": " | ".join(raw_columns),
                        "collision_count": len(raw_columns),
                    }
                )

    return pd.DataFrame(collision_records).sort_values(
        ["filename", "normalized_name"]
    ).reset_index(drop=True) if collision_records else pd.DataFrame(
        columns=["filename", "normalized_name", "raw_columns", "collision_count"]
    )


def describe_csv_collection(data_dir: Path = DATA_DIR, headers_only: bool = True) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    catalog = repository_table_catalog(data_dir).set_index("filename")

    for filename in get_available_tables(data_dir):
        path = data_dir / filename
        header_df = _read_csv(path, nrows=0)
        summary = {
            "filename": filename,
            "domain": catalog.loc[filename, "domain"] if filename in catalog.index else "unmapped",
            "description": catalog.loc[filename, "description"] if filename in catalog.index else "",
            "likely_implementation_level": (
                catalog.loc[filename, "likely_implementation_level"] if filename in catalog.index else "unknown"
            ),
            "column_count": len(header_df.columns),
            "preview_columns": ", ".join(header_df.columns[:8]),
        }
        if not headers_only:
            full_df = _read_csv(path)
            summary.update(
                {
                    "row_count": len(full_df),
                    "unique_project_keys": full_df["Project key"].nunique() if "Project key" in full_df.columns else pd.NA,
                    "cell_missing_pct": round(float(full_df.isna().mean().mean() * 100), 2),
                    "column_missing_ge_50_pct": round(float((full_df.isna().mean() >= 0.5).mean() * 100), 2),
                }
            )
        summaries.append(summary)

    return pd.DataFrame(summaries).sort_values("filename").reset_index(drop=True)


def load_target_metadata(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    enrollment = _read_csv(data_dir / "enrollement.csv")
    clinical = _read_csv(data_dir / "clinical.csv")

    group_col = next(col for col in enrollment.columns if col.startswith("Enrolment Group:"))
    site_col = next(col for col in enrollment.columns if col.startswith("Site:"))
    diagnosis_code_col = next(col for col in clinical.columns if col.startswith("Determined diagnosis:"))
    diagnosis_detail_col = next(
        col for col in clinical.columns if col.startswith("1a. If 'No' or 'Uncertain', is the diagnosis")
    )

    target_df = enrollment[["Project key", group_col, site_col]].rename(
        columns={
            "Project key": "project_key",
            group_col: "target_group_raw",
            site_col: "site",
        }
    )
    target_df["target_multiclass"] = target_df["target_group_raw"].map(TARGET_GROUP_MAP)
    target_df["target_binary"] = target_df["target_multiclass"].map({"PD": 0, "AP": 1})

    clinical_subset = clinical[["Project key", diagnosis_code_col, diagnosis_detail_col]].rename(
        columns={
            "Project key": "project_key",
            diagnosis_code_col: "clinical_diagnosis_code",
            diagnosis_detail_col: "clinical_subtype_raw",
        }
    )
    clinical_subset["clinical_subtype"] = clinical_subset["clinical_subtype_raw"].map(AP_SUBTYPE_MAP)

    return target_df.merge(clinical_subset, on="project_key", how="left")


def load_feature_tables(
    data_dir: Path = DATA_DIR,
    tables: Sequence[str] | None = None,
    drop_metadata_columns: bool = True,
) -> pd.DataFrame:
    selected_tables = list(tables or DEFAULT_FEATURE_TABLES)
    merged: pd.DataFrame | None = None

    for filename in selected_tables:
        path = data_dir / filename
        if not path.exists():
            continue

        table = _read_csv(path)
        keep_columns = ["Project key"]
        for column in table.columns:
            if column in {"Project key", "Event Name"}:
                continue
            if drop_metadata_columns and any(token in column.lower() for token in METADATA_SUBSTRINGS):
                continue
            keep_columns.append(column)

        table = table[keep_columns].copy()
        rename_map = build_unique_feature_names(table.columns, filename)
        table = table.rename(columns=rename_map).rename(columns={"Project key": "project_key"})

        merged = table if merged is None else merged.merge(table, on="project_key", how="outer")

    if merged is None:
        return pd.DataFrame(columns=["project_key"])
    return merged


def build_master_dataset(
    data_dir: Path = DATA_DIR,
    tables: Sequence[str] | None = None,
    drop_metadata_columns: bool = True,
) -> pd.DataFrame:
    targets = load_target_metadata(data_dir)
    features = load_feature_tables(data_dir=data_dir, tables=tables, drop_metadata_columns=drop_metadata_columns)
    return targets.merge(features, on="project_key", how="left")


def summarize_feature_missingness(master: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    excluded = {
        "project_key",
        "target_group_raw",
        "target_multiclass",
        "target_binary",
        "site",
        "clinical_diagnosis_code",
        "clinical_subtype_raw",
        "clinical_subtype",
    }
    feature_cols = [column for column in master.columns if column not in excluded]
    missing_records: list[dict[str, object]] = []
    for column in feature_cols:
        column_data = master.loc[:, column]
        if isinstance(column_data, pd.DataFrame):
            column_missing_pct = round(float(column_data.isna().mean().mean() * 100), 2)
        else:
            column_missing_pct = round(float(column_data.isna().mean() * 100), 2)
        missing_records.append({"feature": column, "missing_pct": column_missing_pct})

    missingness = pd.DataFrame(missing_records)
    return missingness.sort_values(["missing_pct", "feature"], ascending=[False, True]).head(top_n).reset_index(drop=True)


def prepare_modeling_dataset(
    target: str = "binary",
    data_dir: Path = DATA_DIR,
    tables: Sequence[str] | None = None,
    missingness_threshold: float = 0.65,
    drop_metadata_columns: bool = True,
) -> PreparedDataset:
    master = build_master_dataset(data_dir=data_dir, tables=tables, drop_metadata_columns=drop_metadata_columns)

    if target == "binary":
        master = master.loc[master["target_binary"].notna()].copy()
        y = master["target_binary"].astype(int)
        target_name = "target_binary"
    elif target == "multiclass":
        master = master.loc[master["target_multiclass"].notna()].copy()
        y = master["target_multiclass"].copy()
        target_name = "target_multiclass"
    else:
        raise ValueError("target must be 'binary' or 'multiclass'")

    excluded_columns = {
        "project_key",
        "target_group_raw",
        "target_multiclass",
        "target_binary",
        "site",
        "clinical_diagnosis_code",
        "clinical_subtype_raw",
        "clinical_subtype",
    }

    X = master.drop(columns=[column for column in excluded_columns if column in master.columns]).copy()
    leakage_columns = [column for column in X.columns if any(pattern in column for pattern in LEAKAGE_PATTERNS)]
    if leakage_columns:
        X = X.drop(columns=leakage_columns)

    missingness = X.isna().mean()
    X = X.loc[:, missingness <= missingness_threshold]
    X = X.loc[:, X.nunique(dropna=True) > 1]

    feature_summary = pd.DataFrame(
        {
            "feature": X.columns,
            "dtype": [str(X[column].dtype) for column in X.columns],
            "missing_pct": [round(float(X[column].isna().mean() * 100), 2) for column in X.columns],
        }
    ).sort_values(["missing_pct", "feature"]).reset_index(drop=True)

    return PreparedDataset(X=X, y=y, master=master, target_name=target_name, feature_summary=feature_summary)


def select_level_1_or_2_tables(catalog: pd.DataFrame | None = None) -> list[str]:
    catalog = repository_table_catalog() if catalog is None else catalog
    level_mask = catalog["likely_implementation_level"].isin(["Level 1", "Level 2"])
    return catalog.loc[level_mask & catalog["available"], "filename"].tolist()


def iter_table_paths(data_dir: Path = DATA_DIR, tables: Iterable[str] | None = None) -> list[Path]:
    selected = tables or get_available_tables(data_dir)
    return [data_dir / table for table in selected if (data_dir / table).exists()]
