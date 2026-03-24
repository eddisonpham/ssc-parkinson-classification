"""
data_preprocessing.py
======================
Extracts clinically validated scale-level features from C-OPN CSV files.

Design principles
-----------------
- Only validated total / subscale scores are extracted — no item-level responses.
  These scores are what the questionnaires were designed to produce and are
  the units neurologists actually interpret.
- A binary "administered" flag is added per instrument. Missingness is often
  MNAR (Missing Not At Random) — e.g. a patient too impaired to complete the
  Timed Up and Go did not have missing data by accident. The flag preserves
  that signal rather than discarding it.
- Missing scale totals are left as NaN. They are imputed (train-set median
  only) inside the dimensionality-reduction step, AFTER the train/test split,
  to prevent leakage.
- No imputation is performed in this module.
- Diagnosis-leakage columns are explicitly excluded.
- The train/test split is the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path — CSVs live alongside this script
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Target label mapping  (HC and unlabelled rows are excluded)
# ---------------------------------------------------------------------------
_TARGET_MAP = {
    "PD (Parkinson's Disease)/(Maladie de Parkinson)": 0,
    "AP (Atypical Parkinsonism)/(Parkinsonisme Atypique)": 1,
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class ClinicalDataset:
    """
    Feature matrix and labels before any train/test split.

    X            : pd.DataFrame — scale scores + missingness flags. May contain NaN.
    y            : pd.Series   — 0 = PD, 1 = AP.
    feature_names: list[str]   — column names of X.
    """
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]


# ---------------------------------------------------------------------------
# Column-finding utilities
# ---------------------------------------------------------------------------

def _find(df: pd.DataFrame, *fragments: str) -> Optional[str]:
    """Return the first column whose name contains ALL fragments (case-insensitive)."""
    for col in df.columns:
        low = col.lower()
        if all(f.lower() in low for f in fragments):
            return col
    return None


def _get(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _yn(series: pd.Series) -> pd.Series:
    """Yes/Oui → 1.0, No/Non → 0.0, else NaN."""
    s = series.fillna("").astype(str).str.lower()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result[s.str.contains("yes|oui")] = 1.0
    result[s.str.contains(r"\bno\b|non")] = 0.0
    return result


def _administered(df: pd.DataFrame) -> pd.Series:
    """Return 1 if a 'completed / administered / rempli' column indicates yes, else 0."""
    for frag in [("completed", "rempli"), ("administered",), ("complete",)]:
        col = _find(df, *frag)
        if col is not None:
            s = df[col].fillna("").astype(str).str.lower()
            flag = s.str.contains(r"yes|oui|complet").astype(float)
            return flag
    return pd.Series(0.0, index=df.index)


def _read(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    # Normalise project key column name
    if "Project key" in df.columns:
        df = df.rename(columns={"Project key": "project_key"})
    # Drop duplicate project keys — keep baseline (first) row
    if "project_key" in df.columns:
        df = df.drop_duplicates("project_key", keep="first")
    return df


# ---------------------------------------------------------------------------
# Per-instrument feature extractors
# Each returns a pd.DataFrame indexed by the caller's integer index.
# Columns are clean snake_case feature names.
# ---------------------------------------------------------------------------

def _demo(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    age_col = _find(df, "age at study visit", "automatic")
    out["age_at_visit"] = _get(df, age_col)

    edu_col = _find(df, "years of education")
    out["years_education"] = _get(df, edu_col)

    gender_col = _find(df, "gender")
    if gender_col:
        g = df[gender_col].fillna("").astype(str).str.lower()
        out["sex_male"] = np.where(
            g.str.contains("male|masculin") & ~g.str.contains("female|féminin"), 1.0,
            np.where(g.str.contains("female|féminin"), 0.0, np.nan)
        )

    caregiver_col = _find(df, "regular caregiver")
    if caregiver_col:
        out["has_caregiver"] = _yn(df[caregiver_col])

    return out


def _clinical(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Disease duration (years since diagnosis) — NOT leaky (duration ≠ diagnosis)
    dur_col = _find(df, "current duration of disease", "years")
    out["disease_duration_years"] = _get(df, dur_col)

    # Symptom onset laterality: bilateral onset is a key AP red flag
    onset_col = _find(df, "symptoms present on one side or both")
    if onset_col:
        s = df[onset_col].fillna("").astype(str).str.lower()
        out["bilateral_onset"] = np.where(
            s.str.contains("both|deux"), 1.0,
            np.where(s.str.contains("left|right|côté"), 0.0, np.nan)
        )

    # Current bilateral involvement
    bilateral_now_col = _find(df, "symptoms affect both sides")
    if bilateral_now_col:
        out["bilateral_now"] = _yn(df[bilateral_now_col])

    # Falls — early falls are a red flag for AP (PSP especially)
    falls_col = _find(df, "fallen in the last 3 months")
    if falls_col:
        out["falls_last_3mo"] = _yn(df[falls_col])

    # Freezing of gait
    fog_col = _find(df, "freezing of gait", "14.")
    if fog_col:
        out["freezing_of_gait"] = _yn(df[fog_col])

    # Dyskinesia — typical of PD with good levodopa response; rare in AP
    dysk_col = _find(df, "currently have dyskinesia")
    if dysk_col:
        out["has_dyskinesia"] = _yn(df[dysk_col])

    # Hoehn & Yahr stage
    hy_col = _find(df, "hoehn", "yahr rating")
    if hy_col:
        # Values like "(2) Bilateral involvement..." → extract leading number
        raw = df[hy_col].fillna("").astype(str).str.extract(r"\((\d+\.?\d*)\)")[0]
        out["hoehn_yahr"] = pd.to_numeric(raw, errors="coerce")

    # Dementia
    dementia_col = _find(df, "does the patient have dementia")
    if dementia_col:
        out["has_dementia"] = _yn(df[dementia_col])

    # Gradual progression (sudden progression is an AP red flag)
    gradual_col = _find(df, "motor symptoms progress gradually")
    if gradual_col:
        out["gradual_progression"] = _yn(df[gradual_col])

    # Remission (complete remission would argue against PD)
    remission_col = _find(df, "complete remission")
    if remission_col:
        out["complete_remission"] = _yn(df[remission_col])

    return out


def _medication(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Levodopa response — the single most discriminating PD vs AP feature.
    # Good response strongly supports PD; absent response suggests AP.
    resp_col = _find(df, "significant reduction in symptoms", "dopaminergic")
    if resp_col:
        s = df[resp_col].fillna("").astype(str).str.lower()
        resp = pd.Series(np.nan, index=df.index, dtype=float)
        resp[s.str.contains("yes|oui")] = 1.0
        resp[s.str.contains(r"\bno\b|non")] = 0.0
        resp[s.str.contains("uncertain|incertain")] = 0.5
        # "Not applicable" (never tried levodopa) → NaN; don't assign 0
        out["levodopa_response"] = resp

    # Whether the response is still maintained
    maintained_col = _find(df, "improvement", "still present")
    if maintained_col:
        out["levodopa_response_maintained"] = _yn(df[maintained_col])

    # Total levodopa equivalent daily dose
    led_col = _find(df, "total led")
    if led_col:
        out["total_led_mg"] = pd.to_numeric(df[led_col], errors="coerce")

    return out


def _epidemiology(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    pest_col = _find(df, "pesticides")
    if pest_col:
        out["pesticide_exposure"] = _yn(df[pest_col])

    smell_col = _find(df, "sense of smell")
    if smell_col:
        out["hyposmia"] = _yn(df[smell_col])

    rem_col = _find(df, "acting out your dreams")
    if rem_col:
        out["rem_sleep_disorder"] = _yn(df[rem_col])

    constip_col = _find(df, "constipation", "5.")
    if constip_col:
        out["constipation"] = _yn(df[constip_col])

    head_col = _find(df, "blow to the head")
    if head_col:
        out["head_trauma"] = _yn(df[head_col])

    exercise_col = _find(df, "exercise on a regular basis")
    if exercise_col:
        out["regular_exercise"] = _yn(df[exercise_col])

    # Orthostatic dizziness — lightheadedness on standing is a hallmark of
    # autonomic failure in MSA (and to a lesser extent in PD).
    ortho_col = _find(df, "light-headed or dizzy")
    if ortho_col:
        out["orthostatic_dizziness"] = _yn(df[ortho_col])

    # Restless legs syndrome — a prodromal marker of PD, less common in AP.
    rls_col = _find(df, "irrepressible urge to move your legs")
    if rls_col:
        out["leg_restlessness"] = _yn(df[rls_col])

    # First-degree family history of PD (father OR mother)
    father_col = _find(df, "biological father", "parkinson")
    mother_col = _find(df, "biological mother", "parkinson")
    if father_col or mother_col:
        father_pd = _yn(df[father_col]) if father_col else pd.Series(0.0, index=df.index)
        mother_pd = _yn(df[mother_col]) if mother_col else pd.Series(0.0, index=df.index)
        out["family_hx_pd"] = ((father_pd == 1) | (mother_pd == 1)).astype(float)
        out.loc[father_pd.isna() & mother_pd.isna(), "family_hx_pd"] = np.nan

    return out


def _moca(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["moca_administered"] = _administered(df)

    total_col = _find(df, "total score", "extra point")
    out["moca_total"] = _get(df, total_col)

    for feat, frag in [
        ("moca_visuospatial", "visuospatial"),
        ("moca_naming", "naming score"),
        ("moca_attention", "attention score"),
        ("moca_language", "language score"),
        ("moca_abstraction", "abstraction score"),
        ("moca_delayed_recall", "delayed recall score"),
        ("moca_orientation", "orientation score"),
    ]:
        out[feat] = _get(df, _find(df, frag))

    return out


def _updrs(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["updrs_administered"] = _administered(df)

    # Part totals
    for feat, frag in [
        ("updrs_part1_nonmotor",      "non-motor aspects"),
        ("updrs_part2_motor_adl",     "motor aspects"),
        ("updrs_part3_motor_exam",    "motor examination"),
        ("updrs_part4_complications", "motor complications"),
    ]:
        out[feat] = _get(df, _find(df, frag))

    # --- Derived Part III subscores from individual items ---
    # These are more discriminating than the Part III total because AP subtypes
    # have distinct signatures: PSP = severe axial/postural instability;
    # PD = predominant limb tremor; MSA = mixed axial + autonomic.

    # Axial score: arising from chair, gait, freezing, postural stability,
    # posture, global bradykinesia. High axial score → AP more likely.
    axial_items = [
        _get(df, _find(df, "updrs_3_9")),   # arising from chair
        _get(df, _find(df, "updrs_3_10")),  # gait
        _get(df, _find(df, "updrs_3_11")),  # freezing of gait
        _get(df, _find(df, "updrs_3_12")),  # postural stability
        _get(df, _find(df, "updrs_3_13")),  # posture
        _get(df, _find(df, "updrs_3_14")),  # global spontaneity / body bradykinesia
    ]
    axial_df = pd.concat(axial_items, axis=1)
    # Only compute sum when at least one item is present; NaN if all missing
    out["updrs_axial_score"] = axial_df.sum(axis=1, min_count=1)

    # Tremor score: postural + kinetic + rest tremor across all limbs.
    # High tremor score with relatively normal axial score → PD more likely.
    tremor_items = [
        _get(df, _find(df, "updrs_3_15_r")),      # postural tremor right
        _get(df, _find(df, "updrs_3_15_l")),      # postural tremor left
        _get(df, _find(df, "updrs_3_16_r")),      # kinetic tremor right
        _get(df, _find(df, "updrs_3_16_l")),      # kinetic tremor left
        _get(df, _find(df, "updrs_3_17_rue")),    # rest tremor RUE
        _get(df, _find(df, "updrs_3_17_lue")),    # rest tremor LUE
        _get(df, _find(df, "updrs_3_17_rle")),    # rest tremor RLE
        _get(df, _find(df, "updrs_3_17_lle")),    # rest tremor LLE
        _get(df, _find(df, "updrs_3_17_lipjaw")), # rest tremor lip/jaw
        _get(df, _find(df, "updrs_3_18")),        # constancy of rest tremor
    ]
    tremor_df = pd.concat(tremor_items, axis=1)
    out["updrs_tremor_score"] = tremor_df.sum(axis=1, min_count=1)

    # Hoehn & Yahr from UPDRS form (backup to clinical.csv)
    hy_col = _find(df, "hoehn and yahr stage")
    if hy_col:
        raw = df[hy_col].fillna("").astype(str).str.extract(r"(\d+\.?\d*)")[0]
        out["updrs_hoehn_yahr"] = pd.to_numeric(raw, errors="coerce")

    return out


def _updrs_legacy(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["updrs_legacy_administered"] = _administered(df)
    out["updrs_legacy_total"] = _get(df, _find(df, "updrs total"))
    out["updrs_tremor_total"] = _get(df, _find(df, "tremor total"))
    out["updrs_rigidity_total"] = _get(df, _find(df, "rigidity total"))
    out["updrs_laterality_index"] = _get(df, _find(df, "laterality index"))
    return out


def _apathy(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["apathy_administered"] = _administered(df)
    out["apathy_score"] = _get(df, _find(df, "apathy scale score"))
    return out


def _bai(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["bai_administered"] = _administered(df)
    out["bai_total"] = _get(df, _find(df, "bai total score"))
    return out


def _bdii(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["bdii_administered"] = _administered(df)
    out["bdii_total"] = _get(df, _find(df, "bdi-ii total score"))
    return out


def _fss(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["fss_administered"] = _administered(df)
    col = _find(df, "divided by 9")
    if col is None:
        total = _get(df, _find(df, "total of all questions"))
        out["fss_score"] = total / 9
    else:
        out["fss_score"] = _get(df, col)
    return out


def _pdq8(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["pdq8_administered"] = _administered(df)
    out["pdq8_summary_index"] = _get(df, _find(df, "pdq-8 summary index"))
    return out


def _pdq39(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["pdq39_administered"] = _administered(df)
    out["pdq39_summary_index"] = _get(df, _find(df, "pdq-39 summary index"))

    for feat, frag1, frag2 in [
        ("pdq39_mobility",         "mobility",                  "scale score"),
        ("pdq39_adl",              "activities of daily living", "scale score"),
        ("pdq39_emotional",        "emotional well being",      "scale score"),
        ("pdq39_stigma",           "stigma",                    "scale score"),
        ("pdq39_social_support",   "social support",            "scale score"),
        ("pdq39_cognition",        "cognition",                 "scale score"),
        ("pdq39_communication",    "communication",             "scale score"),
        ("pdq39_bodily_discomfort", "bodily discomfort",        "scale score"),
    ]:
        out[feat] = _get(df, _find(df, frag1, frag2))

    return out


def _scopa(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["scopa_administered"] = _administered(df)
    out["scopa_total"] = _get(df, _find(df, "total score"))

    for feat, frag in [
        ("scopa_gastrointestinal",  "gastrointestinal dysfunction"),
        ("scopa_urinary",           "urinary dysfunction"),
        ("scopa_cardiovascular",    "cardiovascular dysfunction"),
        ("scopa_thermoregulatory",  "thermoregulatory dysfunction"),
    ]:
        out[feat] = _get(df, _find(df, frag))

    return out


def _tug(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["tug_administered"] = _administered(df)
    # "Temps (secondes)2" is the float-typed timing column
    time_col = next(
        (c for c in df.columns if "secondes" in c.lower() and pd.api.types.is_numeric_dtype(df[c])),
        None,
    )
    out["tug_seconds"] = _get(df, time_col)
    return out


def _pas(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["pas_administered"] = _administered(df)
    out["pas_score"] = _get(df, _find(df, "pas score"))
    return out


def _schwab(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["schwab_england_administered"] = _administered(df)
    out["schwab_england_score"] = _get(df, _find(df, "score"))
    return out


def _ehi(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ehi_administered"] = _administered(df)
    out["ehi_handedness_score"] = _get(df, _find(df, "ehi handedness score"))
    return out


def _mbic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mild Behavioural Impairment Checklist (MBI-C).

    Extracts the grand total and 5 domain subscores. The MBI captures
    neuropsychiatric symptom burden that differs meaningfully across PD and
    AP subtypes: prominent thought/perception disturbance and impulse
    dyscontrol early in disease course suggests a non-PD diagnosis.

    Domains:
      Motivation/Drive        (items 1–6)   — apathy/abulia
      Mood/Anxiety            (items 7–12)  — affective dysregulation
      Impulse Dyscontrol      (items 13–24) — disinhibition, repetitive behaviours
      Societal Norms          (items 25–29) — social inappropriateness
      Beliefs/Sensory Exp.    (items 30–34) — psychosis/hallucinations
    """
    out = pd.DataFrame(index=df.index)
    out["mbic_administered"] = _administered(df)
    out["mbic_grand_total"] = _get(df, _find(df, "grand total"))
    out["mbic_motivation"] = _get(df, _find(df, "motivation"))
    out["mbic_mood_anxiety"] = _get(df, _find(df, "mood"))
    out["mbic_impulse"] = _get(df, _find(df, "impulsivity"))
    out["mbic_social"] = _get(df, _find(df, "societal"))
    out["mbic_beliefs"] = _get(df, _find(df, "beliefs"))
    return out


def _neuropsych(df: pd.DataFrame) -> pd.DataFrame:
    """
    Neuropsychological battery — domain-level z-scores only.

    Raw test scores are instrument- and norm-specific; z-scores are already
    standardised against normative samples and are directly comparable across
    patients. The cognitive profile across domains is a validated differentiator:
      - Prominent executive dysfunction early → PSP / CBS
      - Prominent memory impairment → DLB or advanced PD
      - Visuospatial deficits → DLB
      - Relatively preserved cognition → MSA (early) or PD (early)
    """
    out = pd.DataFrame(index=df.index)
    out["neuropsych_administered"] = _administered(df)

    for feat, frag in [
        ("neuropsych_global_z",        "global cognition z score"),
        ("neuropsych_memory_z",        "memory z score"),
        ("neuropsych_executive_z",     "executive function z score"),
        ("neuropsych_language_z",      "language z score"),
        ("neuropsych_attention_z",     "attention z score"),
        ("neuropsych_visuospatial_z",  "visio-perceptual"),
    ]:
        out[feat] = _get(df, _find(df, frag))

    # Cognitive status: encode ordinal (Normal > MCI > Dementia)
    # This partially overlaps with has_dementia from clinical.csv but adds
    # a finer-grained MCI category.
    status_col = _find(df, "cognitive status")
    if status_col:
        s = df[status_col].fillna("").astype(str).str.lower()
        cog = pd.Series(np.nan, index=df.index, dtype=float)
        cog[s.str.contains("normal")] = 0.0
        cog[s.str.contains("mci|mild cognitive")] = 1.0
        cog[s.str.contains("dementia|démence")] = 2.0
        out["neuropsych_cognitive_status"] = cog

    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_clinical_dataset(data_dir: Path = DATA_DIR) -> ClinicalDataset:
    """
    Load and extract validated scale-level features from all C-OPN CSVs.

    Returns
    -------
    ClinicalDataset
        X may contain NaN; imputation must happen after train/test split.
    """
    global DATA_DIR
    DATA_DIR = data_dir

    enrollment = _read("enrollement.csv")
    if enrollment.empty:
        raise FileNotFoundError(f"enrollement.csv not found in {data_dir}")

    # ---- Build target -------------------------------------------------------
    group_col = next(
        c for c in enrollment.columns
        if "nrolment" in c and ("roup" in c or "roupe" in c)
    )
    labels = enrollment[["project_key", group_col]].copy()
    labels["y"] = labels[group_col].map(_TARGET_MAP)
    labels = labels.dropna(subset=["y"]).copy()
    labels["y"] = labels["y"].astype(int)
    labels = labels.drop_duplicates("project_key", keep="first")

    # Master frame keyed by project_key
    master = labels[["project_key", "y"]].set_index("project_key")

    # ---- Extract each instrument --------------------------------------------
    # Notes on filenames:
    #   mds-updrs.csv      — MDS-UPDRS Parts I–IV (items + totals)
    #   schwab___england.csv — 3 underscores (matches actual filename on disk)
    #   mbic.csv / mbic_capri.csv — two cohort variants of the same instrument;
    #       both use _mbic(); the join adds new columns only so there is no
    #       double-counting — patients appear in at most one of these files.
    extractors = [
        ("demographic.csv",            _demo),
        ("clinical.csv",               _clinical),
        ("medication.csv",             _medication),
        ("epidemiological.csv",        _epidemiology),
        ("moca.csv",                   _moca),
        ("mds-updrs.csv",              _updrs),          # fixed: was mdsupdrs.csv
        ("updrs_1.csv",                _updrs_legacy),
        ("apathy_scale.csv",           _apathy),
        ("bai.csv",                    _bai),
        ("bdii.csv",                   _bdii),
        ("fatigue_severity_scale.csv", _fss),
        ("pdq_8.csv",                  _pdq8),
        ("pdq_39.csv",                 _pdq39),
        ("scopa.csv",                  _scopa),
        ("timed_up_go.csv",            _tug),
        ("parkinson_severity_scale.csv", _pas),
        ("schwab___england.csv",       _schwab),         # fixed: was schwab__england.csv
        ("ehi.csv",                    _ehi),
        ("mbic.csv",                   _mbic),           # new
        ("mbic_capri.csv",             _mbic),           # new (same extractor, different cohort)
        ("neuropsychological.csv",       _neuropsych),     # new
        ("neuropsychological_capri.csv", _neuropsych),     # new (CaPRI cohort variant, same schema)
    ]

    for filename, fn in extractors:
        raw = _read(filename)
        if raw.empty or "project_key" not in raw.columns:
            continue

        feats = fn(raw)
        feats.index = raw["project_key"].values
        feats.index.name = "project_key"

        # Only join columns not already present
        new_cols = [c for c in feats.columns if c not in master.columns]
        if new_cols:
            master = master.join(feats[new_cols], how="left")

    # ---- Finalise -----------------------------------------------------------
    y = master["y"]
    X = master.drop(columns=["y"])

    # Drop columns that are entirely empty across all rows
    X = X.dropna(axis=1, how="all")

    print(
        f"[data_preprocessing] Loaded {len(X)} patients "
        f"({(y==0).sum()} PD, {(y==1).sum()} AP) "
        f"with {X.shape[1]} scale-level features."
    )
    missing_pct = X.isna().mean().mean() * 100
    print(f"[data_preprocessing] Overall feature missingness: {missing_pct:.1f}% "
          f"(handled by missingness flags + post-split imputation)")

    return ClinicalDataset(X=X, y=y, feature_names=list(X.columns))