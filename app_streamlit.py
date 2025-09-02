import io
import os
from datetime import datetime
from typing import List, Optional

import streamlit as st
import pandas as pd

# Optional libs for extraction
import pdfplumber
from PIL import Image

try:
    import pytesseract
    TESS_AVAILABLE = True
except Exception:
    TESS_AVAILABLE = False

st.set_page_config(page_title="Excel ↔ PDF/Image Comparator", layout="wide")

st.title("Excel ↔ PDF/Image Comparator")
st.caption("Upload an **Excel/CSV** and a **PDF or image**. I'll extract a table from the PDF/image and compare it with your spreadsheet.")

with st.expander("Tips", expanded=False):
    st.markdown("""
- Works best with **text-based PDFs** (not scanned). For images, clear, high-contrast tables work best.
- If the PDF is scanned, install **Tesseract OCR** so images can be read.
- The app auto-detects likely **key columns** (e.g., `id`, `*_id`, `key`, or highly-unique columns) to compare on.
""")

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
    return df

def read_structured(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
    else:
        df = pd.read_excel(uploaded, dtype=str)
    return normalize_df(df)

def extract_tables_from_pdf(file) -> List[pd.DataFrame]:
    tables = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            try:
                candidates = page.extract_tables(table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5
                })
            except Exception:
                candidates = page.extract_tables()
            for tbl in candidates or []:
                if not tbl or len(tbl) < 2:
                    continue
                header = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(tbl[0])]
                rows = [[("" if x is None else str(x).strip()) for x in r] for r in tbl[1:]]
                width = max(len(header), *(len(r) for r in rows)) if rows else len(header)
                header = (header + [f"col_{i}" for i in range(len(header), width)])[:width]
                rows = [ (r + [""] * (width - len(r)))[:width] for r in rows ]
                df = pd.DataFrame(rows, columns=header)
                df = normalize_df(df)
                tables.append(df)
    return tables

def extract_table_from_image(file) -> List[pd.DataFrame]:
    if not TESS_AVAILABLE:
        return []
    img = Image.open(file)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
    if "line_num" not in data.columns or data.empty:
        return []
    data = data.dropna(subset=["text"])
    data = data[data["text"].str.strip() != ""]
    if data.empty:
        return []
    group_cols = ["page_num", "block_num", "par_num", "line_num"]
    lines = (
        data.sort_values(["page_num", "block_num", "par_num", "line_num", "word_num"])
            .groupby(group_cols)["text"]
            .apply(lambda toks: " ".join([str(t) for t in toks]))
            .reset_index()
    )
    if lines.empty:
        return []
    df = pd.DataFrame({"ocr_line": lines["text"].astype(str).str.strip()})
    df = df[df["ocr_line"] != ""]
    if df.empty:
        return []
    return [df.reset_index(drop=True)]

def combine_tables(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    tables = [t for t in tables if t is not None and not t.empty]
    if not tables:
        return None
    cleaned = []
    for df in tables:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        keep_cols = [c for c in df.columns if not (df[c].astype(str).str.strip() == "").all()]
        if keep_cols:
            df = df[keep_cols]
        cleaned.append(df)
    picked = max(cleaned, key=lambda d: d.shape[0] * max(1, d.shape[1]))
    return normalize_df(picked)

def pick_key_columns(dfA: pd.DataFrame, dfB: pd.DataFrame):
    common = [c for c in dfA.columns if c in dfB.columns]
    if not common:
        return []
    preferred = [c for c in common if c.lower() in {"id", "key", "sku"} or c.lower().endswith("_id")]
    if preferred:
        return preferred
    candidate = []
    n = max(1, len(dfA))
    for c in common:
        try:
            uniq = dfA[c].nunique(dropna=True) / n
            if uniq > 0.8:
                candidate.append(c)
        except Exception:
            pass
    if candidate:
        return candidate[:3]
    return common[:3]

def compare_dataframes(dfA: pd.DataFrame, dfB: pd.DataFrame):
    dfA = normalize_df(dfA)
    dfB = normalize_df(dfB)
    keys = pick_key_columns(dfA, dfB)

    if not keys:
        common_cols = [c for c in dfA.columns if c in dfB.columns]
        left = dfA[common_cols].assign(_src="A")
        right = dfB[common_cols].assign(_src="B")
        merged = left.merge(right[common_cols], how="outer", indicator=True)
        missing_in_B = merged[merged["_merge"] == "left_only"][common_cols]
        missing_in_A = merged[merged["_merge"] == "right_only"][common_cols]
        value_mismatches = pd.DataFrame(columns=["column", "value_A", "value_B"])
        return missing_in_B, missing_in_A, value_mismatches, keys

    left_only = dfA.merge(dfB[keys].drop_duplicates(), on=keys, how="left", indicator=True)
    missing_in_B = left_only[left_only["_merge"] == "left_only"].drop(columns=["_merge"])

    right_only = dfB.merge(dfA[keys].drop_duplicates(), on=keys, how="left", indicator=True)
    missing_in_A = right_only[right_only["_merge"] == "left_only"].drop(columns=["_merge"])

    a_idx = dfA.set_index(keys)
    b_idx = dfB.set_index(keys)
    shared_index = a_idx.index.intersection(b_idx.index).drop_duplicates()
    a_aligned = a_idx.loc[shared_index]
    b_aligned = b_idx.loc[shared_index]
    common_cols = [c for c in a_aligned.columns if c in b_aligned.columns and c not in keys]
    mismatch_records = []
    for idx in shared_index:
        idx_tuple = (idx,) if not isinstance(idx, tuple) else idx
        rowA = a_aligned.loc[idx]
        rowB = b_aligned.loc[idx]
        for col in common_cols:
            va = "" if col not in rowA else ("" if pd.isna(rowA[col]) else str(rowA[col]))
            vb = "" if col not in rowB else ("" if pd.isna(rowB[col]) else str(rowB[col]))
            if va != vb:
                rec = {"column": col, "value_A": va, "value_B": vb}
                for i, k in enumerate(keys):
                    rec[k] = idx_tuple[i]
                mismatch_records.append(rec)
    value_mismatches = pd.DataFrame(mismatch_records)
    if not value_mismatches.empty:
        value_mismatches = value_mismatches[[*keys, "column", "value_A", "value_B"]]
    return missing_in_B, missing_in_A, value_mismatches, keys

# --- UI ---
col_left, col_right = st.columns(2)
with col_left:
    structured = st.file_uploader("Excel/CSV file", type=["xlsx", "xls", "csv"], key="structured")
with col_right:
    other = st.file_uploader("PDF or Image", type=["pdf", "png", "jpg", "jpeg"], key="other")

advanced = st.expander("Advanced options", expanded=False)
with advanced:
    force_keys = st.text_input("Force join key columns (comma-separated). Leave blank to auto-detect.", value="")
    case_insensitive = st.checkbox("Case-insensitive comparison", value=True)

run = st.button("Compare", type="primary", use_container_width=True)

def to_bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

if run:
    if not structured or not other:
        st.error("Please upload both files.")
        st.stop()

    try:
        df_struct = read_structured(structured)
    except Exception as e:
        st.error(f"Failed to read Excel/CSV: {e}")
        st.stop()

    # Extraction
    tables = []
    try:
        if other.name.lower().endswith(".pdf"):
            tables = extract_tables_from_pdf(other)
        else:
            tables = extract_table_from_image(other)
    except Exception as e:
        st.error(f"Failed to extract from PDF/Image: {e}")
        st.stop()

    if not tables:
        st.warning("No tables/text could be extracted from the PDF/image. Try a text-based PDF or install Tesseract for image OCR.")
        st.stop()

    extracted = combine_tables(tables)
    if extracted is None or extracted.empty:
        st.warning("Could not assemble a usable table from the PDF/image.")
        st.stop()

    # Optional case-insensitive normalization
    if case_insensitive:
        df_struct = df_struct.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        extracted = extracted.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    # Allow user to force keys
    if force_keys.strip():
        keys = [k.strip() for k in force_keys.split(",") if k.strip()]
        # Only keep keys that exist in both
        keys = [k for k in keys if k in df_struct.columns and k in extracted.columns]
        def override_keys(dfA, dfB, keys_override):
            # build dummy compare to reuse mismatch logic
            missB, missA, valmis, _ = compare_dataframes(dfA, dfB)
            return missB, missA, valmis, keys_override
        # Monkey: Patch compare to use provided keys by temporarily replacing picker
        # Simpler: clone logic for keys provided
        def compare_with_keys(dfA, dfB, keys_override):
            dfA = normalize_df(dfA); dfB = normalize_df(dfB)
            if not keys_override:
                return compare_dataframes(dfA, dfB)
            left_only = dfA.merge(dfB[keys_override].drop_duplicates(), on=keys_override, how="left", indicator=True)
            missing_in_B = left_only[left_only["_merge"] == "left_only"].drop(columns=["_merge"])
            right_only = dfB.merge(dfA[keys_override].drop_duplicates(), on=keys_override, how="left", indicator=True)
            missing_in_A = right_only[right_only["_merge"] == "left_only"].drop(columns=["_merge"])
            a_idx = dfA.set_index(keys_override); b_idx = dfB.set_index(keys_override)
            shared_index = a_idx.index.intersection(b_idx.index).drop_duplicates()
            a_aligned = a_idx.loc[shared_index]; b_aligned = b_idx.loc[shared_index]
            common_cols = [c for c in a_aligned.columns if c in b_aligned.columns and c not in keys_override]
            mismatch_records = []
            for idx in shared_index:
                idx_tuple = (idx,) if not isinstance(idx, tuple) else idx
                rowA = a_aligned.loc[idx]; rowB = b_aligned.loc[idx]
                for col in common_cols:
                    va = "" if col not in rowA else ("" if pd.isna(rowA[col]) else str(rowA[col]))
                    vb = "" if col not in rowB else ("" if pd.isna(rowB[col]) else str(rowB[col]))
                    if va != vb:
                        rec = {"column": col, "value_A": va, "value_B": vb}
                        for i, k in enumerate(keys_override):
                            rec[k] = idx_tuple[i]
                        mismatch_records.append(rec)
            value_mismatches = pd.DataFrame(mismatch_records)
            if not value_mismatches.empty:
                value_mismatches = value_mismatches[[*keys_override, "column", "value_A", "value_B"]]
            return missing_in_B, missing_in_A, value_mismatches, keys_override
        missing_in_B, missing_in_A, value_mismatches, keys_used = compare_with_keys(df_struct, extracted, keys)
    else:
        missing_in_B, missing_in_A, value_mismatches, keys_used = compare_dataframes(df_struct, extracted)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in Excel missing from PDF/Image", len(missing_in_B))
    c2.metric("Rows in PDF/Image missing from Excel", len(missing_in_A))
    c3.metric("Value mismatches on shared keys", len(value_mismatches))

    st.subheader("Detected key columns")
    st.write(keys_used or [])

    st.subheader("Extracted table (preview)")
    st.dataframe(extracted.head(30), use_container_width=True)

    if len(value_mismatches):
        st.subheader("Value mismatches (sample)")
        st.dataframe(value_mismatches.head(200), use_container_width=True)
    if len(missing_in_B):
        st.subheader("Rows present in Excel but missing from PDF/Image (sample)")
        st.dataframe(missing_in_B.head(200), use_container_width=True)
    if len(missing_in_A):
        st.subheader("Rows present in PDF/Image but missing from Excel (sample)")
        st.dataframe(missing_in_A.head(200), use_container_width=True)

    # Downloads
    st.markdown("---")
    st.subheader("Download reports")
    if len(value_mismatches):
        st.download_button("Download value mismatches CSV",
                           data=to_bytes_csv(value_mismatches),
                           file_name="value_mismatches.csv",
                           mime="text/csv")
    if len(missing_in_B):
        st.download_button("Download rows missing from PDF/Image CSV",
                           data=to_bytes_csv(missing_in_B),
                           file_name="missing_in_pdf_image.csv",
                           mime="text/csv")
    if len(missing_in_A):
        st.download_button("Download rows missing from Excel CSV",
                           data=to_bytes_csv(missing_in_A),
                           file_name="missing_in_excel.csv",
                           mime="text/csv")
