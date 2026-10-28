"""빵셀러 end-to-end 정제·변환·저장 파이프라인.

노트북에서는 다음처럼 사용한다.
    from pipeline import full_pipeline
    result = full_pipeline("data/bakery_raw.csv", "outputs")
"""
from pathlib import Path

import pandas as pd

ITEM_ALIASES = {"케익": "케이크"}


# ---------------------------------------------------------------------------
# 품질 점검
# ---------------------------------------------------------------------------
def quality_report_full(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼별 타입·결측·고유값·음수·공백 문제를 한 표로 요약한다."""
    rows = []
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        is_str = s.dtype == object
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "missing_pct": round(s.isna().mean() * 100, 2),
            "n_unique": int(s.nunique()),
            "n_negative": int((s < 0).sum()) if is_num else None,
            "n_whitespace": int((s.str.strip() != s).sum()) if is_str else None,
            "min": s.min() if is_num else None,
            "max": s.max() if is_num else None,
            "sample": ", ".join(map(str, s.dropna().unique()[:5])),
        })
    report = pd.DataFrame(rows).set_index("column")
    report.attrs["n_rows"] = len(df)
    report.attrs["n_duplicated"] = int(df.duplicated().sum())
    return report


# ---------------------------------------------------------------------------
# 결정 로그
# ---------------------------------------------------------------------------
def log_decision(log: list, step: str, rule: str, before: int, after: int,
                 affected: int, reason: str) -> None:
    log.append({
        "step": step,
        "rule": rule,
        "rows_before": before,
        "rows_after": after,
        "affected": affected,
        "reason": reason,
    })


# ---------------------------------------------------------------------------
# 정제 5단계 (각 단계가 스스로 log_decision 호출)
# ---------------------------------------------------------------------------
def drop_duplicates(df: pd.DataFrame, log: list) -> pd.DataFrame:
    out = df.drop_duplicates()
    log_decision(log, "1. 중복 제거", "완전 중복 행 drop_duplicates()",
                 len(df), len(out), len(df) - len(out),
                 "같은 order_id·내용이 그대로 반복 = 로그 재전송. 매출 이중 집계 방지")
    return out


def strip_store_id(df: pd.DataFrame, log: list) -> pd.DataFrame:
    affected = int((df["store_id"] != df["store_id"].str.strip()).sum())
    out = df.assign(store_id=df["store_id"].str.strip())
    log_decision(log, "2. store_id 공백 제거", "str.strip()",
                 len(df), len(out), affected,
                 "' S01'과 'S01'이 다른 매장으로 groupby 되는 문제 방지")
    return out


def normalize_item(df: pd.DataFrame, log: list) -> pd.DataFrame:
    affected = int(df["item"].isin(ITEM_ALIASES).sum())
    out = df.assign(item=df["item"].replace(ITEM_ALIASES))
    log_decision(log, "3. 상품명 표기 통일", "'케익' → '케이크'",
                 len(df), len(out), affected,
                 "같은 상품이 두 줄로 쪼개져 상품 KPI가 과소 집계되는 문제 방지")
    return out


def split_refunds(df: pd.DataFrame, log: list, side: dict) -> pd.DataFrame:
    is_refund = df["revenue"] < 0
    side["refunds"] = df[is_refund].copy()
    out = df[~is_refund]
    log_decision(log, "4. 환불 분리", "revenue < 0 → 별도 테이블(refunds)",
                 len(df), len(out), int(is_refund.sum()),
                 "오류가 아니라 환불 거래. 삭제하지 않고 따로 보관해 후속 분석에 사용")
    return out


def drop_missing_revenue(df: pd.DataFrame, log: list) -> pd.DataFrame:
    out = df.dropna(subset=["revenue"])
    log_decision(log, "5. 매출 결측 제거", "revenue NaN 행 dropna()",
                 len(df), len(out), len(df) - len(out),
                 "qty×unit_price로 복원은 가능하나 결제 실패 여부를 알 수 없어 "
                 "매출로 확정하지 않음 (1.5% 수준이라 영향 작음)")
    return out


def clean(raw: pd.DataFrame, log: list, side: dict) -> pd.DataFrame:
    return (
        raw
        .pipe(drop_duplicates, log=log)
        .pipe(strip_store_id, log=log)
        .pipe(normalize_item, log=log)
        .pipe(split_refunds, log=log, side=side)
        .pipe(drop_missing_revenue, log=log)
        .assign(month=lambda d: d["order_date"].dt.to_period("M").astype(str))
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 변환 (KPI)
# ---------------------------------------------------------------------------
def store_month_kpi(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["month", "store_id"], as_index=False)
        .agg(revenue=("revenue", "sum"),
             orders=("order_id", "nunique"),
             qty=("qty", "sum"))
        .assign(avg_order_value=lambda d: (d["revenue"] / d["orders"]).round(0))
        .sort_values(["month", "store_id"])
        .reset_index(drop=True)
    )


def item_kpi(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("item", as_index=False)
        .agg(revenue=("revenue", "sum"),
             qty=("qty", "sum"),
             orders=("order_id", "nunique"))
        .assign(revenue_share_pct=lambda d: (d["revenue"] / d["revenue"].sum() * 100).round(2))
        .sort_values("revenue", ascending=False)
        .assign(rank=lambda d: range(1, len(d) + 1))
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 적재 + 전체 실행
# ---------------------------------------------------------------------------
def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["order_date"], dtype={"store_id": str})


def full_pipeline(raw_path, out_dir) -> dict:
    """원본 CSV → 정제 → KPI → Parquet 저장까지 한 번에 실행한다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(raw_path)
    log, side = [], {}
    bakery_clean = clean(raw, log, side)
    store_month = store_month_kpi(bakery_clean)
    items = item_kpi(bakery_clean)

    bakery_clean.to_parquet(out_dir / "bakery_clean.parquet", index=False)
    side["refunds"].to_parquet(out_dir / "bakery_refunds.parquet", index=False)
    store_month.to_parquet(out_dir / "bakery_store_month.parquet", index=False)
    items.to_parquet(out_dir / "bakery_item_kpi.parquet", index=False)

    return {
        "raw": raw,
        "clean": bakery_clean,
        "refunds": side["refunds"],
        "store_month": store_month,
        "item_kpi": items,
        "decision_log": pd.DataFrame(log),
    }
