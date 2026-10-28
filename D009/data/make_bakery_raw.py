"""빵셀러(가상) 베이커리 체인 원본 판매 로그 생성 스크립트.

실제 운영 로그에서 자주 보이는 품질 문제를 의도적으로 섞어 둔다.
- 완전 중복 행
- store_id 앞뒤 공백
- item 표기 혼재 ('케익' / '케이크')
- revenue 음수 (환불)
- revenue 결측

실행: python make_bakery_raw.py  ->  같은 폴더에 bakery_raw.csv 생성
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ORDERS = 30_000

STORES = ["S01", "S02", "S03", "S04", "S05"]
STORE_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

ITEMS = {  # 상품명: 단가(원)
    "식빵": 4500,
    "크루아상": 3800,
    "소금빵": 3200,
    "바게트": 4000,
    "베이글": 3500,
    "케이크": 28000,
    "마카롱": 2500,
    "스콘": 3300,
}
ITEM_WEIGHTS = [0.18, 0.15, 0.20, 0.08, 0.12, 0.05, 0.12, 0.10]


def make_raw(seed: int = SEED, n: int = N_ORDERS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2025-04-01", "2025-06-30", freq="D")
    items = rng.choice(list(ITEMS), size=n, p=ITEM_WEIGHTS)
    qty = rng.integers(1, 6, size=n)
    unit_price = np.array([ITEMS[i] for i in items])

    df = pd.DataFrame({
        "order_id": [f"O{i:06d}" for i in range(1, n + 1)],
        "order_date": rng.choice(dates, size=n).astype("datetime64[ns]"),
        "store_id": rng.choice(STORES, size=n, p=STORE_WEIGHTS),
        "item": items,
        "qty": qty,
        "unit_price": unit_price,
    })
    df["revenue"] = (df["qty"] * df["unit_price"]).astype(float)

    # 1) store_id 앞뒤 공백 (약 4%)
    idx = rng.choice(n, size=int(n * 0.04), replace=False)
    df.loc[idx, "store_id"] = [
        f" {s}" if k % 2 else f"{s} " for k, s in enumerate(df.loc[idx, "store_id"])
    ]

    # 2) '케이크' 중 약 35%를 '케익'으로 표기
    cake = df.index[df["item"] == "케이크"]
    df.loc[rng.choice(cake, size=int(len(cake) * 0.35), replace=False), "item"] = "케익"

    # 3) 환불: revenue 음수 (약 0.8%)
    idx = rng.choice(n, size=int(n * 0.008), replace=False)
    df.loc[idx, "revenue"] = -df.loc[idx, "revenue"]

    # 4) revenue 결측 (약 1.5%)
    idx = rng.choice(n, size=int(n * 0.015), replace=False)
    df.loc[idx, "revenue"] = np.nan

    # 5) 완전 중복 행 (약 1%) - 로그 재전송 상황을 흉내
    dup = df.sample(n=int(n * 0.01), random_state=seed)
    df = pd.concat([df, dup], ignore_index=True)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).with_name("bakery_raw.csv")
    raw = make_raw()
    raw.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved {out} ({len(raw):,} rows)")
