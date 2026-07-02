import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Emotion Relabelling API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent
DB_NAME = "emo_tuning_test.db" if os.getenv("TESTING") == "1" else "emo_tuning.db"
DB_PATH = BASE_DIR / DB_NAME
PARQUET_PATH = (
    BASE_DIR / "ds_v6_new_merged__df_dataset_audio_chunks__20260625_2045.parquet"
)

# Global reference for the read-only parquet dataframe
DF = None


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS point_audits (
            point_id INTEGER PRIMARY KEY,
            original_x REAL,
            original_y REAL,
            new_x REAL,
            new_y REAL,
            audited_at TEXT
        )
    """
    )
    # Check if coord_major_emo_override column exists, if not add it
    cursor.execute("PRAGMA table_info(point_audits)")
    columns = [row["name"] for row in cursor.fetchall()]
    if "coord_major_emo_override" not in columns:
        cursor.execute(
            "ALTER TABLE point_audits ADD COLUMN coord_major_emo_override TEXT"
        )
    if "should_exclude_override" not in columns:
        cursor.execute(
            "ALTER TABLE point_audits ADD COLUMN should_exclude_override INTEGER DEFAULT 0"
        )
    conn.commit()
    conn.close()


def get_split(speaker_id: str) -> str:
    """Map speaker_id to train, validation, or test splits."""
    if speaker_id in [
        "Ses03__F",
        "Ses03__M",
        "Ses04__F",
        "Ses04__M",
        "Ses05__F",
        "Ses05__M",
    ]:
        return "train"
    elif speaker_id in ["Ses02__F", "Ses02__M"]:
        return "validation"
    elif speaker_id in ["Ses01__F", "Ses01__M"]:
        return "test"
    return "unknown"


@app.on_event("startup")
def startup_event():
    global DF
    # Initialize DB
    init_db()

    # Load Parquet
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet dataset not found at {PARQUET_PATH}")

    print(f"Loading parquet dataset from {PARQUET_PATH}...")
    DF = pd.read_parquet(PARQUET_PATH)

    # Ensure ID column is present and set as index or integer
    if "id" not in DF.columns:
        DF = DF.reset_index().rename(columns={"index": "id"})
    DF["id"] = DF["id"].astype(int)

    # Cache split mappings
    DF["split"] = DF["speaker_id"].apply(get_split)
    print(f"Loaded {len(DF)} records successfully.")


class CoordinateUpdate(BaseModel):
    merged_coord_x: float
    merged_coord_y: float
    coord_major_emo_override: Optional[str] = None
    should_exclude_override: Optional[bool] = False


@app.get("/api/points")
def get_points():
    global DF
    if DF is None:
        raise HTTPException(status_code=500, detail="Dataframe not loaded.")

    # Fetch all audits from SQLite
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT point_id, new_x, new_y, audited_at, coord_major_emo_override, should_exclude_override FROM point_audits"
    )
    audits = {row["point_id"]: dict(row) for row in cursor.fetchall()}
    conn.close()

    cols = [
        "id",
        "merged_coord_x",
        "merged_coord_y",
        "merged_major_emotion",
        "ann_emotion",
        "ann_agreement",
        "ann_n_annotators",
        "merged_negative",
        "merged_sad",
        "merged_positive",
        "merged_neutral",
        "speaker_id",
        "chunk_compressed_audio_url",
        "split",
        "new_merged_coord_x",
        "new_merged_coord_y",
        "new_merged_major_emotion",
        "inworld_major_emotion",
        "ann_agreement_rate",
        "reviewed_major_emotion",
        "audited_new_x",
        "audited_new_y",
        "audited_coord_major_emo_override",
        "audited_should_exclude_override",
        "new_merged_positive",
        "new_merged_negative",
        "new_merged_neutral",
        "new_merged_unclear",
        "inworld_emotion_fearful",
        "inworld_emotion_neutral",
        "inworld_emotion_sad",
        "inworld_emotion_calm",
        "inworld_emotion_angry",
        "inworld_emotion_happy",
        "inworld_emotion_surprised",
        "inworld_emotion_disgusted",
        "inworld_emotion_tender",
        "inworld_voice_style_whispering",
        "inworld_voice_style_normal",
        "inworld_voice_style_monotone",
        "inworld_voice_style_singing",
        "inworld_voice_style_mumbling",
        "inworld_voice_style_crying",
        "inworld_voice_style_shouting",
        "inworld_voice_style_laughing",
        "inworld_voice_style_unclear",
    ]

    # Make a copy of the essential fields
    points_df = DF[cols].copy()

    # Preserve raw parquet coordinates before SQLite override merging
    points_df["orig_coord_x"] = points_df["merged_coord_x"]
    points_df["orig_coord_y"] = points_df["merged_coord_y"]
    points_df["orig_new_merged_coord_x"] = points_df["new_merged_coord_x"]
    points_df["orig_new_merged_coord_y"] = points_df["new_merged_coord_y"]

    # Calculate agreement ratio dynamically
    points_df["ann_agreement_ratio"] = (
        points_df["ann_agreement"] / points_df["ann_n_annotators"].clip(lower=1)
    ).astype(float)

    # Default columns
    points_df["audited_at"] = None
    points_df["coord_major_emo_override"] = None
    points_df["should_exclude_override"] = False

    # Merge the audits into the returning points
    if audits:
        # We can map the arrays using the lookup dict
        for idx, row in points_df.iterrows():
            pid = int(row["id"])
            if pid in audits:
                points_df.at[idx, "merged_coord_x"] = audits[pid]["new_x"]
                points_df.at[idx, "merged_coord_y"] = audits[pid]["new_y"]
                points_df.at[idx, "new_merged_coord_x"] = audits[pid]["new_x"]
                points_df.at[idx, "new_merged_coord_y"] = audits[pid]["new_y"]
                points_df.at[idx, "audited_at"] = audits[pid]["audited_at"]
                points_df.at[idx, "coord_major_emo_override"] = audits[pid][
                    "coord_major_emo_override"
                ]
                points_df.at[idx, "should_exclude_override"] = bool(
                    audits[pid]["should_exclude_override"]
                )

    # Convert NaNs to None for clean JSON serialization
    points_df = points_df.astype(object).where(points_df.notna(), None)

    return points_df.to_dict(orient="records")


@app.put("/api/points/{point_id}")
def update_point_coordinates(point_id: int, coords: CoordinateUpdate):
    global DF
    if DF is None:
        raise HTTPException(status_code=500, detail="Dataframe not loaded.")

    # Check if point exists in original df
    row_matches = DF[DF["id"] == point_id]
    if row_matches.empty:
        raise HTTPException(status_code=404, detail="Point ID not found.")

    orig_row = row_matches.iloc[0]
    orig_x = float(orig_row["merged_coord_x"])
    orig_y = float(orig_row["merged_coord_y"])

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # UPSERT
        cursor.execute(
            """
            INSERT INTO point_audits (point_id, original_x, original_y, new_x, new_y, audited_at, coord_major_emo_override, should_exclude_override)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(point_id) DO UPDATE SET
                new_x = excluded.new_x,
                new_y = excluded.new_y,
                audited_at = excluded.audited_at,
                coord_major_emo_override = excluded.coord_major_emo_override,
                should_exclude_override = excluded.should_exclude_override
        """,
            (
                point_id,
                orig_x,
                orig_y,
                coords.merged_coord_x,
                coords.merged_coord_y,
                now_str,
                coords.coord_major_emo_override,
                1 if coords.should_exclude_override else 0,
            ),
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

    return {
        "id": point_id,
        "merged_coord_x": coords.merged_coord_x,
        "merged_coord_y": coords.merged_coord_y,
        "audited_at": now_str,
        "coord_major_emo_override": coords.coord_major_emo_override,
        "should_exclude_override": coords.should_exclude_override,
        "status": "updated",
    }


@app.post("/api/points/{point_id}/audit")
def mark_point_audited(point_id: int):
    global DF
    if DF is None:
        raise HTTPException(status_code=500, detail="Dataframe not loaded.")

    # Check if point exists
    row_matches = DF[DF["id"] == point_id]
    if row_matches.empty:
        raise HTTPException(status_code=404, detail="Point ID not found.")

    orig_row = row_matches.iloc[0]
    orig_x = float(orig_row["merged_coord_x"])
    orig_y = float(orig_row["merged_coord_y"])

    # Check if there is an existing audit to get current coordinates
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT new_x, new_y, coord_major_emo_override, should_exclude_override FROM point_audits WHERE point_id = ?",
        (point_id,),
    )
    row = cursor.fetchone()

    if row:
        current_x = row["new_x"]
        current_y = row["new_y"]
        current_override = row["coord_major_emo_override"]
        current_exclude = row["should_exclude_override"]
    else:
        current_x = orig_x
        current_y = orig_y
        current_override = None
        current_exclude = 0

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        cursor.execute(
            """
            INSERT INTO point_audits (point_id, original_x, original_y, new_x, new_y, audited_at, coord_major_emo_override, should_exclude_override)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(point_id) DO UPDATE SET
                audited_at = excluded.audited_at,
                coord_major_emo_override = excluded.coord_major_emo_override,
                should_exclude_override = excluded.should_exclude_override
        """,
            (
                point_id,
                orig_x,
                orig_y,
                current_x,
                current_y,
                now_str,
                current_override,
                current_exclude,
            ),
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

    return {
        "id": point_id,
        "merged_coord_x": current_x,
        "merged_coord_y": current_y,
        "audited_at": now_str,
        "coord_major_emo_override": current_override,
        "should_exclude_override": bool(current_exclude),
        "status": "audited",
    }


@app.post("/api/points/{point_id}/reset")
def reset_point_coordinates(point_id: int):
    global DF
    if DF is None:
        raise HTTPException(status_code=500, detail="Dataframe not loaded.")

    # Check if point exists
    row_matches = DF[DF["id"] == point_id]
    if row_matches.empty:
        raise HTTPException(status_code=404, detail="Point ID not found.")

    orig_row = row_matches.iloc[0]
    orig_x = float(orig_row["merged_coord_x"])
    orig_y = float(orig_row["merged_coord_y"])

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM point_audits WHERE point_id = ?", (point_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

    return {
        "id": point_id,
        "merged_coord_x": orig_x,
        "merged_coord_y": orig_y,
        "audited_at": None,
        "status": "reset",
    }
