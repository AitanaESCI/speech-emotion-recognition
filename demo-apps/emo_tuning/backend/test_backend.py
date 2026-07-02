import os
import shutil
from pathlib import Path

# Force testing database to be used so we do not wipe the user's working database
os.environ["TESTING"] = "1"

from main import (
    DB_PATH,
    CoordinateUpdate,
    get_points,
    mark_point_audited,
    reset_point_coordinates,
    startup_event,
    update_point_coordinates,
)


def run_tests():
    print("Running backend tests...")

    # 1. Clean environment
    if DB_PATH.exists():
        print(f"Removing existing test database: {DB_PATH}")
        os.remove(DB_PATH)

    # 2. Trigger startup
    print("Triggering startup_event...")
    startup_event()

    # 3. Retrieve points
    print("Retrieving points...")
    points = get_points()
    assert isinstance(points, list), "get_points should return a list"
    assert len(points) > 0, "get_points should return at least one point"
    print(f"Successfully retrieved {len(points)} points.")

    # Check first point
    p1 = points[0]
    p1_id = p1["id"]
    p1_orig_x = p1["merged_coord_x"]
    p1_orig_y = p1["merged_coord_y"]
    assert p1["audited_at"] is None, "New point should not be audited"
    print(f"Point {p1_id} original coords: ({p1_orig_x}, {p1_orig_y})")

    # 4. Update coordinates
    new_coords = CoordinateUpdate(
        merged_coord_x=0.5,
        merged_coord_y=-0.5,
        should_exclude_override=True,
        coord_major_emo_override="sad",
    )
    print(
        f"Updating point {p1_id} coordinates to (0.5, -0.5), override to sad, exclude to True..."
    )
    res = update_point_coordinates(p1_id, new_coords)
    assert res["status"] == "updated"
    assert res["merged_coord_x"] == 0.5
    assert res["merged_coord_y"] == -0.5
    assert res["should_exclude_override"] is True
    assert res["coord_major_emo_override"] == "sad"
    assert res["audited_at"] is not None
    print("Update successful.")

    # 5. Retrieve points again and verify merge
    print("Retrieving points after update...")
    points_after = get_points()
    p1_after = next(p for p in points_after if p["id"] == p1_id)
    assert p1_after["merged_coord_x"] == 0.5
    assert p1_after["merged_coord_y"] == -0.5
    assert p1_after["should_exclude_override"] is True
    assert p1_after["coord_major_emo_override"] == "sad"
    assert p1_after["audited_at"] is not None
    print(
        f"Verified merged coordinates: ({p1_after['merged_coord_x']}, {p1_after['merged_coord_y']}) at {p1_after['audited_at']}"
    )

    # 6. Mark audited
    print(f"Marking point {p1_id} as audited...")
    res_audit = mark_point_audited(p1_id)
    assert res_audit["status"] == "audited"
    assert res_audit["merged_coord_x"] == 0.5
    print("Audit status marked successfully.")

    # 7. Reset coordinates
    print(f"Resetting point {p1_id} coordinates...")
    res_reset = reset_point_coordinates(p1_id)
    assert res_reset["status"] == "reset"
    assert res_reset["merged_coord_x"] == p1_orig_x
    assert res_reset["merged_coord_y"] == p1_orig_y
    assert res_reset["audited_at"] is None
    print("Reset successful.")

    # 8. Verify reset in GET API
    print("Verifying reset in points list...")
    points_final = get_points()
    p1_final = next(p for p in points_final if p["id"] == p1_id)
    assert p1_final["merged_coord_x"] == p1_orig_x
    assert p1_final["merged_coord_y"] == p1_orig_y
    assert p1_final["audited_at"] is None
    print("Reset verified in points list.")

    print("\nAll backend tests passed successfully!")


if __name__ == "__main__":
    run_tests()
