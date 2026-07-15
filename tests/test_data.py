import pandas as pd

from f1lap.data import _collect_ergast_pages, _jolpica_laps_to_frame


class FakeErgastResponse:
    def __init__(self, frames, next_page=None):
        self.content = frames
        self._next_page = next_page

    def get_next_result_page(self):
        if self._next_page is None:
            raise ValueError("No more data after this response.")
        return self._next_page


def test_collect_ergast_pages_collects_all_pages():
    second_page = FakeErgastResponse(
        [pd.DataFrame({"number": [2], "driverId": ["norris"]})]
    )
    first_page = FakeErgastResponse(
        [pd.DataFrame({"number": [1], "driverId": ["norris"]})],
        next_page=second_page,
    )

    result = _collect_ergast_pages(first_page)

    assert result["number"].tolist() == [1, 2]
    assert result["driverId"].tolist() == ["norris", "norris"]


def test_jolpica_laps_are_converted_to_fastf1_style_columns():
    laps = pd.DataFrame(
        {
            "number": [1, 1, 2, 2],
            "driverId": ["norris", "verstappen", "norris", "verstappen"],
            "position": [1, 2, 1, 2],
            "time": ["1:32.100", "1:32.500", "1:31.900", "1:32.200"],
        }
    )
    results = pd.DataFrame(
        {
            "driverId": ["norris", "verstappen"],
            "driverCode": ["NOR", "VER"],
            "constructorName": ["McLaren", "Red Bull"],
        }
    )

    converted = _jolpica_laps_to_frame(
        laps,
        results,
        year=2025,
        event_name="Australian Grand Prix",
    )

    assert len(converted) == 4
    assert set(converted["Driver"]) == {"NOR", "VER"}
    assert set(converted["Team"]) == {"McLaren", "Red Bull"}
    assert converted["LapTime"].notna().all()
    assert converted["DataBackend"].eq("jolpica").all()
    assert converted["TelemetryLoaded"].eq(False).all()
