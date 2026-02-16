"""Tests for WeatherClient."""

import pytest

from clients.weather_client import WeatherClient, _WMO_CODES


class TestWeatherClient:
    def test_init_defaults(self):
        client = WeatherClient()
        assert client.latitude == 25.0143
        assert client.longitude == 121.4673

    def test_init_custom(self):
        client = WeatherClient(latitude=35.0, longitude=139.0)
        assert client.latitude == 35.0
        assert client.longitude == 139.0

    def test_format_brief_sunny(self):
        client = WeatherClient()
        data = {
            "current": {
                "temperature_2m": 22.5,
                "weather_code": 0,
                "relative_humidity_2m": 65,
                "wind_speed_10m": 5.0,
            },
            "daily": {
                "temperature_2m_max": [28.0],
                "temperature_2m_min": [18.0],
                "precipitation_probability_max": [10],
                "weather_code": [0],
            },
        }
        brief = client.format_brief(data)
        assert "22.5°C" in brief
        assert "18.0~28.0°C" in brief
        assert "晴天" in brief
        assert "傘" not in brief  # rain chance 10% < 30%

    def test_format_brief_rainy(self):
        client = WeatherClient()
        data = {
            "current": {
                "temperature_2m": 15.0,
                "weather_code": 61,
                "relative_humidity_2m": 90,
                "wind_speed_10m": 12.0,
            },
            "daily": {
                "temperature_2m_max": [18.0],
                "temperature_2m_min": [13.0],
                "precipitation_probability_max": [80],
                "weather_code": [61],
            },
        }
        brief = client.format_brief(data)
        assert "小雨" in brief
        assert "80%" in brief
        assert "帶傘" in brief

    def test_format_brief_bad_data(self):
        client = WeatherClient()
        brief = client.format_brief({})
        assert "暫時無法取得" in brief

    def test_wmo_codes_coverage(self):
        assert _WMO_CODES[0] == "晴天"
        assert _WMO_CODES[95] == "雷雨"

    @pytest.mark.asyncio
    async def test_get_brief_error_handling(self):
        """Should return fallback string when API fails."""
        client = WeatherClient(latitude=999, longitude=999)
        brief = await client.get_brief()
        # Should not raise, returns fallback
        assert isinstance(brief, str)
