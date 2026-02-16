"""Weather client — Open-Meteo (free, no API key).

Provides current weather + daily forecast for Heartbeat morning brief.
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

# WMO Weather Code → readable description
_WMO_CODES: dict[int, str] = {
    0: "晴天", 1: "大致晴朗", 2: "局部多雲", 3: "多雲",
    45: "霧", 48: "霧凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "陣雨", 81: "中陣雨", 82: "大陣雨",
    95: "雷雨", 96: "冰雹雷雨", 99: "大冰雹雷雨",
}


class WeatherClient:
    """Fetch weather data from Open-Meteo API."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(
        self,
        latitude: float = 25.0143,
        longitude: float = 121.4673,
        timeout: float = 10.0,
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.timeout = timeout

    async def get_current(self) -> dict[str, Any]:
        """Fetch current weather + today's forecast."""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
            "timezone": "auto",
            "forecast_days": 1,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(self.BASE_URL, params=params)
            resp.raise_for_status()
            return resp.json()

    def format_brief(self, data: dict[str, Any]) -> str:
        """Format weather data into a human-readable brief."""
        try:
            current = data["current"]
            daily = data["daily"]

            temp = current["temperature_2m"]
            humidity = current.get("relative_humidity_2m", 0)
            weather_code = current.get("weather_code", 0)
            weather_desc = _WMO_CODES.get(weather_code, "未知")

            high = daily["temperature_2m_max"][0]
            low = daily["temperature_2m_min"][0]
            rain_chance = daily.get("precipitation_probability_max", [0])[0] or 0

            brief = f"{weather_desc}，現在 {temp}°C（濕度 {humidity}%），今天 {low}~{high}°C"
            if rain_chance > 30:
                brief += f"，降雨機率 {rain_chance}%，記得帶傘"
            return brief
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Weather format error: {e}")
            return "天氣資訊暫時無法取得"

    async def get_brief(self) -> str:
        """One-call convenience: fetch + format."""
        try:
            data = await self.get_current()
            return self.format_brief(data)
        except Exception as e:
            logger.warning(f"Weather fetch failed: {e}")
            return "天氣資訊暫時無法取得"
