"""
高德天气 API 封装
文档: https://lbs.amap.com/api/webservice/guide/api/weatherinfo
"""
import httpx
from typing import Literal
from pydantic import BaseModel
from config import config
from graph.state import WeatherInfo

# 高德城市编码映射 (常用城市，完整版需要加载 CSV)
CITY_CODES = {
    "北京": "110000", "上海": "310000", "广州": "440100", "深圳": "440300",
    "杭州": "330100", "南京": "320100", "苏州": "320500", "成都": "510100",
    "重庆": "500000", "西安": "610100", "武汉": "420100", "长沙": "430100",
    "天津": "120000", "青岛": "370200", "厦门": "350200", "三亚": "460200",
}


class AMapWeatherTool:
    """高德天气查询工具"""

    BASE_URL = "https://restapi.amap.com/v3/weather/weatherInfo"

    def __init__(self):
        self.api_key = config.amap_api_key
        self.client = httpx.Client(timeout=10.0)

    def _get_city_code(self, city: str) -> str | None:
        city_clean = city.rstrip("市")
        return CITY_CODES.get(city_clean)

    def _generate_suggestion(self, d: dict) -> str:
        weather = d.get("weather") or d.get("dayweather") or d.get("nightweather") or ""
        # 预报：daytemp/nighttemp；实时：temperature
        temp = d.get("temperature")
        if temp is None:
            try:
                dt = float(d.get("daytemp"))
                nt = float(d.get("nighttemp"))
                temp = (dt + nt) / 2
            except Exception:
                temp = None

        tips = []
        if isinstance(temp, (int, float)):
            if temp <= 5:
                tips.append("偏冷，建议羽绒/厚外套+保暖内衣，注意围巾手套。")
            elif temp <= 12:
                tips.append("偏凉，建议外套+毛衣/卫衣，早晚加一层。")
            elif temp <= 20:
                tips.append("温和，长袖为主，备薄外套防风。")
            else:
                tips.append("较暖，短袖/薄长袖即可，注意防晒。")

        if "雨" in weather:
            tips.append("建议雨具与防滑鞋。")
        if "雪" in weather:
            tips.append("注意防滑保暖。")
        if "霾" in weather or "沙" in weather:
            tips.append("空气可能较差，建议口罩。")

        return " ".join(tips) if tips else "注意根据体感增减衣物。"

    def _mock_weather(self, city: str) -> WeatherInfo:
        return WeatherInfo(
            city=city.rstrip("市"),
            temperature=10.0,
            weather="晴",
            humidity=50,
            wind_power="2",
            suggestion="Mock: 早晚偏凉，建议外套。",
            raw_data={"mock": True},
        )

    def _mock_forecast(self, city: str, days: int = 3) -> list[WeatherInfo]:
        base = self._mock_weather(city)
        out = []
        for i in range(max(1, days)):
            out.append(
                WeatherInfo(
                    city=base.city,
                    temperature=float(base.temperature) + (i - 1) * 2,
                    weather=base.weather,
                    humidity=base.humidity,
                    wind_power=base.wind_power,
                    suggestion=f"Mock Day{i+1}: {base.suggestion}",
                    raw_data={"mock": True, "date": f"Day{i+1}", "daytemp": 12 + i, "nighttemp": 6 + i},
                )
            )
        return out

    def get_weather(self, city: str, extensions: Literal["base", "all"] = "base") -> WeatherInfo:
        """
        获取天气信息
        extensions: "base"=实时天气, "all"=预报天气（若要多日请用 get_forecast）
        """
        if not self.api_key:
            return self._mock_weather(city)

        city_code = self._get_city_code(city)
        if not city_code:
            return WeatherInfo(city=city, weather="未知", suggestion=f"暂不支持查询 {city} 的天气，请检查城市名称")

        resp = self.client.get(self.BASE_URL, params={
            "key": self.api_key,
            "city": city_code,
            "extensions": extensions,
            "output": "JSON"
        })
        data = resp.json()

        if data.get("status") != "1":
            return WeatherInfo(city=city, weather="查询失败", suggestion=data.get("info", ""), raw_data=data)

        if extensions == "base":
            live = data["lives"][0]
            return WeatherInfo(
                city=live.get("city", city),
                temperature=float(live.get("temperature") or 0.0),
                weather=live.get("weather", "未知"),
                humidity=int(live.get("humidity") or 0),
                wind_power=str(live.get("windpower") or ""),
                suggestion=self._generate_suggestion(live),
                raw_data=live,
            )

        # extensions == "all"：返回预报（这里折中返回第 1 天）
        try:
            fc = data["forecasts"][0]
            cast = fc["casts"][0]
            dt = float(cast.get("daytemp"))
            nt = float(cast.get("nighttemp"))
            avg = (dt + nt) / 2
            return WeatherInfo(
                city=fc.get("city", city),
                temperature=avg,
                weather=cast.get("dayweather") or cast.get("nightweather") or "未知",
                humidity=0,
                wind_power=str(cast.get("daypower") or cast.get("nightpower") or ""),
                suggestion=self._generate_suggestion(cast),
                raw_data=cast,
            )
        except Exception as e:
            return WeatherInfo(city=city, weather="解析失败", suggestion=f"预报解析失败: {e}", raw_data=data)

    def get_forecast(self, city: str, days: int = 3) -> list[WeatherInfo]:
        """获取未来 N 天预报（按天返回 WeatherInfo 列表）"""
        days = max(1, int(days or 1))

        if not self.api_key:
            return self._mock_forecast(city, days)

        city_code = self._get_city_code(city)
        if not city_code:
            return [WeatherInfo(city=city, weather="未知", suggestion=f"暂不支持查询 {city} 的天气，请检查城市名称")]

        resp = self.client.get(self.BASE_URL, params={
            "key": self.api_key,
            "city": city_code,
            "extensions": "all",
            "output": "JSON"
        })
        data = resp.json()

        if data.get("status") != "1":
            return [WeatherInfo(city=city, weather="查询失败", suggestion=data.get("info", ""), raw_data=data)]

        forecasts = data.get("forecasts") or []
        if not forecasts:
            return [WeatherInfo(city=city, weather="无数据", suggestion="未获取到预报数据", raw_data=data)]

        fc = forecasts[0]
        casts = fc.get("casts") or []
        city_name = fc.get("city", city)

        out: list[WeatherInfo] = []
        for cast in casts[:days]:
            try:
                dt = float(cast.get("daytemp"))
                nt = float(cast.get("nighttemp"))
                avg = (dt + nt) / 2
            except Exception:
                avg = 0.0

            out.append(
                WeatherInfo(
                    city=city_name,
                    temperature=avg,
                    weather=cast.get("dayweather") or cast.get("nightweather") or "未知",
                    humidity=0,
                    wind_power=str(cast.get("daypower") or cast.get("nightpower") or ""),
                    suggestion=self._generate_suggestion(cast),
                    raw_data=cast,  # ✅ 单天原始数据：date/daytemp/nighttemp/dayweather...
                )
            )

        return out

# 工具单例
weather_tool = AMapWeatherTool()

# ---- 测试 ----
if __name__ == "__main__":
    result = weather_tool.get_weather("杭州")
    print(f"🌤️ {result.city}: {result.weather} {result.temperature}°C")
    print(f"💡 建议: {result.suggestion}")
