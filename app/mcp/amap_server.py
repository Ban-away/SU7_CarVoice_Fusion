"""Amap (高德地图) Maps MCP server.

Exposes 13 tools wrapping the Amap Web API for geocoding, weather, directions,
POI search, and distance measurement.

Requires the environment variable ``AMAP_API_KEY`` to be set.
"""

from typing import Any, Dict, List, Optional

import requests

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "The 'mcp' package is required for the Amap MCP server. "
        "Install it with: pip install mcp fastmcp"
    )

from app.shared.config import get_settings

settings = get_settings()
AMAP_MAPS_API_KEY = settings.amap_api_key

if not AMAP_MAPS_API_KEY:
    raise RuntimeError(
        "AMAP_API_KEY environment variable is not set. "
        "Please set it before starting the Amap MCP server."
    )

mcp = FastMCP("amap-maps")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_status(data: Dict[str, Any], context: str) -> Dict[str, Any]:
    """Raise a ValueError if the Amap API response status is not "1"."""
    if data.get("status") != "1":
        raise ValueError(
            f"{context} failed: {data.get('info') or data.get('infocode')}"
        )
    return data


# ---------------------------------------------------------------------------
# Tool 1: maps_regeocode
# ---------------------------------------------------------------------------

@mcp.tool()
def maps_regeocode(location: str) -> Dict[str, Any]:
    """Convert an Amap longitude,latitude coordinate into an administrative
    address (province / city / district)."""
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/geocode/regeo",
            params={"key": AMAP_MAPS_API_KEY, "location": location},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        _check_status(data, "Regeocoding")

        return {
            "province": data["regeocode"]["addressComponent"]["province"],
            "city": data["regeocode"]["addressComponent"]["city"],
            "district": data["regeocode"]["addressComponent"]["district"],
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except (KeyError, ValueError) as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 2: maps_geo
# ---------------------------------------------------------------------------

@mcp.tool()
def maps_geo(address: str, city: Optional[str] = None) -> Dict[str, Any]:
    """Convert a structured address (or landmark name) into longitude,latitude
    coordinates."""
    try:
        params: Dict[str, Any] = {
            "key": AMAP_MAPS_API_KEY,
            "address": address,
        }
        if city:
            params["city"] = city

        response = requests.get(
            "https://restapi.amap.com/v3/geocode/geo",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        _check_status(data, "Geocoding")

        results: List[Dict[str, Any]] = []
        for geo in data.get("geocodes", []):
            results.append({
                "country": geo.get("country"),
                "province": geo.get("province"),
                "city": geo.get("city"),
                "citycode": geo.get("citycode"),
                "district": geo.get("district"),
                "street": geo.get("street"),
                "number": geo.get("number"),
                "adcode": geo.get("adcode"),
                "location": geo.get("location"),
                "level": geo.get("level"),
            })
        return {"return": results}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except (ValueError, KeyError) as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 3: maps_ip_location
# ---------------------------------------------------------------------------

@mcp.tool()
def maps_ip_location(ip: str) -> Dict[str, Any]:
    """Locate the geographic position of an IP address."""
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/ip",
            params={"key": AMAP_MAPS_API_KEY, "ip": ip},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        _check_status(data, "IP Location")

        return {
            "province": data.get("province"),
            "city": data.get("city"),
            "adcode": data.get("adcode"),
            "rectangle": data.get("rectangle"),
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except (ValueError, KeyError) as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 4: maps_weather
# ---------------------------------------------------------------------------

@mcp.tool()
def maps_weather(city: str, date: str) -> Dict[str, Any]:
    """Query weather forecast for a city by name or adcode."""
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/weather/weatherInfo",
            params={
                "key": AMAP_MAPS_API_KEY,
                "city": city,
                "extensions": "all",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        _check_status(data, "Weather")

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return {"error": "No forecast data available"}

        formatted: Dict[str, Any] = {"城市": forecasts[0]["city"]}
        field_map = {
            "dayweather": "天气",
            "daytemp": "温度",
            "daywind": "风向",
            "daypower": "风力",
        }
        for forecast in forecasts[0]["casts"]:
            if forecast["date"] != date:
                continue
            for key, name in field_map.items():
                if forecast.get(key):
                    formatted[name] = forecast[key]
            break

        return formatted
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except (ValueError, KeyError, IndexError) as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 5: maps_bicycling_by_address
# ---------------------------------------------------------------------------

@mcp.tool()
def maps_bicycling_by_address(
    origin_address: str,
    destination_address: str,
    origin_city: Optional[str] = None,
    destination_city: Optional[str] = None,
) -> Dict[str, Any]:
    """Plan a bicycle route between two addresses.

    Args:
        origin_address: Starting address (e.g. "北京市朝阳区阜通东大街6号").
        destination_address: Destination address.
        origin_city: Optional city for better geocoding of origin.
        destination_city: Optional city for better geocoding of destination.

    Returns:
        Route info including distance, duration, and turn-by-turn directions.
        Supports routes up to 500 km.
    """
    try:
        # Geocode origin
        origin_result = maps_geo(origin_address, origin_city)
        if "error" in origin_result:
            return {"error": f"Failed to geocode origin address: {origin_result['error']}"}
        if not origin_result.get("return"):
            return {"error": "No geocoding results found for origin address"}
        origin_location = origin_result["return"][0].get("location")
        if not origin_location:
            return {"error": "Could not extract coordinates from origin geocoding result"}

        # Geocode destination
        dest_result = maps_geo(destination_address, destination_city)
        if "error" in dest_result:
            return {"error": f"Failed to geocode destination address: {dest_result['error']}"}
        if not dest_result.get("return"):
            return {"error": "No geocoding results found for destination address"}
        dest_location = dest_result["return"][0].get("location")
        if not dest_location:
            return {"error": "Could not extract coordinates from destination geocoding result"}

        # Plan route
        route_result = maps_bicycling(origin_location, dest_location)
        if "error" not in route_result:
            route_result["addresses"] = {
                "origin": {"address": origin_address, "coordinates": origin_location},
                "destination": {"address": destination_address, "coordinates": dest_location},
            }

        return route_result
    except Exception as e:
        return {"error": f"Route planning failed: {str(e)}"}


# ---------------------------------------------------------------------------
# Tool 6: maps_bicycling
# ---------------------------------------------------------------------------

@mcp.tool()
def maps_bicycling(
    origin_coordinates: str, destination_coordinates: str
) -> Dict[str, Any]:
    """Plan a bicycle route between two coordinate pairs.

    Args:
        origin_coordinates: "longitude,latitude" (e.g. "116.434307,39.90909").
        destination_coordinates: "longitude,latitude".

    Returns:
        Route info including distance, duration, and turn-by-turn directions.
        Supports routes up to 500 km.
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v4/direction/bicycling",
            params={
                "key": AMAP_MAPS_API_KEY,
                "origin": origin_coordinates,
                "destination": destination_coordinates,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("errcode") != 0:
            return {
                "error": f"Direction bicycling failed: "
                f"{data.get('info') or data.get('infocode')}"
            }

        paths: List[Dict[str, Any]] = []
        for path in data["data"]["paths"]:
            steps: List[Dict[str, Any]] = []
            for step in path["steps"]:
                steps.append({
                    "instruction": step.get("instruction"),
                    "road": step.get("road"),
                    "distance": step.get("distance"),
                    "orientation": step.get("orientation"),
                    "duration": step.get("duration"),
                })
            paths.append({
                "distance": path.get("distance"),
                "duration": path.get("duration"),
                "steps": steps,
            })

        return {
            "data": {
                "origin": data["data"]["origin"],
                "destination": data["data"]["destination"],
                "paths": paths,
            }
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


# ---------------------------------------------------------------------------
# Tool 7: maps_direction_walking
# ---------------------------------------------------------------------------

@mcp.tool()
def maps_direction_walking(origin: str, destination: str) -> Dict[str, Any]:
    """Plan a walking route (up to 100 km) between two coordinate pairs."""
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/direction/walking",
            params={
                "key": AMAP_MAPS_API_KEY,
                "origin": origin,
                "destination": destination,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        _check_status(data, "Direction Walking")

        paths: List[Dict[str, Any]] = []
        for path in data["route"]["paths"]:
            steps: List[Dict[str, Any]] = []
            for step in path["steps"]:
                steps.append({
                    "instruction": step.get("instruction"),
                    "road": step.get("road"),
                    "distance": step.get("distance"),
                    "orientation": step.get("orientation"),
                    "duration": step.get("duration"),
                })
            paths.append({
                "distance": path.get("distance"),
                "duration": path.get("duration"),
                "steps": steps,
            })

        return {
            "route": {
                "origin": data["route"]["origin"],
                "destination": data["route"]["destination"],
                "paths": paths,
            }
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except (ValueError, KeyError) as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 8: maps_direction_driving
# ---------------------------------------------------------------------------

@mcp.tool()
def maps_direction_driving(origin: str, destination: str) -> Dict[str, Any]:
    """Plan a driving route for a passenger car between two coordinate pairs."""
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/direction/driving",
            params={
                "key": AMAP_MAPS_API_KEY,
                "origin": origin,
                "destination": destination,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        _check_status(data, "Direction Driving")

        paths: List[Dict[str, Any]] = []
        for path in data["route"]["paths"]:
            steps: List[Dict[str, Any]] = []
            for step in path["steps"]:
                steps.append({
                    "instruction": step.get("instruction"),
                    "road": step.get("road"),
                    "distance": step.get("distance"),
                    "orientation": step.get("orientation"),
                    "duration": step.get("duration"),
                })
            paths.append({
                "path": path.get("path"),
                "distance": path.get("distance"),
                "duration": path.get("duration"),
                "steps": steps,
            })

        return {
            "route": {
                "origin": data["route"]["origin"],
                "destination": data["route"]["destination"],
                "paths": paths,
            }
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except (ValueError, KeyError) as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 9: maps_direction_transit_integrated
# ---------------------------------------------------------------------------

@mcp.tool()
def maps_direction_transit_integrated(
    origin: str, destination: str, city: str, cityd: str
) -> Dict[str, Any]:
    """Plan a public-transit route (bus, metro, train) between two coordinates.

    Args:
        origin: Origin "longitude,latitude".
        destination: Destination "longitude,latitude".
        city: Origin city name (required for cross-city trips).
        cityd: Destination city name (required for cross-city trips).
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/direction/transit/integrated",
            params={
                "key": AMAP_MAPS_API_KEY,
                "origin": origin,
                "destination": destination,
                "city": city,
                "cityd": cityd,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        _check_status(data, "Direction Transit Integrated")

        transits: List[Dict[str, Any]] = []
        for transit in data["route"].get("transits", []):
            segments: List[Dict[str, Any]] = []
            for segment in transit.get("segments", []):
                # Walking steps
                walking_steps: List[Dict[str, Any]] = []
                for step in segment.get("walking", {}).get("steps", []):
                    walking_steps.append({
                        "instruction": step.get("instruction"),
                        "road": step.get("road"),
                        "distance": step.get("distance"),
                        "action": step.get("action"),
                        "assistant_action": step.get("assistant_action"),
                    })

                # Bus lines
                buslines: List[Dict[str, Any]] = []
                for busline in segment.get("bus", {}).get("buslines", []):
                    via_stops = [
                        {"name": stop.get("name")}
                        for stop in busline.get("via_stops", [])
                    ]
                    buslines.append({
                        "name": busline.get("name"),
                        "departure_stop": {
                            "name": busline.get("departure_stop", {}).get("name")
                        },
                        "arrival_stop": {
                            "name": busline.get("arrival_stop", {}).get("name")
                        },
                        "distance": busline.get("distance"),
                        "duration": busline.get("duration"),
                        "via_stops": via_stops,
                    })

                segments.append({
                    "walking": {
                        "origin": segment.get("walking", {}).get("origin"),
                        "destination": segment.get("walking", {}).get("destination"),
                        "distance": segment.get("walking", {}).get("distance"),
                        "duration": segment.get("walking", {}).get("duration"),
                        "steps": walking_steps,
                    },
                    "bus": {"buslines": buslines},
                    "entrance": {"name": segment.get("entrance", {}).get("name")},
                    "exit": {"name": segment.get("exit", {}).get("name")},
                    "railway": {
                        "name": segment.get("railway", {}).get("name"),
                        "trip": segment.get("railway", {}).get("trip"),
                    },
                })

            transits.append({
                "duration": transit.get("duration"),
                "walking_distance": transit.get("walking_distance"),
                "segments": segments,
            })

        return {
            "route": {
                "origin": data["route"]["origin"],
                "destination": data["route"]["destination"],
                "distance": data["route"].get("distance"),
                "transits": transits,
            }
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except (ValueError, KeyError) as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 10: maps_distance
# ---------------------------------------------------------------------------

@mcp.tool()
def maps_distance(
    origins: str, destination: str, type: str = "1"
) -> Dict[str, Any]:
    """Measure distance between coordinate pairs.

    Args:
        origins: Origin "longitude,latitude" (supports multiple, pipe-separated).
        destination: Destination "longitude,latitude".
        type: ``"0"`` straight-line, ``"1"`` driving, ``"2"`` walking.
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/distance",
            params={
                "key": AMAP_MAPS_API_KEY,
                "origins": origins,
                "destination": destination,
                "type": type,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        _check_status(data, "Distance")

        results: List[Dict[str, Any]] = []
        for result in data["results"]:
            results.append({
                "origin_id": result.get("origin_id"),
                "dest_id": result.get("dest_id"),
                "distance": result.get("distance"),
                "duration": result.get("duration"),
            })

        return {"results": results}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except (ValueError, KeyError) as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 11: maps_text_search
# ---------------------------------------------------------------------------

@mcp.tool()
def maps_text_search(
    keywords: str,
    city: str = "",
    citylimit: str = "false",
    top_k: int = 3,
) -> Dict[str, Any]:
    """Keyword-based POI search.

    Args:
        keywords: Search keywords.
        city: Optional city to limit the search scope.
        citylimit: ``"true"`` to restrict results to the given city only.
        top_k: Maximum number of POI results to return (default 3).
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/place/text",
            params={
                "key": AMAP_MAPS_API_KEY,
                "keywords": keywords,
                "city": city,
                "citylimit": citylimit,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        _check_status(data, "Text Search")

        pois: List[Dict[str, Any]] = []
        for poi in data.get("pois", []):
            pois.append({
                "id": poi.get("id"),
                "name": poi.get("name"),
                "address": poi.get("address"),
                "typecode": poi.get("typecode"),
            })

        return {"pois": pois[:top_k]}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except (ValueError, KeyError) as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 12: maps_around_search
# ---------------------------------------------------------------------------

@mcp.tool()
def maps_around_search(
    location: str, radius: str = "1000", keywords: str = ""
) -> Dict[str, Any]:
    """Search for POIs within a radius around a coordinate.

    Args:
        location: Center "longitude,latitude".
        radius: Search radius in meters (default 1000).
        keywords: Optional keywords to filter results.
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/place/around",
            params={
                "key": AMAP_MAPS_API_KEY,
                "location": location,
                "radius": radius,
                "keywords": keywords,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        _check_status(data, "Around Search")

        pois: List[Dict[str, Any]] = []
        for poi in data.get("pois", []):
            pois.append({
                "id": poi.get("id"),
                "name": poi.get("name"),
                "address": poi.get("address"),
                "typecode": poi.get("typecode"),
            })

        return {"pois": pois}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except (ValueError, KeyError) as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 13: maps_search_detail
# ---------------------------------------------------------------------------

@mcp.tool()
def maps_search_detail(id: str) -> Dict[str, Any]:
    """Retrieve detailed information for a POI by its Amap ID."""
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/place/detail",
            params={"key": AMAP_MAPS_API_KEY, "id": id},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        _check_status(data, "POI Detail")

        if not data.get("pois"):
            return {"error": "No POI found"}

        poi = data["pois"][0]
        result: Dict[str, Any] = {
            "id": poi.get("id"),
            "name": poi.get("name"),
            "location": poi.get("location"),
            "address": poi.get("address"),
            "business_area": poi.get("business_area"),
            "city": poi.get("cityname"),
            "type": poi.get("type"),
            "alias": poi.get("alias"),
        }

        if poi.get("biz_ext"):
            result.update(poi["biz_ext"])

        return result
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except (ValueError, KeyError, IndexError) as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
