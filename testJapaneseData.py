# Get Japanese CPI data from API
import requests
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import os

# Path
ROOT = Path(__file__).resolve().parents[0]

# Load private environment variables
load_dotenv(ROOT / ".env")
key=os.getenv("ESTAT-API")

url = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"

params = {
    "appId": key,
    "statsDataId": "0004052037",
    "lang": "E",
    "cdCat01": "0001",
    "areaCode": "13000"
}

r = requests.get(url, params=params)
r.raise_for_status()

data = r.json()

# data = r.json()["GET_META_INFO"]

# classes = data["METADATA_INF"]["CLASS_INF"]["CLASS_OBJ"]

# for obj in classes:
#     print(obj["@id"], obj["@name"])
    
# time_obj = next(
#     obj for obj in classes
#     if obj["@id"] == "time"
# )

# dfDates = pd.DataFrame(time_obj["CLASS"])

# print(dfDates[["@code", "@name"]].to_string(index=False))


## Trial

# BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json"
# TABLE_ID = "0004052037"

# filters = {}

# # 1. Metadata
# r = requests.get(
#     f"{BASE_URL}/getMetaInfo",
#     params={
#         "appId": key,
#         "statsDataId": TABLE_ID,
#         "lang": "E"
#     },
#     timeout=60
# )
# r.raise_for_status()

# meta = r.json()["GET_META_INFO"]["METADATA_INF"]
# objects = meta["CLASS_INF"]["CLASS_OBJ"]

# dimensions = {
#     obj["@id"]: pd.DataFrame(
#         obj["CLASS"] if isinstance(obj["CLASS"], list)
#         else [obj["CLASS"]]
#     )
#     for obj in objects
# }

# # 2. Download observations (pagination)
# observations = []
# start = 1

# while True:
#     params = {
#         "appId": key,
#         "statsDataId": TABLE_ID,
#         "lang": "E",
#         "metaGetFlg": "N",
#         "cntGetFlg": "Y",
#         "limit": 100000,
#         "startPosition": start,
#         **filters
#     }

#     r = requests.get(
#         f"{BASE_URL}/getStatsData",
#         params=params,
#         timeout=120
#     )
#     r.raise_for_status()

#     result = r.json()["GET_STATS_DATA"]

#     if result["RESULT"]["STATUS"] != 0:
#         raise RuntimeError(result["RESULT"]["ERROR_MSG"])

#     info = result["STATISTICAL_DATA"]
#     values = info.get("DATA_INF", {}).get("VALUE", [])

#     if isinstance(values, dict):
#         values = [values]

#     if not values:
#         break

#     observations.extend(values)

#     next_key = info.get("RESULT_INF", {}).get("NEXT_KEY")

#     if not next_key:
#         break

#     start = int(next_key)

# df = pd.DataFrame(observations)

