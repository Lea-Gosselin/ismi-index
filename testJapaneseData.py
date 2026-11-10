# Get Japanese CPI data from API
import requests
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
from io import BytesIO
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Path
ROOT = Path(__file__).resolve().parents[0]

# Load private environment variables
load_dotenv(ROOT / ".env")
key=os.getenv("ESTAT-API")

url = ("https://www.e-stat.go.jp/en/stat-search/file-download?statInfId=000040513603&fileKind=4")

r = requests.get(url, timeout=60)
r.raise_for_status()

df = pd.read_excel(BytesIO(r.content))
df = df.iloc[9:,7:]
cols = ["Date","All items", "All items, less fresh food", "All items, less fresh food and energy", "Goods", "Services", "Year and month"]
df.columns = cols
year = pd.to_numeric(
    df["Year and month"].astype("string").str[-4:],
    errors="coerce"
).ffill().astype("Int64").astype("string")
month = df["Year and month"].astype("string").str[:3]

# Construct dates
df["Date"] = pd.to_datetime(
    month + " " + year,
    format="%b %Y",
    errors="coerce"
)

headline = df.drop(columns=["Year and month"]).set_index("Date").sort_values("Date",ascending=True)
# Convert to numeric (invalid values become NaN)
headline["All items"] = pd.to_numeric(headline["All items"], errors="coerce")

# Keep only rows with numeric values
headline = headline.dropna(subset=["All items"])

mom = np.log(headline["All items"]).diff(1).dropna()
yoy = np.log(headline["All items"]).diff(12).dropna()

# Date limits
startDate = np.min(yoy.index)
endDate = np.max(yoy.index)

### Plot monthly and yearly inflation rate
fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()

# MoM inflation (left axis)
ax1.plot(
    mom.index, mom*100,
    color="steelblue",
    linewidth=1,
    label="MoM inflation"
)
ax1.set_ylabel("MoM inflation (%)", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")

# YoY inflation (right axis)
ax2.plot(
    yoy.index, yoy*100,
    color="darkred",
    linewidth=1.5,
    label="YoY inflation"
)
ax2.set_ylabel("YoY inflation (%)", color="darkred")
ax2.tick_params(axis="y", labelcolor="darkred")

ax1.set_xlim(pd.Timestamp(startDate), pd.Timestamp(endDate))

# Adjust Y-axis limits to visible observations
for ax, series in [(ax1, mom), (ax2, yoy)]:
    visible = series.loc[startDate:endDate].dropna() * 100

    ymin, ymax = visible.min(), visible.max()
    margin = 0.05 * (ymax - ymin)

    ax.set_ylim(ymin - margin, ymax + margin)

# Grid
ax1.grid(
    True,
    which="major",
    axis="both",
    linestyle="--",
    linewidth=0.6,
    alpha=0.3
)

ax1.set_axisbelow(True)

plt.tight_layout()
plt.show()

### AR(1) sufficient ? 
p = 1

# AR(1) à coefficient constant
Y = mom.values[p:]
X = sm.add_constant(mom.values[:-p])
model = sm.OLS(Y, X).fit()
print(model.summary())
resid = model.resid

# ACF and PACF
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

plot_acf(Y, lags=36, ax=axes[0, 0])
axes[0, 0].set_title("ACF")

plot_pacf(Y, lags=36, ax=axes[0, 1], method="ywm")
axes[0, 1].set_title("PACF") 

plot_acf(resid, lags=36, ax=axes[1, 0])
axes[1, 0].set_title("Residuals (ACF)")

plot_pacf(resid, lags=36, ax=axes[1, 1], method="ywm")
axes[1, 1].set_title("Residuals (PACF)")

for ax in axes.flat:
    ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()


# url = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"

# params = {
#     "appId": key,
#     "statsDataId": "000040513603",
#     "metaGetFlg":"Y",
#     "cntGetFlg":"N",
#     "annotationGetFlg":"Y",
#     "sectionHeaderFlg":"1",
#     "replaceSpChars":"0"
# }

# r = requests.get(url, params=params)
# r.raise_for_status()

# data = r.json()
# headline = pd.DataFrame(data["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"]["VALUE"])

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

