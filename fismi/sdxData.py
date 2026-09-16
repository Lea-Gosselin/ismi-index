import requests
import pandas as pd
import io

def getdata(url: str, colName: str,
            headers={"Accept": "text/csv"},
            params={"format": "SDMX-CSV", "startPeriod": 1996}):

    request_kwargs = {"verify": False}
    if headers:
        request_kwargs["headers"] = headers
    if params:
        request_kwargs["params"] = params

    response = requests.get(url, **request_kwargs)

    if response.status_code == 200:
        temp = pd.read_csv(io.StringIO(response.text))
    else:
        print(f"Failed to retrieve data: Status code {response.status_code}")
        return

    df = temp.pivot(index="TIME_PERIOD", columns=colName, values="OBS_VALUE")
    return df, temp