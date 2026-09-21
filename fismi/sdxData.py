import requests
import pandas as pd
import io

# Get data from ECB Api
def getEcbData(url: str, colName: str,
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


# Normalise ECOICOP codes from EC "01.1.1.2" into ECB's 6-characters ones "011120"
def normalize_code(code):
    return str(code).replace('.', '').ljust(6, '0')


# Get official ECOICOP codes from EC website
def getEcoicopCodes(namespace="http://data.europa.eu/ed1/ecoicop2/",
                          endpoint="https://publications.europa.eu/webapi/rdf/sparql",
                          lang="en"):
    query = f"""
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT ?concept ?code ?label ?parentCode
    WHERE {{
      ?concept skos:notation ?code ;
               skos:prefLabel ?label .
      FILTER(STRSTARTS(STR(?concept), "{namespace}"))
      FILTER(LANG(?label) = "{lang}")
      OPTIONAL {{
        ?concept skos:broader ?parent .
        ?parent skos:notation ?parentCode .
      }}
    }}
    """

    resp = requests.get(endpoint, params={
        "query": query,
        "format": "application/sparql-results+json"
    })
    resp.raise_for_status()
    data = resp.json()

    rows = [
        {
            "code": b["code"]["value"],
            "label": b["label"]["value"],
            "parent_code": b.get("parentCode", {}).get("value")
        }
        for b in data["results"]["bindings"]
    ]

    official = pd.DataFrame(rows).drop_duplicates()
    official["code_norm"] = official["code"].apply(normalize_code)
    return official



# Get US PCE data from BEA website 

def getBeaData(table_name, api_key):
    """
    Gather 'Underlying Detail' table from BEA (NIPA) at a monthly frequency
    Shape it as a (Dates x Categories) dataframe
    table_name : 'U20404' (quantities), 'U20404' (price), 'U20405' (expenditures in current $)
    """
    
    BEA_URL = "https://apps.bea.gov/api/data"
    
    params = {
        "UserID": api_key,
        "method": "GetData",
        "datasetname": "NIUnderlyingDetail",
        "TableName": table_name,
        "Frequency": "M",
        "Year": "X",
        "ResultFormat": "JSON",
    }
    r = requests.get(BEA_URL, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()["BEAAPI"]["Results"]

    if "Error" in payload:
        raise RuntimeError(payload["Error"])

    df = pd.DataFrame(payload["Data"])
    df["TIME_PERIOD"] = pd.to_datetime(df["TimePeriod"].str.replace("M", "-"), format="%Y-%m")
    df["DataValue"] = pd.to_numeric(df["DataValue"].str.replace(",", ""), errors="coerce")

    # LineDescription original BEA table indentation
    # Hierarchy proxy similar to ECOICOP sub-classes
    # df["indent_level"] = df["LineDescription"].str.len() - df["LineDescription"].str.lstrip().str.len()
    # df["LineDescription"] = df["LineDescription"].str.strip()

    return df[["TIME_PERIOD", "LineNumber", "LineDescription", "SeriesCode",  "DataValue"]] #"indent_level",
