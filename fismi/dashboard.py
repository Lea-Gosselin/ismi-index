"""Export EA/US research results for the static Inflation Monitor website."""
import json
import numpy as np
import pandas as pd
from fismi.io import load_input, ROOT, load_config


def export_dashboard(country):
    country = country.upper()
    config = load_config(country)
    prices, _ = load_input(country)
    headline = prices['HEADLINE'].sort_index().astype(float)
    rho_col = "Rho"
    
    # Year-over-year headline inflation, in percent (simple index change)
    yoy = headline.pct_change(12, fill_method=None) * 100
    ismi = pd.read_parquet(ROOT / 'results' / f'{country}Ismi.parquet')
    persistence = pd.read_parquet(ROOT / 'results' / f'{country}Persistence.parquet')
    
    # Resize for the dashboard...
    sd = "2008-01-01"
    yoy = yoy[yoy.index >= sd]
    ismi = ismi[ismi.index >= sd]
    persistence = persistence[persistence.index >= sd]

    
    required_ismi = {'ISMI', 'ISMI_Pos', 'ISMI_Neg'}
    if not required_ismi.issubset(ismi.columns):
        raise ValueError(f'{country}: missing ISMI columns: {required_ismi - set(ismi.columns)}')

    metadata_path = ROOT / 'results' / f'{country}Persistence.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f'Long-run persistence requires {metadata_path}')
    
    with metadata_path.open(encoding='utf-8') as f:
        metadata = json.load(f)
        
    if 'rho_bar' in metadata:
        rho_bar = float(metadata['rho_bar'])
    else:
        params = metadata.get('parameters', [])
        if len(params) < 2 or not config['persistence'].get('estimate_rho_bar', True):
            raise ValueError(f'{country}: cannot identify estimated long-run persistence (rho_bar)')
        # Current model convention: X0 = [mu, rho_bar, ...].
        rho_bar = float(params[1])

    for frame in (ismi, persistence):
        frame.index = pd.to_datetime(frame.index)
        if frame.index.has_duplicates:
            raise ValueError(f'{country}: duplicate model result dates')

    df = pd.concat([
        ismi[['ISMI', 'ISMI_Pos', 'ISMI_Neg']],
        persistence[[rho_col]].rename(columns={rho_col: 'Rho'}),
        yoy.rename('HeadlineInflation'),
    ], axis=1).sort_index()
    df['RhoBar'] = rho_bar
    df.index.name = 'date'
    df = df.reset_index()
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    output = ROOT / 'dashboard' / 'data'
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'{country}_dashboard.json'
    df.to_json(path, orient='records', indent=2)
    print(f'[{country}] {len(df)} rows -> {path}')
    return path


if __name__ == '__main__':
    for c in ('EA', 'US', 'UK'):
        export_dashboard(c)
