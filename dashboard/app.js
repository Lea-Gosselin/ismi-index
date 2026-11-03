
const state = {
  country: 'US',
  range: 'FULL',
  rows: []
};

const el = id => document.getElementById(id);

const num = v =>
  v === null || v === undefined || v === ''
    ? null
    : (Number.isFinite(Number(v)) ? Number(v) : null);

const date = r => String(r.date || '').slice(0, 10);

const series = (rows, key) => rows.map(r => num(r[key]));


function filtered() {
  if (state.range === 'FULL') {
    return state.rows;
  }

  const last = new Date(date(state.rows.at(-1)));
  const cutoff = new Date(last);

  cutoff.setUTCFullYear(
    cutoff.getUTCFullYear() - parseInt(state.range)
  );

  return state.rows.filter(
    r => new Date(date(r)) >= cutoff
  );
}


function shadePeriods(rows) {
  const shapes = [];
  let start = null;

  for (let i = 0; i < rows.length; i++) {
    const hit = num(rows[i].HeadlineInflation) >= 3;

    if (hit && start === null) {
      start = date(rows[i]);
    }

    if (start !== null && (!hit || i === rows.length - 1)) {
      const end = date(rows[i]);

      shapes.push({
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: start,
        x1: end,
        y0: 0,
        y1: 1,
        line: {
          width: 0
        },
        fillcolor: 'rgba(130,93,159,0.14)',
        layer: 'below'
      });

      start = null;
    }
  }

  return shapes;
}


const trace = (x, y, name, color, extra = {}) => ({
  x,
  y,
  type: 'scatter',
  mode: 'lines',
  name,
  line: {
    color,
    width: 1.6
  },
  connectgaps: false,
  hovertemplate:
    `%{x|%b %Y}<br>${name}: %{y:.3f}<extra></extra>`,
  ...extra
});


function layoutBase() {
  return {
    paper_bgcolor: '#fff',
    plot_bgcolor: '#fff',

    font: {
      family: 'Inter,Arial,sans-serif',
      size: 11,
      color: '#59666d'
    },

    margin: {
      l: 72,
      r: 75,
      t: 75,
      b: 58
    },

    hovermode: 'x unified',

    legend: {
      orientation: 'h',
      x: 0.5,
      xanchor: 'center',
      y: 1.20,
      yanchor: 'bottom',
      font: {
        size: 11
      }
    },

    xaxis: {
      type: 'date',
      showgrid: true,
      gridcolor: '#e9edf0',
      tickformat: '%Y',
      zeroline: false
    },

    yaxis: {
      gridcolor: '#e1e6e8',
      zeroline: false
    },

    showlegend: true
  };
}


function renderPersistenceCI(rows, x, longrun, cfg) {
  const rho = series(rows, 'Rho');
  const rhoStd = series(rows, 'RhoStd');

  // Confidence intervals, from widest to narrowest

    const confidenceLevels = [
    { level: 99, color: 'rgba(239,196,108,0.45)' },
    { level: 95, color: 'rgba(239,196,108,0.35)' },
    { level: 90, color: 'rgba(239,196,108,0.25)' },
    { level: 75, color: 'rgba(239,196,108,0.15)' }
    ];


  const traces = [];

  // --------------------------------------------------
  // Pointwise confidence intervals
  // --------------------------------------------------

  confidenceLevels.forEach(({ level, color }) => {
    const lower = series(rows, `RhoCI${level}Lower`);
    const upper = series(rows, `RhoCI${level}Upper`);

    // Lower boundary
    traces.push({
      x,
      y: lower,
      type: 'scatter',
      mode: 'lines',
      line: {
        width: 0,
        color: 'rgba(0,0,0,0)'
      },
      showlegend: false,
      hoverinfo: 'skip',
      connectgaps: false
    });

    // Upper boundary and shaded interval
    traces.push({
      x,
      y: upper,
      type: 'scatter',
      mode: 'lines',
      name: `${level}% CI`,
      line: {
        width: 0,
        color: 'rgba(0,0,0,0)'
      },
      fill: 'tonexty',
      fillcolor: color,
      connectgaps: false,
      hoverinfo: 'skip',
      showlegend: true
    });
  });

  // --------------------------------------------------
  // Long-run persistence
  // --------------------------------------------------

  if (longrun !== null) {
    traces.push({
      x: [x[0], x.at(-1)],
      y: [longrun, longrun],
      type: 'scatter',
      mode: 'lines',
      name: 'Long-term persistence (ρ̄)',
      line: {
        color: '#efc46c',
        dash: 'dash',
        width: 1.6
      },
      hoverinfo: 'skip'
    });
  }

  // --------------------------------------------------
  // RTS-smoothed persistence
  // --------------------------------------------------

  traces.push({
    x,
    y: rho,
    type: 'scatter',
    mode: 'lines',
    name: 'Smoothed persistence (ρ̂ₜ|T)',
    line: {
      color: '#111',
      width: 2
    },
    customdata: rhoStd,
    connectgaps: false,
    hovertemplate:
      '%{x|%b %Y}' +
      '<br>Smoothed persistence: %{y:.3f}' +
      '<br>Smoothed std. dev.: %{customdata:.3f}' +
      '<extra></extra>'
  });

  // --------------------------------------------------
  // Layout
  // --------------------------------------------------

  const layout = layoutBase();

  layout.yaxis = {
    ...layout.yaxis,
    title: 'AR(1) coefficient',
    zeroline: true,
    zerolinecolor: '#b8b8b8'
  };

  layout.legend = {
    ...layout.legend,
    y: 1.20
  };

  Plotly.react(
    'chart-rho-ci',
    traces,
    layout,
    cfg
  );
}

function render() {
  if (!state.rows.length) {
    return;
  }

  const rows = filtered();
  const x = rows.map(date);

  if (!rows.length) {
    return;
  }

  // Data
  const rho = series(rows, 'Rho');
  const ismi = series(rows, 'ISMI');
  const headline = series(rows, 'HeadlineInflation');
  const pos = series(rows, 'ISMI_Pos');
  const neg = series(rows, 'ISMI_Neg');

  const longrun = num(
    rows.find(r => num(r.RhoBar) !== null)?.RhoBar
  );

  // --------------------------------------------------
  // Chart 1: ISMI and inflation persistence
  // --------------------------------------------------

  const a = layoutBase();

  a.yaxis = {
    ...a.yaxis,
    title: 'ISMI',
    rangemode: 'tozero'
  };

  a.yaxis2 = {
    title: 'AR(1) coefficient',
    overlaying: 'y',
    side: 'right',
    showgrid: false
  };

  a.shapes = shadePeriods(rows);

  const first = [
    trace(
      x,
      ismi,
      'ISMI',
      '#a8a8a8'
    ),

    trace(
      x,
      rho,
      'Persistence (ρ̂ₜ)',
      '#111',
      {
        yaxis: 'y2'
      }
    )
  ];

  if (longrun !== null) {
    first.push({
      x: [x[0], x.at(-1)],
      y: [longrun, longrun],
      type: 'scatter',
      mode: 'lines',
      name: 'Long-term persistence (ρ̄)',

      line: {
        color: '#906bb2',
        dash: 'dash',
        width: 1.5
      },

      yaxis: 'y2',
      hoverinfo: 'skip'
    });
  }

  first.push({
    x: [null],
    y: [null],
    type: 'scatter',
    mode: 'lines',
    name: 'Headline inflation ≥ 3%',

    line: {
      color: 'rgba(130,93,159,.35)',
      width: 10
    },

    hoverinfo: 'skip'
  });

  // --------------------------------------------------
  // Chart 2: Inflation shock momentum and inflation
  // --------------------------------------------------

  const b = layoutBase();

  b.yaxis = {
    ...b.yaxis,
    title: 'Inflation shock momentum breadth',
    tickformat: '.0%',
    zeroline: true,
    zerolinecolor: '#999'
  };

  b.yaxis2 = {
    title: 'Headline inflation (%)',
    overlaying: 'y',
    side: 'right',
    showgrid: false,
    ticksuffix: '%'
  };

  const fillPositive = {
    x,
    y: pos,
    type: 'scatter',
    mode: 'lines',
    name: 'Positive momentum (S⁺)',

    line: {
      color: '#efc46c',
      width: 0.7
    },

    fill: 'tozeroy',
    fillcolor: 'rgba(245,190,78,.33)',

    hovertemplate:
      '%{x|%b %Y}<br>S⁺: %{y:.1%}<extra></extra>'
  };

  const fillNegative = {
    x,
    y: neg.map(
      v => v === null ? null : -Math.abs(v)
    ),

    type: 'scatter',
    mode: 'lines',
    name: 'Negative momentum (S⁻)',

    line: {
      color: '#a9c6f5',
      width: 0.7
    },

    fill: 'tozeroy',
    fillcolor: 'rgba(139,181,247,.3)',

    hovertemplate:
      '%{x|%b %Y}<br>S⁻: %{y:.1%}<extra></extra>'
  };

  const second = [
    fillPositive,
    fillNegative,

    trace(
      x,
      ismi,
      'Inflation Shock Momentum Index',
      '#aaa'
    ),

    trace(
      x,
      headline,
      'Headline inflation',
      '#111',
      {
        yaxis: 'y2'
      }
    )
  ];

  // --------------------------------------------------
  // Plotly configuration
  // --------------------------------------------------


  // --------------------------------------------------
  // Plotly configuration
  // --------------------------------------------------

  const cfg = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: [
      'lasso2d',
      'select2d'
    ]
  };

  // Chart 1: ISMI and persistence
  Plotly.react('chart-rho', first, a, cfg);

  // Chart 2: Inflation shock momentum
  Plotly.react('chart-headline', second, b, cfg);

  // Chart 3: RTS-smoothed persistence and confidence intervals
  renderPersistenceCI(rows, x, longrun, cfg);

  // Metadata
  el('sample').textContent =
    `${rows.length} observations · ${x[0]} to ${x.at(-1)}`;

  el('updated').textContent = x.at(-1);
}


// --------------------------------------------------
// Load dashboard data
// --------------------------------------------------

async function load() {
  el('status').textContent = 'Loading data…';

  try {
    const response = await fetch(
      `data/${state.country}_dashboard.json`,
      {
        cache: 'no-store'
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const rows = await response.json();

    if (!Array.isArray(rows) || !rows.length) {
      throw new Error('Empty JSON');
    }

    const required = [
      'date',
      'ISMI',
      'ISMI_Pos',
      'ISMI_Neg',
      'Rho',
      'RhoBar',
      'HeadlineInflation'
    ];

    const missing = required.filter(
      k => !(k in rows[0])
    );

    if (missing.length) {
      throw new Error(
        'Missing columns: ' + missing.join(', ')
      );
    }

    state.rows = rows
      .filter(
        r => !Number.isNaN(
          new Date(date(r)).getTime()
        )
      )
      .sort(
        (a, b) => date(a).localeCompare(date(b))
      );

    el('status').textContent = '';

    render();

  } catch (err) {
    state.rows = [];

    el('status').textContent =
      `Unable to display ${state.country}: ${err.message}. ` +
      'Re-run python -m fismi.dashboard.';

    Plotly.purge('chart-rho');
    Plotly.purge('chart-headline');
    Plotly.purge('chart-rho-ci');
  }
}


// --------------------------------------------------
// Export CSV
// --------------------------------------------------

function download() {
  const rows = filtered();

  if (!rows.length) {
    return;
  }

  const keys = Object.keys(rows[0]);

  const esc = v =>
    '"' + String(v ?? '').replaceAll('"', '""') + '"';

  const csv = [
    keys.map(esc).join(','),

    ...rows.map(
      r => keys.map(
        k => esc(r[k])
      ).join(',')
    )
  ].join('\r\n');

  const url = URL.createObjectURL(
    new Blob(
      [csv],
      {
        type: 'text/csv;charset=utf-8'
      }
    )
  );

  const a = document.createElement('a');

  a.href = url;
  a.download =
    `${state.country}_inflation_${state.range}.csv`;

  a.click();

  URL.revokeObjectURL(url);
}


// --------------------------------------------------
// Event listeners
// --------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {

  // Country selection
  el('country').addEventListener('change', e => {
    state.country = e.target.value;
    load();
  });

  // Time horizon selection
  document.querySelectorAll('[data-range]').forEach(
    btn => btn.addEventListener('click', () => {

      state.range = btn.dataset.range;

      document.querySelectorAll('[data-range]').forEach(
        b => b.classList.toggle(
          'active',
          b === btn
        )
      );

      render();
    })
  );

  // CSV export
  el('download').addEventListener(
    'click',
    download
  );

  // Initial load
  load();
});
