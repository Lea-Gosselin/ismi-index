"""Analyse reproductible EA/UK/JP pour le dépôt ismi-index.

Exécuter depuis la racine du dépôt :
    python analyse_persistance_ea_uk_jp.py

Entrées : input/{EA,UK,JP}IPC.parquet, colonne HEADLINE.
Sorties : results/persistence_comparison/{tables,figures}.

La série HEADLINE est supposée être un indice SA, strictement positif.
Le code ne télécharge ni ne modifie les données sources.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from scipy.stats import jarque_bera, norm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

try:
    from fismi.io import load_input
except ImportError as exc:
    raise SystemExit("Lancer le script à la racine de ismi-index (pip install -e .)") from exc

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'persistence_comparison'
TABLES = OUT / 'tables'
FIGURES = OUT / 'figures'
for p in [TABLES, FIGURES]:
    p.mkdir(parents=True, exist_ok=True)

COUNTRIES = ['EA', 'UK', 'JP']
# Grille complète mais raisonnable. Augmenter au besoin.
P_MAX = 3
Q_MAX_MOM = 4
Q_MAX_YOY = 12
AUTOLAG = 'BIC'
MAX_ADF_LAG = 18
BIC_TOLERANCE_AR1 = 2.0
LJUNGBOX_CRITICAL_P = 0.05
TVP_STARTS = 3
TVP_MAXITER = 450
BURN = 12  # exclut les premières innovations pour diagnostics communs
RANDOM_SEED = 271828

BLUE, ORANGE, TEAL, INK, MUTED = '#2255E8', '#F4A340', '#27A59A', '#263445', '#9AA8B5'
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13,
                     'axes.spines.top': False, 'axes.spines.right': False})


def input_inflation(country: str):
    prices, _ = load_input(country)
    if 'HEADLINE' not in prices:
        raise ValueError(f'{country}: HEADLINE absent')
    headline = pd.to_numeric(prices['HEADLINE'], errors='coerce')
    headline.index = pd.to_datetime(headline.index).to_period('M').to_timestamp()
    headline = headline.sort_index()
    if headline.index.has_duplicates:
        raise ValueError(f'{country}: dates dupliquées')
    headline = headline.dropna()
    if (headline <= 0).any():
        raise ValueError(f'{country}: prix nuls/négatifs')
    expected = pd.date_range(headline.index.min(), headline.index.max(), freq='MS')
    if not expected.equals(headline.index):
        raise ValueError(f'{country}: historique HEADLINE incomplet; ne pas utiliser diff après dropna')
    logprice = np.log(headline)
    # unités : points de pourcentage (et non décimales)
    mom = 100 * logprice.diff()
    yoy = 100 * logprice.diff(12)
    return headline, {'MoM': mom.dropna(), 'YoY': yoy.dropna()}


def stationarity_table(country, frequencies):
    rows = []
    for freq, x in frequencies.items():
        for regression in ('c', 'ct'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                a = adfuller(x, maxlag=min(MAX_ADF_LAG, len(x)//5),
                             regression=regression, autolag=AUTOLAG)
                try:
                    k = kpss(x, regression=regression, nlags='auto')
                except (ValueError, np.linalg.LinAlgError):
                    k = (np.nan, np.nan)
            rows.append({'country': country, 'frequency': freq,
                         'spec': regression, 'T':len(x),
                         'first':str(x.index.min().date()), 'last':str(x.index.max().date()),
                         'ADF_stat':a[0], 'ADF_p':a[1], 'ADF_lags':a[2],
                         'KPSS_stat':k[0], 'KPSS_p':k[1],
                         'ADF_reject_5%':a[1]<.05,
                         'KPSS_reject_5%':k[1]<.05})
    return pd.DataFrame(rows)


def fit_fixed_arma(x, p, q):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = ARIMA(x, order=(p,0,q), trend='c',
                       enforce_stationarity=True,
                       enforce_invertibility=True).fit(
                           method_kwargs={'maxiter':300})
    if not result.mle_retvals.get('converged', True):
        raise RuntimeError('ARMA did not converge')
    return result


def select_fixed_models(country, frequencies):
    tables = {}
    fitted = {}
    for freq, x in frequencies.items():
        candidates = []
        qmax = Q_MAX_MOM if freq == 'MoM' else Q_MAX_YOY
        for p,q in product(range(P_MAX+1),range(qmax+1)):
            try:
                mod = fit_fixed_arma(x, p, q)
                fitted[(freq,p,q)] = mod
                err = np.asarray(mod.filter_results.standardized_forecasts_error[0],float)
                err = err[BURN:]
                err = err[np.isfinite(err)]
                lbp = acorr_ljungbox(err,lags=[12],return_df=True)['lb_pvalue'].iloc[0]
                candidates.append({'country':country,'frequency':freq,'p':p,'q':q,
                                   'AIC':mod.aic,'BIC':mod.bic,'LL':mod.llf,
                                   'LB12_p':lbp,'T':len(x)})
            except Exception:
                pass
        if not candidates:
            raise RuntimeError(f'{country}/{freq}: aucune estimation ARMA convergente')
        tables[freq] = pd.DataFrame(candidates).sort_values('BIC').reset_index(drop=True)
    return tables, fitted


def choose_mom_q(table):
    """Préférer AR(1) seulement si BIC proche du meilleur et LB non rejeté."""
    best = table.iloc[0]
    ar1 = table.loc[(table.p == 1) & (table.q == 0)]
    if len(ar1) and ar1.iloc[0].BIC - best.BIC <= BIC_TOLERANCE_AR1 \
            and ar1.iloc[0].LB12_p >= LJUNGBOX_CRITICAL_P:
        return 0, 'AR(1) MoM adéquat et parcimonieux'
    one_ar = table.loc[table.p == 1].sort_values('BIC')
    if one_ar.empty:
        raise RuntimeError('ARMA(1,q) non disponible')
    chosen = one_ar.iloc[0]
    return int(chosen.q), 'ARMA(1,q) MoM retenu sur BIC conditionnel à p=1'


def is_invertible(theta, eps=1e-4):
    if len(theta) == 0:
        return True
    # racines de 1 + theta_1 L + ... + theta_q L^q
    coeffs = np.r_[theta[::-1],1.0]
    coeffs = np.trim_zeros(coeffs,trim='f')
    roots = np.roots(coeffs) if len(coeffs)>1 else []
    return all(abs(r)>1+eps for r in roots)


def unpack(par,q):
    mu, rhobar, F = par[:3]
    theta = np.asarray(par[3:3+q])
    D, B = np.exp(par[-2:])
    return mu,rhobar,F,theta,D,B


def tvp_filter(par, pi, q, store=True):
    """Kalman pour ARMA(1,q) avec AR variable et MA constants.

    Etat = [xi_t, eps_t, eps_{t-1},...,eps_{t-q}]
    xi_t = F xi_{t-1} + u_t; rho_t = rho_bar + xi_t.
    Observation sans bruit séparé: eps_t fait partie de l'état, comme
    dans fismi/models/kalAlgoArma.py. Sa variance D² entre donc dans Z P Z'.
    """
    y = np.asarray(pi,dtype=float)
    if len(y)<30:
        raise ValueError('Sample too short')
    mu,rhobar,F,theta,D,B = unpack(par,q)
    if not all(np.isfinite(par)) or abs(F)>=0.995 or abs(rhobar)>=0.995 \
            or D<=0 or B<=0 or not is_invertible(theta):
        return None if store else 1e30

    n=q+2
    Tmat=np.zeros((n,n))
    Tmat[0,0]=F
    if q:
        Tmat[2:,1:-1]=np.eye(q)
    W=np.zeros((n,n))
    W[0,0]=B**2
    W[1,1]=D**2

    state=np.zeros(n)
    P=np.diag([B**2/(1-F**2)]+[D**2]*(n-1))
    nobs=len(y)-1
    pred=np.zeros(nobs); var=np.zeros(nobs); resid=np.zeros(nobs)
    filtered=np.zeros(nobs); smooth_dates=np.arange(1,len(y))
    P_pred=np.zeros((nobs,n,n)) if store else None
    P_filt=np.zeros((nobs,n,n)) if store else None
    S_pred=np.zeros((nobs,n)) if store else None
    S_filt=np.zeros((nobs,n)) if store else None
    loglike=0.0

    for t in range(nobs):
        if t:
            state=Tmat@state
            P=Tmat@P@Tmat.T+W
            P=.5*(P+P.T)
        x=y[t]-mu
        Z=np.r_[x,1.0,theta]
        f=mu+rhobar*x+Z@state
        v=float(Z@P@Z)
        if not np.isfinite(v) or v<1e-12 or not np.isfinite(f):
            return None if store else 1e30
        e=y[t+1]-f
        loglike+=-.5*(np.log(2*np.pi)+np.log(v)+e**2/v)
        K=P@Z/v
        if store:
            S_pred[t]=state
            P_pred[t]=P
        state=state+K*e
        # Joseph update: H=0 since contemporaneous eps is in the state
        M=np.eye(n)-np.outer(K,Z)
        P=M@P@M.T
        P=.5*(P+P.T)
        if store:
            S_filt[t]=state
            P_filt[t]=P
        pred[t]=f;var[t]=v;resid[t]=e;filtered[t]=rhobar+state[0]

    if not store:
        return -loglike
    S_smooth=S_filt.copy()
    P_smooth=P_filt.copy()
    for t in range(nobs-2,-1,-1):
        a=P_filt[t]@Tmat.T
        J=a@np.linalg.pinv(P_pred[t+1],rcond=1e-12)
        S_smooth[t]=S_filt[t]+J@(S_smooth[t+1]-S_pred[t+1])
        P_smooth[t]=P_filt[t]+J@(P_smooth[t+1]-P_pred[t+1])@J.T
        P_smooth[t]=.5*(P_smooth[t]+P_smooth[t].T)
    return {'pred':pred,'var':var,'resid':resid,
            'rho_filtered':filtered,'rho_smoothed':rhobar+S_smooth[:,0],
            'rho_var_smoothed':np.maximum(P_smooth[:,0,0],0),
            'loglike':loglike,'q':q,'params':par}


def estimate_tvp(x, q, fixed_result, starts=TVP_STARTS):
    sd=max(float(x.std()),.04)
    mu=float(x.mean())
    rho0=float(fixed_result.arparams[0]) if len(fixed_result.arparams) else .3
    theta=np.array(fixed_result.maparams[:q],dtype=float) if q else np.array([])
    theta=np.clip(theta,-1.8,1.8)
    if not is_invertible(theta): theta=np.zeros(q)
    x0=np.r_[mu,np.clip(rho0,-.95,.95),.70,theta,
             np.log(sd*.7),np.log(.04)]
    bounds=[(mu-4*sd,mu+4*sd),(-.985,.985),(-.985,.985)]
    bounds += [(-1.99,1.99)]*q
    bounds += [(np.log(sd/30),np.log(sd*5)),(np.log(1e-5),np.log(.8))]
    rng=np.random.default_rng(RANDOM_SEED)
    estimates=[]
    for j in range(starts):
        xstart=x0.copy()
        if j:
            xstart[1]=np.clip(xstart[1]+rng.normal(0,.12),-.85,.85)
            xstart[2]=rng.uniform(.1,.85)
            xstart[-1]=np.log(rng.uniform(.01,.12))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res=minimize(tvp_filter,xstart,args=(x.to_numpy(),q,False),
                         method='L-BFGS-B',bounds=bounds,
                         options={'maxiter':TVP_MAXITER,'ftol':1e-10})
        if np.isfinite(res.fun) and res.fun<1e25:
            estimates.append(res)
    if not estimates:
        raise RuntimeError('TVP optimization failed')
    best=min(estimates,key=lambda z:z.fun)
    if not best.success:
        warnings.warn(f'TVP optimization status: {best.message}')
    result=tvp_filter(best.x,x.to_numpy(),q,store=True)
    result['opt_success']=bool(best.success)
    result['opt_message']=str(best.message)
    result['dates']=x.index[1:]
    result['observed']=x.iloc[1:].to_numpy()
    k=len(best.x); T=len(x)-1
    result['AIC']=-2*result['loglike']+2*k
    result['BIC']=-2*result['loglike']+k*np.log(T)
    return result


def frozen_acf1(rho, theta):
    # ACF locale non définie si |rho| >= 1.
    if not np.isfinite(rho) or abs(rho) >= 1:
        return np.nan
    process=ArmaProcess(np.r_[1.,-rho],np.r_[1.,theta])
    return float(process.acf(lags=2)[1])


def diagnostics(z, country, label, q):
    z=np.asarray(z,dtype=float)[BURN:]
    z=z[np.isfinite(z)]
    lbs=acorr_ljungbox(z,lags=[6,12,24],return_df=True)
    sq=acorr_ljungbox(z*z,lags=[6,12,24],return_df=True)
    jb=jarque_bera(z)
    arch=het_arch(z,nlags=12)
    row={'country':country,'model':label,'T':len(z),'z_mean':z.mean(),
         'z_std':z.std(ddof=1),'JB_stat':float(jb.statistic),
         'JB_p':float(jb.pvalue),'ARCH_LM':float(arch[0]),
         'ARCH_p':float(arch[1])}
    for lag in (6,12,24):
        row[f'LB{lag}_p']=float(lbs.loc[lag,'lb_pvalue'])
        row[f'LBsq{lag}_p']=float(sq.loc[lag,'lb_pvalue'])
    return row, z


def fig_save(fig, country, name):
    fn=FIGURES/f'{country}_{name}'
    fig.savefig(str(fn)+'.png',dpi=230,bbox_inches='tight')
    fig.savefig(str(fn)+'.pdf',bbox_inches='tight')
    plt.close(fig)


def date_style(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x',labelrotation=30)
    ax.grid(alpha=.18)
    ax.margins(x=0)


def plot_persistence(country, result):
    dates=result['dates']; rho=result['rho_smoothed']
    se=np.sqrt(result['rho_var_smoothed']);z90=norm.ppf(.95)
    mu,rhobar,F,theta,D,B=unpack(result['params'],result['q'])
    fig,axes=plt.subplots(2,1,figsize=(15,8),sharex=True,layout='constrained')
    axes[0].plot(dates,result['rho_filtered'],color=ORANGE,lw=1.25,label='Filtered')
    axes[0].axhline(rhobar,color=TEAL,ls='--',label=f'Long-run mean: {rhobar:.3f}')
    axes[0].set_title(f'{country}: filtered persistence',loc='left')
    axes[1].fill_between(dates,rho-z90*se,rho+z90*se,color=BLUE,alpha=.16,label='90% conditional CI')
    axes[1].plot(dates,rho,color=BLUE,lw=1.6,label='Smoothed')
    axes[1].axhline(rhobar,color=TEAL,ls='--',label=f'Long-run mean: {rhobar:.3f}')
    axes[1].set_title(f'{country}: smoothed persistence',loc='left')
    for ax in axes:
        ax.axhline(1.,color='#BB4350',ls=':',lw=.9)
        ax.set_ylabel(r'$\rho_t$')
        ax.legend(loc='upper left',frameon=False,ncol=3)
        date_style(ax)
    fig_save(fig,country,'persistence')


def plot_forecasts(country, result, frequency, frequencies):
    mom,yoy=frequencies['MoM'],frequencies['YoY']
    forecast=pd.Series(result['pred'],index=result['dates'])
    observed=frequencies[frequency]
    if frequency=='MoM':
        target=yoy
        reconstructed=yoy.shift(1)+forecast-mom.shift(12)
        title='YoY reconstructed from MoM forecast'
    else:
        target=mom
        reconstructed=forecast-yoy.shift(1)+mom.shift(12)
        title='MoM reconstructed from YoY forecast'
    fig,axes=plt.subplots(2,1,figsize=(15,8),layout='constrained')
    for ax,o,p,title_ in [(axes[0],observed,forecast,f'{frequency} direct one-step forecast'),
                           (axes[1],target,reconstructed,title)]:
        frame=pd.concat([o.rename('obs'),p.rename('pred')],axis=1).dropna()
        rmse=np.sqrt(np.mean((frame.obs-frame.pred)**2))
        ax.plot(frame.index,frame.obs,color=MUTED,lw=1.5,label='Observed')
        ax.plot(frame.index,frame.pred,color=BLUE,lw=1.25,label='Predicted')
        ax.set_title(f'{title_} | RMSE={rmse:.3f} pp',loc='left')
        ax.set_ylabel('Inflation (pp)')
        ax.legend(loc='upper left',frameon=False,ncol=2)
        date_style(ax)
    fig_save(fig,country,'forecasts')


def plot_diagnostics(country, result, z):
    d=result['dates'][BURN:]
    fig,axes=plt.subplots(3,2,figsize=(15,12),layout='constrained')
    ax=axes[0,0]
    ax.plot(d,z,color=BLUE,lw=1)
    ax.axhline(1.96,color=ORANGE,ls='--');ax.axhline(-1.96,color=ORANGE,ls='--')
    ax.set_title('Standardized one-step innovations',loc='left')
    date_style(ax)
    lag=min(36,len(z)//3)
    plot_acf(z,lags=lag,zero=False,ax=axes[0,1])
    axes[0,1].set_title('ACF standardized innovations',loc='left')
    plot_pacf(z,lags=lag,zero=False,method='ywm',ax=axes[1,0])
    axes[1,0].set_title('PACF standardized innovations',loc='left')
    axes[1,1].hist(z,bins=35,density=True,color=BLUE,alpha=.5)
    xx=np.linspace(z.min(),z.max(),300)
    axes[1,1].plot(xx,norm.pdf(xx),color=INK,ls='--',label='N(0,1)')
    axes[1,1].legend(frameon=False)
    axes[1,1].set_title('Standardized residual histogram',loc='left')
    from scipy.stats import probplot
    probplot(z,dist='norm',plot=axes[2,0])
    axes[2,0].set_title('Normal QQ plot',loc='left')
    plot_acf(z*z,lags=lag,zero=False,ax=axes[2,1])
    axes[2,1].set_title('ACF squared innovations',loc='left')
    fig_save(fig,country,'diagnostics')


def plot_local_propagation(country,result,frequencies):
    _,_,_,theta,_,_=unpack(result['params'],result['q'])
    rho=result['rho_smoothed']
    irf1=rho+(theta[0] if len(theta) else 0.)
    acf1=np.array([frozen_acf1(r,theta) for r in rho])
    fig,axes=plt.subplots(2,1,figsize=(15,7),sharex=True,layout='constrained')
    axes[0].plot(result['dates'],irf1,color=BLUE,label=r'IRF(1): $\rho_t+\theta_1$')
    axes[0].set_title(f'{country}: local frozen-parameter IRF at h=1',loc='left')
    axes[1].plot(result['dates'],acf1,color=TEAL,label='Local model-implied ACF(1)')
    freq=frequencies[result['frequency']]
    empirical=float(acf(freq,nlags=1,fft=True)[1])
    axes[1].axhline(empirical,ls='--',color=INK,label=f'Sample ACF(1) = {empirical:.3f}')
    axes[1].set_title('ACF(1): conditional ARMA versus empirical',loc='left')
    for ax in axes:
        ax.legend(loc='upper left',frameon=False)
        date_style(ax)
    fig_save(fig,country,'acf_irf')
    return float(np.nanmean(acf1)),float(np.mean(irf1)),empirical


def main():
    adf_rows=[];arma_rows=[];tvp_rows=[];diag_rows=[]
    for country in COUNTRIES:
        print(f'\n========== {country} ==========')
        headline,inf=input_inflation(country)
        print('Headline',headline.index[0],headline.index[-1],len(headline))
        st=stationarity_table(country,inf)
        adf_rows.append(st)
        print(st[['frequency','spec','ADF_p','KPSS_p','ADF_lags']].round(4).to_string(index=False))
        rankings,fixed=select_fixed_models(country,inf)
        for freq,tab in rankings.items():
            arma_rows.append(tab)
            print(f'\n{freq} best fixed ARMA candidates:\n',tab.head(5).round(3).to_string(index=False))
        q,reason=choose_mom_q(rankings['MoM'])
        # Yoy remains a robustness specification (not the automatic preferred one)
        yoy_ar1_candidates=rankings['YoY'].loc[rankings['YoY'].p==1].sort_values('BIC')
        q_yoy=int(yoy_ar1_candidates.iloc[0].q)
        print('\nMoM selection:',reason,'q=',q)
        print('YoY ARMA(1,q) benchmark q=',q_yoy)
        # TVP-AR(1) benchmark and TVP-ARMA(1,q) preferred
        cases=[('MoM',0,'MoM_AR1_benchmark')]
        if q!=0: cases.append(('MoM',q,'MoM_preferred'))
        cases.append(('YoY',q_yoy,'YoY_benchmark'))
        tvps={}
        for freq,qmodel,label in cases:
            print('Fitting TVP',country,freq,qmodel,label,flush=True)
            fixed_mod=fixed[(freq,1,qmodel)]
            tvp=estimate_tvp(inf[freq],qmodel,fixed_mod)
            tvp['frequency']=freq
            tvps[label]=tvp
            param=unpack(tvp['params'],qmodel)
            row={'country':country,'model':label,'frequency':freq,'q':qmodel,
                 'mu':param[0],'rho_bar':param[1],'F':param[2],
                 'theta':repr(np.round(param[3],5).tolist()),
                 'sigma_eps':param[4],'sigma_u':param[5],
                 'LL':tvp['loglike'],'AIC':tvp['AIC'],'BIC':tvp['BIC'],
                 'converged':tvp['opt_success'],
                 'rmse':np.sqrt(np.mean(tvp['resid'][BURN:]**2)),
                 'nobs':len(tvp['resid']),
                 'share_abs_rho_ge_1':float(np.mean(np.abs(tvp['rho_smoothed']) >= 1))}
            row['mean_ACF1_model'],row['mean_IRF1'],row['sample_ACF1']=plot_local_propagation(country,tvp,inf)
            tvp_rows.append(row)
            z=tvp['resid']/np.sqrt(tvp['var'])
            diagnostic,z_plot=diagnostics(z,country,label,qmodel)
            diag_rows.append(diagnostic)
            if label=='MoM_preferred' or (label=='MoM_AR1_benchmark' and q==0):
                plot_persistence(country,tvp)
                plot_forecasts(country,tvp,freq,inf)
                plot_diagnostics(country,tvp,z_plot)
        print('TVP models:',[(k,round(v['BIC'],2)) for k,v in tvps.items()])

    st=pd.concat(adf_rows,ignore_index=True)
    ar=pd.concat(arma_rows,ignore_index=True)
    tv=pd.DataFrame(tvp_rows)
    di=pd.DataFrame(diag_rows)
    st.to_csv(TABLES/'stationarity.csv',index=False)
    ar.to_csv(TABLES/'arma_grid.csv',index=False)
    tv.to_csv(TABLES/'tvp_models.csv',index=False)
    di.to_csv(TABLES/'residual_diagnostics.csv',index=False)
    for name,table in [('stationarity',st),('tvp_models',tv),('residual_diagnostics',di)]:
        (TABLES/f'{name}.tex').write_text(table.to_latex(index=False,float_format='%.4f'),encoding='utf-8')
    print('\n============== TVP SUMMARY ==============')
    print(tv.drop(columns=['theta']).round(4).to_string(index=False))
    print('\n============== DIAGNOSTICS ==============')
    print(di.round(4).to_string(index=False))
    print('\nResults in:',OUT)


if __name__=='__main__':
    main()
