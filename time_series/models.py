import numpy as np
from scipy import stats, optimize

class SeasonalRW:
    """
    y_{t+h} = y_{t+h-4} · exp(μ_q + ε)
    որտեղ μ_q-ն տվյալ եռամսյակի (q) լոգարիթմական աճի միջինն է: 
    Վստահության միջակայքը (CI) հիմնված σ²_q · h արժեքի վրա:
    m- պարբերականություն
    _log_growth- միջին աճի տեմպերը
    _log_growth_std- ստանդարտ շեղում
    _y- սկզբնական տվյալները
    _resid - մնացորդները
    """

    def __init__(self, period: int = 4):
        self.m = period
        self._log_growth = None
        self._log_growth_std = None
        self._y = None
        self._resid = None

    def fit(self, y: np.ndarray) -> "SeasonalRW":
        '''
        Հաշվարկում է․
            Յուրաքանչյուր եռամսյակի միջին աճը (_log_growth)
            Այդ աճի անկայունությունը (_log_growth_std)
            Իր սեփական սխալի չափը (_resid)
        '''

        self._y = np.asarray(y, float)
        n, m   = len(self._y), self.m
        log_y  = np.log(self._y)

        log_growth = log_y[m:] - log_y[:-m]
        self._log_growth = np.array(
            [np.mean(log_growth[q::m]) for q in range(m)])
        
        self._log_growth_std = np.array(
            [np.std(log_growth[q::m], ddof=1) if len(log_growth[q::m]) > 1
             else 0.05 for q in range(m)])

        fitted = np.full(n, np.nan)
        for t in range(m, n):
            q = t % m
            fitted[t] = self._y[t - m] * np.exp(self._log_growth[q])
        self._resid = (log_y - np.log(np.where(fitted > 0, fitted, np.nan)))[m:]
        
        return self

    def forecast(self, h: int, alpha_ci: float = 0.10) -> tuple:
        '''
        Կատարում է կանխատեսում․
        h - քանի եռամսյակ է պետք կանխատեսել
        alpha_ci֊ սխալվելու հավանականություն
        ---------------------------------------
        pts- Ամենահավանական արժեքները
        lows- Նվազագույն սպասվելիք արժեքները 
        his- Առավելագույն սպասվելիք արժեքները
        '''
        n  = len(self._y)
        m  = self.m
        z  = stats.norm.ppf(1 - alpha_ci / 2)
        pts, lows, his = [], [], []

        for j in range(1, h + 1):
            q       = (n + j - 1) % m
            base    = np.log(self._y[n - m + q])
            mu_log  = base + self._log_growth[q]
            se      = self._log_growth_std[q] * np.sqrt(max(j // m + 1, 1))
            pts.append(np.exp(mu_log + se**2 / 2))
            lows.append(np.exp(mu_log - z * se))
            his.append(np.exp(mu_log + z * se))
        return np.array(pts), np.array(lows), np.array(his)

    @property
    def residuals(self): return self._resid


#  HOLT-WINTERS (additive, damped trend)  in log-space

class HoltWinters:
    """
        ETS(A,Ad,A) մոդել լոգարիթմական տիրույթում:
        Բաժանում է շարքը Level-ի, Trend-ի և Seasonality-ի:
        --------------------------------------------------
        _params - օպտիմալացված գործակիցները (alpha, beta, gamma, phi)
        _state - մոդելի վերջնական վիճակը (L, B, S)
        _resid - մնացորդները (սխալները)
        _sigma2 - սխալների դիսպերսիան (վարիացիան)
        _fitted_log - պատմական տվյալների հիման վրա ստացված մոդելը
    """

    def __init__(self, period: int = 4):
        self.m = period
        self._params = None
        self._state  = None
        self._resid  = None
        self._sigma2 = None
        self._fitted_log = None

    def _run(self, y_log, a, b, g, phi, L0, B0, S0):
        '''
            հաշվում է Level, Trend և Seasonal 
        '''
        n, m = len(y_log), self.m
        L, B = L0, B0
        S    = np.array(S0, dtype=float)
        f    = np.zeros(n)
        for t in range(n):
            sm  = S[t % m]
            f[t] = L + phi * B + sm
            e    = y_log[t] - f[t]
            Ln   = L + phi * B + a * e
            B    = phi * B + a * b * e
            S[t % m] = sm + (1 - a) * g * e
            L    = Ln
        return f, L, B, S

    def _sse(self, params, y_log):
        '''
            Սխալի քառակուսիների գումարն է հաշվում
        '''
        a, b, g, phi = params[:4]
        if not (0 < a < 1 and 0 < b < 0.5 and 0 < g < 0.5 and 0.7 < phi < 1.0):
            return 1e15
        m  = self.m
        L0 = float(np.mean(y_log[:m]))
        B0 = float((np.mean(y_log[m:2*m]) - np.mean(y_log[:m])) / m)
        S0 = y_log[:m] - L0;  S0 -= S0.mean()
        f, *_ = self._run(y_log, a, b, g, phi, L0, B0, S0)
        return float(np.sum((y_log - f)**2))

    def fit(self, y: np.ndarray) -> "HoltWinters":
        '''
        Գտնում է լավագույն պարամետրերը․
            1. Սահմանում է սկզբնական կետերը (L0, B0, S0)
            2. Nelder-Mead մեթոդով մինիմալացնում է սխալը (SSE)
            3. Ֆիքսում է լավագույն alpha, beta, gamma և phi գործակիցները
        '''
        y_log = np.log(np.maximum(y, 1e-9))
        n, m  = len(y_log), self.m
        L0 = float(np.mean(y_log[:m]))
        B0 = float((np.mean(y_log[m:2*m]) - np.mean(y_log[:m])) / m) if n >= 2*m else 0.0
        S0 = y_log[:m] - L0;  S0 -= S0.mean()

        best_sse, best_p = np.inf, None
        for a0, b0, g0, phi0 in [
            (0.4, 0.05, 0.1, 0.9), (0.6, 0.03, 0.15, 0.85),
            (0.3, 0.02, 0.08, 0.95), (0.5, 0.04, 0.12, 0.88),
            (0.2, 0.01, 0.05, 0.92),
        ]:
            res = optimize.minimize(
                self._sse, [a0, b0, g0, phi0], args=(y_log,),
                method="Nelder-Mead",
                options={"maxiter": 8000, "xatol": 1e-10, "fatol": 1e-10})
            if res.fun < best_sse:
                best_sse, best_p = res.fun, res.x

        a, b, g, phi = best_p
        f, L, B, S = self._run(y_log, a, b, g, phi, L0, B0, S0)
        self._params     = dict(alpha=a, beta=b, gamma=g, phi=phi)
        self._state      = dict(L=L, B=B, S=S.copy())
        self._fitted_log = f
        self._resid      = y_log - f
        self._sigma2     = float(np.var(self._resid, ddof=4))
        return self

    def forecast(self, h: int, alpha_ci: float = 0.10) -> tuple:
        '''
            Կատարում է կանխատեսում․
            h - քանի եռամսյակ առաջ նայել
            phi - թրենդի մարման գործակից (թույլ չի տալիս անվերջ աճ)
            ---------------------------------------
            pts - միջին սպասվելիք արժեքները
            lows/his - վստահության միջակայքի սահմանները
        '''

        L, B, S = (self._state["L"], self._state["B"], self._state["S"].copy())
        phi     = self._params["phi"]
        a       = self._params["alpha"]
        m       = self.m
        n       = len(self._fitted_log)
        z       = stats.norm.ppf(1 - alpha_ci / 2)
        pts, lows, his = [], [], []
        c = 0.0
        for j in range(1, h + 1):
            phi_j = sum(phi**(k + 1) for k in range(j))
            st    = S[(n + j - 1) % m]
            mu    = L + phi_j * B + st
            c    += (1 + a * phi_j)**2
            var_h = self._sigma2 * (1 + a**2 * c)
            se    = np.sqrt(max(var_h, 1e-12))
            pts.append(np.exp(mu + var_h / 2))
            lows.append(np.exp(mu - z * se))
            his.append(np.exp(mu + z * se))
        return np.array(pts), np.array(lows), np.array(his)

    @property
    def residuals(self): return self._resid
    @property
    def fitted_orig(self): return np.exp(self._fitted_log)



#  MODEL C — THETA METHOD

class ThetaModel:
    """
        Seasonal Theta մոդել:
        Շարքը բաժանում է թրենդային և կարճաժամկետ տատանումների գծերի:
        --------------------------------------------------
        _seasonal - սեզոնայնության գործակիցները (ամեն եռամսյակի համար)
        _alpha - SES (Simple Exponential Smoothing) հարթեցման գործակիցը
        _drift - երկարաժամկետ թրենդի թեքությունը
        _L_ses - կարճաժամկետ գծի վերջին մակարդակը
        _intercept - թրենդային գծի սկզբնակետը
    """

    def __init__(self, period: int = 4):
        self.m = period
        self._seasonal = None
        self._alpha    = None
        self._drift    = None
        self._L_ses    = None
        self._intercept = None
        self._resid    = None
        self._sigma2   = None
        self._n        = None
        self._fitted_log = None

    def _seasonal_indices(self, y_log: np.ndarray) -> np.ndarray:
        '''
            Հաշվում է սեզոնայնությունը․
            Հեռացնում է թրենդը և գտնում է յուրաքանչյուր եռամսյակի 
            միջին շեղումը ընդհանուր մակարդակից:
        '''
        n, m = len(y_log), self.m
        half = m // 2
        trend = np.full(n, np.nan)
        for t in range(half, n - half):
            trend[t] = np.mean(y_log[t - half: t + half + 1])
        known = np.where(~np.isnan(trend))[0]
        slope, intercept, *_ = stats.linregress(known, trend[known])
        for t in range(n):
            if np.isnan(trend[t]):
                trend[t] = intercept + slope * t
        de = y_log - trend
        si = np.array([np.median(de[q::m]) for q in range(m)])
        si -= si.mean()
        return si

    def fit(self, y: np.ndarray) -> "ThetaModel":
        '''
            Մոդելավորում է շարքը․
                1. Հանում է սեզոնայնությունը (Deseasonalize)
                2. Գծային ռեգրեսիայով գտնում է թրենդը (Theta-0 գիծ)
                3. Ստեղծում է կորացված գիծ (Theta-2)
                4. Գտնում է օպտիմալ alpha-ն SES-ի համար
        '''
        y_log = np.log(np.maximum(y, 1e-9))
        n, m  = len(y_log), self.m
        self._n = n

        si = self._seasonal_indices(y_log)
        self._seasonal = si
        s_full = np.array([si[t % m] for t in range(n)])
        y_des  = y_log - s_full 

        t_arr = np.arange(n)
        slope, intercept, *_ = stats.linregress(t_arr, y_des)
        self._drift     = slope
        self._intercept = intercept
        trend_line      = intercept + slope * t_arr

        theta2 = 2 * y_des - trend_line

        def ses_sse(alpha):
            a = float(alpha)
            if not 0 < a < 1: return 1e15
            L = theta2[0]; s = 0.0
            for yt in theta2[1:]:
                s += (yt - L)**2; L = a * yt + (1 - a) * L
            return s

        res = optimize.minimize_scalar(ses_sse, bounds=(0.01, 0.99), method="bounded")
        self._alpha = float(res.x)

        L = theta2[0]
        fitted_ses = [L]
        for yt in theta2[1:]:
            L = self._alpha * yt + (1 - self._alpha) * L
            fitted_ses.append(L)
        self._L_ses = L

        fitted_des   = (np.array(fitted_ses) + trend_line) / 2
        fitted_log   = fitted_des + s_full
        self._fitted_log = fitted_log
        self._resid  = y_log - fitted_log
        self._sigma2 = float(np.var(self._resid, ddof=2))
        return self

    def forecast(self, h: int, alpha_ci: float = 0.10) -> tuple:
        '''
            Կատարում է կանխատեսում․
            Միավորում է գծային թրենդը և SES-ի արդյունքը, 
            ապա վերադարձնում է սեզոնայնությունը:
        '''
        n   = self._n
        m   = self.m
        z   = stats.norm.ppf(1 - alpha_ci / 2)
        pts, lows, his = [], [], []
        for j in range(1, h + 1):

            s_fut = self._seasonal[(n + j - 1) % m]

            theta0_h = self._intercept + self._drift * (n + j - 1)

            mu_des   = (self._L_ses + theta0_h) / 2
            mu_log   = mu_des + s_fut

            var_h    = self._sigma2 * (1 + (j - 1) * self._alpha**2)
            se       = np.sqrt(max(var_h, 1e-12))
            pts.append(np.exp(mu_log + var_h / 2))
            lows.append(np.exp(mu_log - z * se))
            his.append(np.exp(mu_log + z * se))
        return np.array(pts), np.array(lows), np.array(his)

    @property
    def residuals(self): return self._resid
    @property
    def fitted_orig(self): return np.exp(self._fitted_log)

