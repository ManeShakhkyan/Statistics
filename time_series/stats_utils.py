import numpy as np
from scipy import stats


class StatTests:

    @staticmethod
    def adf(y: np.ndarray, max_lags: int = 4) -> dict:
        """
        Տվյալների ստացիոնարությունը
        """
        y = np.asarray(y, float)
        dy = np.diff(y)
        n  = len(dy)
        lags = min(max_lags, n // 5)
        X = [np.ones(n - lags), y[lags:-1]]
        for k in range(1, lags + 1):
            X.append(dy[lags - k: n - k])
        X = np.column_stack(X)
        Y = dy[lags:]
        b, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ b
        s2    = resid @ resid / max(len(Y) - X.shape[1], 1)
        cov   = s2 * np.linalg.pinv(X.T @ X)
        tau   = b[1] / np.sqrt(max(cov[1, 1], 1e-20))
        p     = float(np.clip(
            stats.norm.cdf(0.6344 + 0.4739 * tau - 0.0605 * tau**2), 0.001, 0.999))
        return dict(tau=round(float(tau), 4), p=round(p, 4), cv5="-2.86",
                    verdict="Stationary ✓" if tau < -2.86 else "Non-stationary ✗")

    @staticmethod
    def ljung_box(resid: np.ndarray, lags: int = 8) -> dict:
        """
        Մնացորդների ավտոկորելյացիան
        """
        r = np.asarray(resid, float)
        r = r[~np.isnan(r)]
        n = len(r)
        if n < lags + 2:
            return dict(Q=np.nan, p=np.nan, verdict="n/a")
        rk = np.array([np.corrcoef(r[:-k], r[k:])[0, 1] for k in range(1, lags + 1)])
        Q  = n * (n + 2) * np.sum(rk**2 / (n - np.arange(1, lags + 1)))
        p  = float(1 - stats.chi2.cdf(Q, df=lags))
        return dict(Q=round(float(Q), 3), p=round(p, 4),
                    verdict="White noise ✓" if p > 0.05 else "Autocorrelation ✗")

    @staticmethod
    def shapiro(resid: np.ndarray) -> dict:
        """
        Մնացորդների նորմալ բաշխվածությունը
        """
        r = np.asarray(resid, float)
        r = r[~np.isnan(r)]
        W, p = stats.shapiro(r)
        return dict(W=round(float(W), 4), p=round(float(p), 4),
                    verdict="Normal ✓" if p > 0.05 else "Non-normal ✗")

    @staticmethod
    def diebold_mariano(actual: np.ndarray,
                         f1: np.ndarray, f2: np.ndarray, h: int = 1) -> dict:
        """
        երկու մոդելների համեմատում
        """
        d  = (actual - f1)**2 - (actual - f2)**2
        n  = len(d)
        mu = np.mean(d)
        v  = np.var(d, ddof=1)
        for k in range(1, max(h, 1)):
            v += 2 * (1 - k / h) * np.mean((d[k:] - mu) * (d[:-k] - mu))
        v  = max(v, 1e-30)
        DM = mu / np.sqrt(v / n) * np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
        p  = float(2 * (1 - stats.t.cdf(abs(DM), df=n - 1)))
        winner = "Model 1 better" if mu < 0 else "Model 2 better"
        sig    = " (significant)" if p < 0.05 else " (not significant)"
        return dict(DM=round(float(DM), 4), p=round(p, 4),
                    verdict=winner + sig)

    @staticmethod
    def cusum(resid: np.ndarray) -> dict:
        """
        Մոդելի կայունությունը ժամանակի մեջ
        """
        r  = np.asarray(resid, float)
        r  = r[~np.isnan(r)]
        n  = len(r)
        cs = np.cumsum(r) / (np.std(r, ddof=1) * np.sqrt(n))
        mx = float(np.max(np.abs(cs)))
        return dict(max_dev=round(mx, 4), bound=0.948,
                    cusum_series=cs,
                    verdict="Stable ✓" if mx < 0.948 else "Structural break ✗")


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = actual > 0
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)

def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    e = actual - pred
    mask = actual > 0
    return dict(
        MAE  = round(float(np.mean(np.abs(e))), 2),
        RMSE = round(float(np.sqrt(np.mean(e**2))), 2),
        MAPE = round(mape(actual, pred), 2),
    )
