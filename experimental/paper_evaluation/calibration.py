"""Dimensionless optimization of the unchanged nonnegative affine Gaussian NLL."""
import numpy as np
from scipy.optimize import minimize
from rmdm_noise_estimation.calibration import VarianceCalibration

def fit_variance_calibration(raw_by_rate,error_by_rate,*,floor=1e-8):
    rates=sorted(raw_by_rate,key=float)
    if not rates or set(rates)!=set(error_by_rate):raise ValueError('Mismatched rates')
    raw=[np.asarray(raw_by_rate[r],dtype=np.float64) for r in rates]
    error=[np.asarray(error_by_rate[r],dtype=np.float64) for r in rates]
    if any(x.shape!=y.shape or x.size==0 or not np.isfinite(x).all() or not np.isfinite(y).all() or (x<0).any() or (y<0).any() for x,y in zip(raw,error)):raise ValueError('Invalid calibration arrays')
    raw_scale=max(float(np.mean([x.mean() for x in raw])),floor)
    error_scale=max(float(np.mean([x.mean() for x in error])),floor)
    pairs=[(x/raw_scale,y/error_scale) for x,y in zip(raw,error)]
    threshold=floor/error_scale
    def objective(p):
        values=[];grads=[]
        for x,y in pairs:
            proposed=p[0]*x+p[1];v=np.maximum(proposed,threshold)
            values.append(np.mean(np.log(v)+y/v))
            d=(1/v-y/v**2)*(proposed>threshold)
            grads.append([np.mean(d*x),np.mean(d)])
        return float(np.mean(values)),np.mean(grads,axis=0)
    fits=[minimize(objective,x0,jac=True,method='L-BFGS-B',bounds=((0,None),(0,None)),options={'ftol':1e-13,'gtol':1e-8,'maxiter':1000}) for x0 in ([.1,.9],[.5,.5],[.9,.1])]
    good=[]
    for fit in fits:
        value,grad=objective(fit.x)
        kkt=np.where(fit.x>1e-8,np.abs(grad),np.maximum(-grad,0))
        if np.isfinite(value) and np.max(kkt)<1e-5:good.append((value,fit))
    if not good:raise RuntimeError(f'Affine calibration failed KKT check: {[(f.message,f.x.tolist(),f.jac.tolist()) for f in fits]}')
    best=min(good,key=lambda item:item[0])[1]
    return VarianceCalibration(float(best.x[0]*error_scale/raw_scale),float(best.x[1]*error_scale),floor)
