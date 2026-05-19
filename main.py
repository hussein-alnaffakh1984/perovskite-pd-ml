from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, numpy as np, os, json

app = FastAPI(title="PerovskitePD API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "../model")

model    = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
FEATURES = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))

# ============================================================
# Physics data من ورقتنا
# ============================================================
RA  = {"Cs":1.88,"Rb":1.72,"K":1.64,"Na":1.18,"Li":0.90,"Tl":1.70}
RB  = {"Pb":1.19,"Sn":1.10,"Ge":0.73,"Ti":0.605,"Zr":0.72,"Hf":0.71,
       "Ag":1.15,"Na":1.02,"Au":1.37,"In":0.80,"Bi":1.03,"Sb":0.76,
       "Te":0.97,"Pt":0.625,"Pd":0.615,"Cu":0.73,"Ga":0.62,"Tl":1.50,"Li":0.76}
RX  = {"F":1.33,"Cl":1.81,"Br":1.96,"I":2.20}
CHI = {"Cs":0.79,"Rb":0.82,"K":0.82,"Na":0.93,"Li":0.98,"Tl":1.62,
       "Pb":2.33,"Sn":1.96,"Ge":2.01,"Ti":1.54,"Zr":1.33,"Hf":1.30,
       "Ag":1.93,"Au":2.54,"In":1.78,"Bi":2.02,"Sb":2.05,
       "Te":2.10,"Pt":2.28,"Pd":2.20,"Cu":1.90,"Ga":1.81,
       "F":3.98,"Cl":3.16,"Br":2.96,"I":2.66}
MASS= {"Cs":132.91,"Rb":85.47,"K":39.10,"Na":22.99,"Li":6.94,"Tl":204.38,
       "Pb":207.2,"Sn":118.71,"Ge":72.63,"Ti":47.87,"Zr":91.22,"Hf":178.49,
       "Ag":107.87,"Au":196.97,"In":114.82,"Bi":208.98,"Sb":121.76,
       "Te":127.60,"Pt":195.08,"Pd":106.42,"Cu":63.55,"Ga":69.72,
       "F":19.0,"Cl":35.45,"Br":79.90,"I":126.90}
SOC = {"Cs":9150625,"Rb":1874161,"K":130321,"Na":14641,"Li":81,"Tl":43046721,
       "Pb":45212176,"Sn":6250000,"Ge":1048576,"Ti":234256,"Zr":2560000,"Hf":26873856,
       "Ag":4879681,"Au":38950081,"In":5764801,"Bi":47458321,"Sb":6765201,
       "Te":7311616,"Pt":37015056,"Pd":4477456,"Cu":707281,"Ga":923521,
       "F":6561,"Cl":83521,"Br":1500625,"I":7890481}
POL = {"Cs":15.8,"Rb":9.10,"K":0.83,"Na":0.18,"Li":0.03,"Tl":7.5,
       "Pb":6.98,"Sn":7.9,"Ge":5.4,"Ti":14.8,"Zr":17.9,"Hf":16.2,
       "Ag":7.2,"Au":5.8,"In":10.2,"Bi":7.4,"Sb":6.6,
       "Te":5.5,"Pt":4.8,"Pd":4.8,"Cu":6.1,"Ga":8.1,
       "F":0.56,"Cl":2.18,"Br":3.05,"I":5.35}
DXC = {"Pb_I":0.252,"Pb_Br":0.468,"Pb_Cl":0.802,"Pb_F":0.900,
       "Sn_I":0.540,"Sn_Br":0.362,"Sn_Cl":0.500,"Sn_F":0.600,
       "Ge_I":0.359,"Ge_Br":0.409,"Ge_Cl":0.459,"Ge_F":0.500,
       "Bi_I":0.450,"Bi_Br":0.550,"Bi_Cl":0.700,"Bi_F":0.800,
       "Sb_I":0.400,"Sb_Br":0.500,"Sb_Cl":0.620,"Sb_F":0.720,
       "In_I":0.350,"In_Br":0.420,"In_Cl":0.500,"In_F":0.580,
       "Ag_I":0.280,"Ag_Br":0.350,"Ag_Cl":0.430,"Ag_F":0.510,
       "Au_I":0.260,"Au_Br":0.330,"Au_Cl":0.410,"Au_F":0.490,
       "Ti_I":0.400,"Ti_Br":0.470,"Ti_Cl":0.550,"Ti_F":0.630,
       "Zr_I":0.430,"Zr_Br":0.500,"Zr_Cl":0.580,"Zr_F":0.660,
       "Hf_I":0.450,"Hf_Br":0.520,"Hf_Cl":0.600,"Hf_F":0.680,
       "Pt_I":0.480,"Pt_Br":0.550,"Pt_Cl":0.650,"Pt_F":0.730,
       "Pd_I":0.460,"Pd_Br":0.530,"Pd_Cl":0.620,"Pd_F":0.700,
       "Na_I":0.230,"Na_Br":0.290,"Na_Cl":0.350,"Na_F":0.420,
       "Ga_I":0.320,"Ga_Br":0.390,"Ga_Cl":0.460,"Ga_F":0.540,
       "Li_I":0.300,"Li_Br":0.370,"Li_Cl":0.440,"Li_F":0.520}
TE  = {"single":1,"double":2,"vacancy_ordered":3,"Sb_Bi_ternary":4}

BIAS = 0.618  # systematic bias from our validation

def eff(v1_key, v2_key, lk, default=np.nan):
    v1 = lk.get(v1_key, default)
    if v2_key:
        v2 = lk.get(v2_key, default)
        try:
            if not (np.isnan(v1) or np.isnan(v2)):
                return (v1+v2)/2
        except: pass
    return v1

class PredictRequest(BaseModel):
    A:  str
    B1: str
    B2: str = ""
    X:  str
    structure_type: str
    volume_per_atom: float = 40.0
    density:         float = 4.5
    n_sites:         int   = 5
    lat_aniso:       float = 1.0
    formation_energy:float = -2.0

@app.get("/")
def root():
    return {"status":"PerovskitePD API","model":"LightGBM",
            "r2":0.919,"mae_eV":0.236,"spearman":0.829,
            "training_materials":370,"candidates":253}

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/candidates")
def get_candidates():
    path = os.path.join(BASE,"../data/candidates.json")
    with open(path) as f:
        return json.load(f)

@app.post("/predict")
def predict(req: PredictRequest):
    A  = req.A.strip()
    B1 = req.B1.strip()
    B2 = req.B2.strip()
    X  = req.X.strip()
    pt = req.structure_type

    # Validate elements
    if A not in RA:
        raise HTTPException(400, f"Unknown A-site: {A}. Valid: {list(RA.keys())}")
    if B1 not in RB:
        raise HTTPException(400, f"Unknown B-site: {B1}. Valid: {list(RB.keys())}")
    if X not in RX:
        raise HTTPException(400, f"Unknown X-site: {X}. Valid: {list(RX.keys())}")

    rA = RA[A];              rB = eff(B1,B2,RB); rX = RX[X]
    cA = CHI.get(A,np.nan); cB = eff(B1,B2,CHI); cX = CHI.get(X,np.nan)
    mA = MASS.get(A,np.nan);mB = eff(B1,B2,MASS);mX = MASS.get(X,np.nan)
    sB = eff(B1,B2,SOC,0);  sX = SOC.get(X,0)
    pA = POL.get(A,np.nan); pB = eff(B1,B2,POL); pX = POL.get(X,np.nan)

    dxc = DXC.get(f"{B1}_{X}", 0.45)
    if B2: dxc = (dxc + DXC.get(f"{B2}_{X}", 0.45)) / 2

    tol      = (rA+rX) / (np.sqrt(2)*(rB+rX))
    oct_f    = rB/rX
    chi_BX   = cX-cB
    ionicity = 1 - np.exp(-0.25*chi_BX**2)
    bond_BX  = rB+rX
    log_soc  = np.log1p(sB+sX)
    pol_tot  = (pA or 0)+(pX or 0)+(pB or 0)
    ratio_BX = rB/rX
    type_num = TE.get(pt,1)

    fm = {
        "r_A":rA,"r_B":rB,"r_X":rX,
        "chi_A":cA,"chi_B":cB,"chi_X":cX,
        "mass_A":mA,"mass_B":mB,"mass_X":mX,
        "tol":tol,"oct":oct_f,"chi_BX":chi_BX,
        "ionicity":ionicity,"bond_BX":bond_BX,"log_soc":log_soc,
        "pol_A":pA,"pol_X":pX,"pol_total":pol_tot,"ratio_BX":ratio_BX,
        "volume_per_atom_A3":req.volume_per_atom,
        "density_g_cm3":req.density,
        "n_sites":req.n_sites,
        "lat_aniso":req.lat_aniso,
        "type_num":type_num,
        "formation_energy_per_atom_eV":req.formation_energy,
        "delta_xc":dxc
    }

    import pandas as pd
    Xdf = pd.DataFrame([fm])[FEATURES]
    Xs  = scaler.transform(Xdf)
    bg_raw = float(model.predict(Xs)[0])
    bg     = round(bg_raw - BIAS, 3)

    if   bg < 1.4: app_t = "NIR PD"
    elif bg < 2.5: app_t = "Visible PD"
    elif bg < 3.5: app_t = "UV-Vis PD"
    else:          app_t = "Wide Gap"

    formula = f"{A}{B1}{B2}{X}" if pt=="single" else \
              f"{A}2{B1}{B2}{X}6" if pt in ["double","vacancy_ordered"] else \
              f"{A}3{B1}2{X}9"

    return {
        "formula":        formula,
        "bandgap_eV":     bg,
        "bandgap_raw_eV": round(bg_raw, 3),
        "bias_correction": BIAS,
        "application":    app_t,
        "lead_free":      "Pb" not in B1 and "Pb" not in B2,
        "delta_xc":       round(dxc, 3),
        "features":       {k: round(float(v),4) if not (isinstance(v,float) and np.isnan(v)) else None
                           for k,v in fm.items()},
        "model_info": {
            "type":     "LightGBM",
            "cv_r2":    0.911,
            "test_r2":  0.919,
            "mae_eV":   0.236,
            "spearman": 0.829,
            "n_train":  370,
            "families": 4
        }
    }
