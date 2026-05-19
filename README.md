# PerovskitePD

ML framework for inorganic perovskite photodetector discovery.

## Results
- CV R2 = 0.911 +/- 0.009
- Test R2 = 0.919
- External MAE = 0.236 eV
- Spearman r = 0.829
- 253 lead-free candidates

## Structure
```
api/          FastAPI backend  → deploy on Render
model/        LightGBM + scaler + features
data/         253 candidates JSON + CSV
frontend/     Web app          → deploy on Netlify
render.yaml   Render config
netlify.toml  Netlify config
```

## Deploy API (Render)
1. Push repo to GitHub
2. Go to render.com → New Web Service
3. Connect repo → Deploy
4. Copy your API URL

## Deploy Frontend (Netlify)
1. Go to netlify.com → Add site → GitHub
2. Publish directory: frontend
3. After deploy, open frontend/index.html
4. Set API_URL to your Render URL
5. Push update

## Citation
Machine Learning Framework for Inorganic Perovskite Photodetector Discovery.
Multi-Family Screening with Physics-Informed Corrections. 2026.
