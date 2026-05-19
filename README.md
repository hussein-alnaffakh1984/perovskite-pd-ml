# PerovskitePD — ML-Driven Photodetector Discovery

A machine learning framework for discovering inorganic perovskite photodetector materials.

## 🔬 About

This web application presents results from a comprehensive ML study screening 370 inorganic perovskite compounds across 4 structural families for photodetector applications.

**Key Results:**
- CV R² = 0.911 ± 0.009
- Test R² = 0.919
- External MAE = 0.236 eV (bias-corrected)
- Spearman ranking r = 0.829 (p = 0.042)
- 253 stable, lead-free photodetector candidates identified

## 🏗️ Structure

```
perovskite-pd-app/
├── index.html          # Main application
├── data/
│   ├── candidates.json          # 253 PD candidates
│   ├── FINAL_CANDIDATES_FULL.csv
│   ├── LITERATURE_COMPARISON.csv
│   └── PAPER_DATA.json
└── netlify.toml        # Deployment config
```

## 🚀 Deploy

### Netlify (Recommended)
1. Fork this repository
2. Connect to Netlify
3. Deploy — no build step required

### Local
```bash
# Any static server works
python -m http.server 8000
# Then open http://localhost:8000
```

## 📊 Dataset

- **Source:** Materials Project (MP API, 2023)
- **Materials:** 370 inorganic perovskites (best-phase selection)
- **Families:** Single ABX₃ (43), Double A₂B'B''X₆ (247), Vacancy-ordered A₂BX₆ (69), Sb/Bi ternary (11)
- **Features:** 26 physics-informed descriptors
- **Target:** DFT-PBE bandgap + Δxc correction

## 📄 Citation

If you use this work, please cite:

```
[Authors]. Machine Learning Framework for Inorganic Perovskite Photodetector Discovery:
Multi-Family Screening with Physics-Informed Δxc Corrections. 2026.
```

## 📜 License

MIT License — see LICENSE file.
