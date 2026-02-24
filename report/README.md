# OVO Project Report - DGUNet for Image Denoising

## Files

- `main.tex` - Main 4-page report (required submission)
- `appendix.tex` - Separate 4-page appendix (optional)
- `full_report.tex` - Combined version with main + appendix

## Compilation

### Using pdflatex
```bash
pdflatex full_report.tex
pdflatex full_report.tex  # Run twice for references
```

### Using latexmk (recommended)
```bash
latexmk -pdf full_report.tex
```

## Figures

Create a `figures/` directory and add:
- `stage_progression_placeholder.png` - Stage-by-stage PSNR plot

You can generate this figure using `visualize_stages.py`:
```bash
python visualize_stages.py --checkpoint path/to/model.pth --image path/to/test.png
```

## Structure

### Main Report (~4 pages)
1. Introduction
2. Mathematical Framework (PGD, unfolding)
3. Network Architecture (GDM, PMM, ISFF)
4. Experiments (cross-domain, ablations, own images)
5. Critical Analysis
6. Conclusion
7. References

### Appendix (~4 pages)
1. Detailed Architecture
2. Training Configuration
3. Extended Ablation Results
4. Own Images Testing Protocol
5. Code Structure

## Note

Replace placeholder values in tables with your actual experimental results before submission.
