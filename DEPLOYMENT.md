# 🚀 Free Deployment Guide

## Option 1: Streamlit Community Cloud (⭐ RECOMMENDED)

### Requirements:

- GitHub account
- Streamlit app (✓ you have this)

### Steps:

1. **Push to GitHub**

   ```bash
   git init
   git add .
   git commit -m "Add Sound Threat Detection System"
   git push -u origin main
   ```

2. **Deploy**
   - Go to: https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"
   - Repository: `sadhikasahana/Sound-Threat-Detection-System`
   - Branch: `main`
   - File: `app.py`
   - Click Deploy!

3. **Your app will be live at:** `https://<username>-sound-threat-detection-system-<random>.streamlit.app`

**Pricing:** FREE  
**Limits:** 3 apps, ~1GB RAM, CPU-only  
**Auto-updates:** Yes (from GitHub)

---

## Option 2: Hugging Face Spaces

### Steps:

1. Create new Space: https://huggingface.co/new-space
2. Choose "Streamlit"
3. Clone the repo and push code
4. Auto-deploys

**Pricing:** FREE  
**Limits:** 16GB storage, ephemeral /tmp storage

---

## Option 3: Railway.app

### Steps:

1. Go to: https://railway.app
2. Connect GitHub account
3. Create new project → Deploy from GitHub
4. Select your repo
5. Railway auto-detects Streamlit

**Pricing:** $5 credit free/month  
**Limits:** CPU-only initially

---

## Pre-Deployment Checklist:

- [x] Relative paths in `app.py` (already fixed)
- [x] `.streamlit/config.toml` created (already created)
- [ ] All model files committed to Git
- [ ] `requirements.txt` updated
- [ ] No hardcoded absolute paths

---

## File Size Warning:

Your model is likely >500MB. For GitHub:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.keras"
git lfs track "*.csv"

git add .
git commit -m "Add model files with Git LFS"
git push
```

---

## Recommended Approach:

1. **Immediate:** Use Streamlit Community Cloud (easiest)
2. **Scale later:** Move to Railway/Hugging Face if needed
3. **Optional:** Add GitHub LFS for large model files

---

## Troubleshooting:

**"Model not found"**

- Check relative paths in `app.py`
- Ensure model files are in repo

**"Out of memory"**

- Deploy to Hugging Face Spaces (more RAM)
- Or split model into smaller components

**"App runs slow"**

- Normal on free tier (CPU-only)
- Upgrade to Railway for faster performance
