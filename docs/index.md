# CommPy Documentation

Welcome to the CommPy documentation. This site contains usage notes and API documentation for the `commpy` package. The documentation lives in the `docs/` folder so it can be published easily with GitHub Pages.

## Contents

- [channelCoding module](channelCoding.md) — channel models and examples.

## Publishing to GitHub Pages

You can publish this `docs/` folder as a GitHub Pages site from the repository settings.

Steps:

1. Push the branch to GitHub (for example, `main`).
2. In your repository on GitHub, go to Settings → Pages.
3. Under "Source" choose "Deploy from a branch", select branch `main` and folder `/docs`.
4. Save. The site will be available at `https://<your-username>.github.io/CommPy/` (may take a minute to build).

Alternatively, add a GitHub Actions workflow to build and deploy documentation automatically.

If you'd like, I can add a simple workflow to auto-publish the `docs/` folder.
