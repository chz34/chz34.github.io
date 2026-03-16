# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GitHub Pages personal tech blog (`chz34.github.io`) built with **Jekyll + Chirpy theme**. Content is primarily in Chinese, focused on HPC and AI framework engineering.

## Local Development

```bash
# Install dependencies (requires Ruby 3.3+)
bundle install

# Serve locally with live reload at http://localhost:4000
bundle exec jekyll serve

# Build for production
JEKYLL_ENV=production bundle exec jekyll build
```

Deployment is automated via GitHub Actions (`.github/workflows/pages.yml`) on push to `master`. GitHub Pages source must be set to **GitHub Actions** in repository settings (not the legacy "Deploy from branch" mode).

## Content Structure

- `_posts/YYYY-MM-DD-slug.md` — Blog posts (front matter required: `title`, `date`, `categories`, `tags`)
- `_tabs/` — Navigation tab pages (about, categories, tags, archives) — Chirpy-specific
- `_config.yml` — Site configuration (title, social links, theme mode, pagination)

## Adding Content

New posts must follow the naming convention `YYYY-MM-DD-title.md` in `_posts/`. Minimum front matter:

```yaml
---
title: Post Title
date: 2024-01-01 00:00:00 +0800
categories: [Top, Sub]
tags: [tag1, tag2]
---
```

## Theme

Uses `jekyll-theme-chirpy ~> 7.1`. Chirpy provides: dark/light mode toggle, sidebar navigation, TOC, code highlighting, search, categories/tags pages, and archives. Theme docs: [chirpy.cotes.page](https://chirpy.cotes.page).

The old `stylesheets/`, `HPF.md`, and `mindspore.md` files are excluded in `_config.yml` — their content has been migrated to `_posts/`.
