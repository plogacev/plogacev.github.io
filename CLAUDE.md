# plogacev.github.io

Personal website for Pavel Logačev, freelance data scientist.

## Build Commands

```bash
make          # Render the site
make preview  # Live preview with hot reload
make clean    # Remove generated files
```

Or directly:
```bash
quarto render
quarto preview
```

## Structure

- `_quarto.yml` - Site configuration
- `index.qmd` - Homepage
- `about/` - About page
- `projects/` - Project portfolio
- `blog/` - Blog posts
- `contact.qmd` - Contact page
- `assets/css/custom.scss` - Custom styling
- `docs/` - Generated output (GitHub Pages)

## Deployment

Push to main branch. GitHub Pages serves from `docs/` directory.
