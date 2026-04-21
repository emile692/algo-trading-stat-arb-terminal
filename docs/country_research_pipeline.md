# Country Research Pipeline

Pipeline standardisee pour analyser un pays avec le meme protocole de recherche, sans imposer une regle de trading uniforme a tous les marches.

## Lancement

```powershell
python scripts/run_country_research_pipeline.py --country sweden
python scripts/run_country_research_pipeline.py --country france
python scripts/run_country_research_pipeline.py --country germany
```

Options utiles :

```powershell
python scripts/run_country_research_pipeline.py --country france --smoke --log-level INFO
python scripts/run_country_research_pipeline.py --country france --start 2018-01-01 --end 2025-12-31
python scripts/run_country_research_pipeline.py --country france --reference-name raw_composite --skip-robustness
```

## Phases

1. Reference locale : selectionne une reference deja presente dans les experiences/configs existantes. Pas de nouvelle optimisation.
2. Diagnostic : enrichit les trades et exporte les decompositions standardisees par regime, entree, qualite de paire, exits et concentration.
3. Hypotheses : genere 2 a 4 hypotheses maximum avec des heuristiques lisibles, a partir des segments destructeurs observes.
4. Ablation : teste peu de variantes autour de la reference locale, sans grille massive.
5. Robustesse : teste les candidats sur des splits temporels contigus.
6. Decision : classe le pays dans une taxonomy comparable entre marches.

## Sorties

Chaque run cree un dossier :

```text
data/experiments/country_research_<country>_<timestamp>/
```

Fichiers principaux :

- `metadata.json`
- `reference_selection.json`
- `hypotheses_generated.json`
- `diagnostic_*.csv`
- `ablation_*.csv`
- `robustness_*.csv`
- `country_research_scorecard.csv`
- `campaign_summary.txt`
- `conclusion.txt`

## Lecture de la decision

La decision finale privilegie la robustesse portfolio-level. Une amelioration trade-level qui ne se traduit pas dans le moteur portfolio est classee comme non confirmee ou risk-control only.

Statuts possibles :

- `promote`
- `promising_needs_validation`
- `risk_control_only`
- `rejected`
- `insufficient_signal`

## Limites

La pipeline standardise la methode, pas les seuils alpha locaux. Les hypotheses generees sont des propositions prudentes, pas une optimisation. Les pays sans reference experimentale existante utilisent une baseline locale documentee si l'univers est disponible.
