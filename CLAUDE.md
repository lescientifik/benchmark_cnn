## Stack technique

- **Gestionnaire de projet** : `uv` (exclusivement)
  - Installation dépendances : `uv add`
  - Dev dependencies : `uv add --dev`
  - Exécution scripts : `uv run`
  - Sync environnement : `uv sync`

### Commandes uv essentielles

```bash
uv sync                          # Synchroniser l'environnement
uv add <package>                 # Ajouter une dépendance
uv add --dev <package>           # Ajouter une dépendance de dev
uv add -e ./path/to/local        # Ajouter un package local en mode editable
uv run pytest                    # Lancer les tests
uv run pytest -v -k "test_name"  # Test spécifique
uv run ruff check .              # Linting
uv run TotalSegmentator --help   # CLI principal
```

**IMPORTANT**: 
* Ne JAMAIS utiliser `pip` ou `uv pip`. Toujours utiliser `uv add` pour les dépendances.
* ne fais pas de script que tu écris et éxécutes directement avec ton outil Bash, si le script fait plusieurs lignes, écris le dans un fichier, comme ça s'il ne marche pas tu peux
 le corriger facilement sans tout réécrire

**Note**: `uv run` exécute directement les scripts Python (pas besoin de `uv run python script.py`, juste `uv run script.py`).

