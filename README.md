# What wins football matches? Premier League, La Liga and Champions League, 2022/23

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uhonglim/football-win-factors/blob/main/analysis.ipynb)

**Website:** https://uhonglim.github.io/football-win-factors/

We compared 18 match stats across 767 games from the 2022/23 Premier League, La Liga and Champions League. The stat that matters most is shots on target: teams with more than their opponent won 63% of matches, and teams with three more won two in three. Teams with more of the ball won only 43%. Possession marks a strong team over a season, but it doesn't decide a single match.

## Findings

1. **Shots on target matters most.** Correlation with winning is 0.46 to 0.60 per match and 0.82 per season, and its model importance is 2 to 4 times the next stat.
2. **Shots on target beat shots.** Total shots correlates around 0.3 with winning, shots on target around 0.5.
3. **Possession marks a strong team but doesn't decide a match.** Per match the correlation is 0.12; per season it is 0.68. Teams that lead often give up the ball.
4. **Some stats follow the score.** Many clearances and long balls usually mean protecting a lead; many crosses usually mean chasing the game.
5. **Red cards hurt** in all three competitions. **Home advantage** is about 20 percentage points.
6. **Style changes how teams get their shots, not what wins.** Grouping the 40 league teams by how they play gives four styles: control the ball (Man City, Barcelona, Real Madrid, Arsenal), balanced, defend deep, and direct and physical (Getafe, Mallorca). Ball-control teams average 2.0 points per game and direct teams 1.1, but in every style teams won far more often with more shots on target (53% to 72%) than with more of the ball (20% to 61%). Once shots-on-target difference is known, style adds nothing (p = 0.89).

**How to win:** take better shots, use the ball to create chances, keep eleven players on the pitch, and make home games count. These are patterns in the data, not guaranteed causes.

## Files

| File | What it is |
|---|---|
| `analysis.ipynb` | The analysis notebook (runs in Colab) |
| `fetch_data.py` | Downloads 2022/23 match stats from ESPN's public API |
| `data/matches_2223.csv` | The downloaded data, two rows per match (one per team) |
| `data/fd_*_2223.csv` | football-data.co.uk Premier League and La Liga data, used as a cross-check |
| `docs/` | The website (`index.html` and `results.json` exported by the notebook) |
| `build_notebook.py` | Script that generates the notebook |

## Run it locally

```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests jupyter
python fetch_data.py            # optional: download the data again
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb   # also updates docs/results.json
python -m http.server -d docs   # preview the website
```

Data: ESPN and football-data.co.uk. For learning and research.
