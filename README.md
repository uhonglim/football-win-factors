# 什么指标对赢球最关键？— 2022/23 英超 · 西甲 · 欧冠

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uhonglim/football-win-factors/blob/main/analysis.ipynb)

🌐 **网站**：https://uhonglim.github.io/football-win-factors/

## In plain English

We looked at 767 matches from the 2022/23 Premier League, La Liga and Champions League, and compared 20 stats between the two teams in each game. The stat that matters most is shots on target: a team with 3 more than its opponent wins about 60 to 70% of the time. Having more of the ball is a sign of a strong team, but it doesn't decide a single match.

**How to win:** take better shots, use possession to create chances, avoid red cards, and make home games count. These are patterns in the data, not guaranteed causes.

## 核心结论

1. **射正是赢球第一指标**：单场相关 0.46–0.60，赛季相关 0.82；射正比对手多 3 次，胜率 63%–74%。
2. **射正 > 射门**：射门次数的相关（≈0.3）远低于射正（≈0.5）。
3. **控球是"强队标志"，不是"赢球按钮"**：单场相关只有 0.12，赛季层面达到 0.68。
4. **解围、长传、传中要反着读**：它们更多反映比分状态（领先收缩 / 落后追分）。
5. **红牌**在三项赛事里都是负面因素；主场优势约 19–20 个百分点。

## 文件

| 文件 | 说明 |
|---|---|
| `analysis.ipynb` | Colab 分析 notebook（可直接在 Colab 运行） |
| `fetch_data.py` | 从 ESPN 公开接口抓取 2022/23 比赛技术统计 |
| `data/matches_2223.csv` | 抓取结果，每场比赛两行（主/客视角） |
| `data/fd_*_2223.csv` | football-data.co.uk 英超/西甲全量数据，用于稳健性检验 |
| `docs/` | GitHub Pages 网站（`index.html` + notebook 导出的 `results.json`） |
| `build_notebook.py` | 生成 notebook 的脚本 |

## 本地运行

```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests
python fetch_data.py            # 可选：重新抓取数据
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb   # 会更新 docs/results.json
python -m http.server -d docs   # 预览网站
```

数据来源：ESPN、football-data.co.uk。仅供学习研究。
