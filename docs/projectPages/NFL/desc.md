---
layout: page
permalink: /description/
title: The NFL
---

## About

The National Football Leauge (NFL) is an American football league based in the United States. It consists of two conferences, the NFC and AFC, with 4 divisions of 4 teams each. The regular season consists of 16 games for each of the 32 teams, totaling 256 games. Describing the extensive rules for this game goes slightly beyond the scope of this page. Interested readers should read the Wikipedia page [here](https://en.wikipedia.org/wiki/American_football_rules) for all the rules.

## About the Data Used

The data I have used in all NFL projects are play-by-play data. Play-by-play data has been available since the 2009 regular season through NFL's Game Center Archives. However, the data currently is in JSON format, which on its own is difficult to convert to readable .csv files. Thankfully, Github user `maksimhorowitz` has created an R package to scrape this data. Additionally, user `ryurko` has a repository that already has play-by-play data in .csv format [here](https://github.com/ryurko/nflscrapR-data). This data has a vast number of columns, too many describe one by one. The columns describe every possible aspect of the play outside of actual player positions. For example, everything from filed goal length, two point conversion attempts, tackles and tackle assists, to quarterback scrambles and hits are described in this table. The full breadth of this data is currently seen being utilized in the notebooks below.

## Notebooks in this Project

[Exploratory Data Analysis](EDA.md) 