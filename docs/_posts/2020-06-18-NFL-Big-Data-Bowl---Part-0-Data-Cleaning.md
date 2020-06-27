---
layout: post
title: "NFL Big Data Bowl - Part 0: Data Cleaning"
date: 2020-06-18
mathjax: true
github_url: "https://github.com/MughilM/Data-Science-Projects/blob/master/NFL%20NextGen/1%20-%20Data%20Cleaning.ipynb"
---

*Welcome to a project series! The goal for the blog posts is to not flood you with code, but instead to provide a high-level analysis of the steps that were taken for the step. For detailed code that was written, please click on the Github ribbon in the top corner, which each blog post has. The link has the original notebook that the post was based on, complete with all code.*

{% include toc %}

## Introduction

In the following series of blog posts, I will attempt to compete in a past Kaggle challenge called the **NFL Big Data Bowl 2019**. The project page is [here](https://www.kaggle.com/c/nfl-big-data-bowl-2020). The motivation for the challenge comes from a common occurrence where a spectator watching from their couch, sees a handoff for a rushing play takes place, and immediately says to his/herself "Oh that's going all the way", or "That won't work".

The goal here, is to make the same type of decision i.e. predicting how many yards a rushing play will go for. The data we are given are all rushing plays from the start of the 2017 season, up to the first couple weeks of the 2019 season. When this contest was released, the aim was to rank contestants based on how the model performed on *real-time plays* as the season progressed.

For **actual submission**, we give a **cumulative distribution of the probability that a play goes for a loss of 99 yards, a loss of 98 yards, all the way up to a gain of 99 yards.** Now of course, due to play positioning, it's impossible for a rushing play at the 35-yard line to go for a gain/loss of 99 yards, so those would be 1, and 0 respectively.

Since we are submitting late, the performance is slightly different, but the goal is the same.

## The Data

A closer look at the data reveals that we have positional data for all 22 players on the field for each rushing play. In addition, we have physical characteristics for each player such as the weight and height. We also have information about the play, such as the scores before and after the play, yard line, the down and distance, and obviously how many yards to go for.

We have about 49 columns to start with, I'll provide the first 5 rows here. For a description of what each column represents (if it's not immediately clear), please check the Kaggle link.

<div class="table-wrapper" markdown="block" style="overflow-x: scroll;">

| GameId     | PlayId         | Team | X     | Y     | S    | A    | Dis  | Orientation | Dir    | NflId   | DisplayName     | JerseyNumber | Season | YardLine | Quarter | GameClock | PossessionTeam | Down | Distance | FieldPosition | HomeScoreBeforePlay | VisitorScoreBeforePlay | NflIdRusher | OffenseFormation | OffensePersonnel | DefendersInTheBox | DefensePersonnel | PlayDirection | TimeHandoff              | TimeSnap                 | Yards | PlayerHeight | PlayerWeight | PlayerBirthDate | PlayerCollegeName | Position | HomeTeamAbbr | VisitorTeamAbbr | Week | Stadium          | Location       | StadiumType | Turf       | GameWeather    | Temperature | Humidity | WindSpeed | WindDirection |
| ---------- | -------------- | ---- | ----- | ----- | ---- | ---- | ---- | ----------- | ------ | ------- | --------------- | ------------ | ------ | -------- | ------- | --------- | -------------- | ---- | -------- | ------------- | ------------------- | ---------------------- | ----------- | ---------------- | ---------------- | ----------------- | ---------------- | ------------- | ------------------------ | ------------------------ | ----- | ------------ | ------------ | --------------- | ----------------- | -------- | ------------ | --------------- | ---- | ---------------- | -------------- | ----------- | ---------- | -------------- | ----------- | -------- | --------- | ------------- |
| 2017090700 | 20170907000118 | away | 73.91 | 34.84 | 1.69 | 1.13 | 0.4  | 81.99       | 177.18 | 496723  | Eric Berry      | 29           | 2017   | 35       | 1       | 14:14:00  | NE             | 3    | 2        | NE            | 0                   | 0                      | 2543773     | SHOTGUN          | 1 RB, 1 TE, 3 WR | 6                 | 2 DL, 3 LB, 6 DB | left          | 2017-09-08T00:44:06.000Z | 2017-09-08T00:44:05.000Z | **8** | 6-0          | 212          | 12/29/1988      | Tennessee         | SS       | NE           | KC              | 1    | Gillette Stadium | Foxborough, MA | Outdoor     | Field Turf | Clear and warm | 63          | 77       | 8         | SW            |
| 2017090700 | 20170907000118 | away | 74.67 | 32.64 | 0.42 | 1.35 | 0.01 | 27.61       | 198.7  | 2495116 | Allen Bailey    | 97           | 2017   | 35       | 1       | 14:14:00  | NE             | 3    | 2        | NE            | 0                   | 0                      | 2543773     | SHOTGUN          | 1 RB, 1 TE, 3 WR | 6                 | 2 DL, 3 LB, 6 DB | left          | 2017-09-08T00:44:06.000Z | 2017-09-08T00:44:05.000Z | **8** | 6-3          | 288          | 03/25/1989      | Miami             | DE       | NE           | KC              | 1    | Gillette Stadium | Foxborough, MA | Outdoor     | Field Turf | Clear and warm | 63          | 77       | 8         | SW            |
| 2017090700 | 20170907000118 | away | 74    | 33.2  | 1.22 | 0.59 | 0.31 | 3.01        | 202.73 | 2495493 | Justin Houston  | 50           | 2017   | 35       | 1       | 14:14:00  | NE             | 3    | 2        | NE            | 0                   | 0                      | 2543773     | SHOTGUN          | 1 RB, 1 TE, 3 WR | 6                 | 2 DL, 3 LB, 6 DB | left          | 2017-09-08T00:44:06.000Z | 2017-09-08T00:44:05.000Z | **8** | 6-3          | 270          | 01/21/1989      | Georgia           | DE       | NE           | KC              | 1    | Gillette Stadium | Foxborough, MA | Outdoor     | Field Turf | Clear and warm | 63          | 77       | 8         | SW            |
| 2017090700 | 20170907000118 | away | 71.46 | 27.7  | 0.42 | 0.54 | 0.02 | 359.77      | 105.64 | 2506353 | Derrick Johnson | 56           | 2017   | 35       | 1       | 14:14:00  | NE             | 3    | 2        | NE            | 0                   | 0                      | 2543773     | SHOTGUN          | 1 RB, 1 TE, 3 WR | 6                 | 2 DL, 3 LB, 6 DB | left          | 2017-09-08T00:44:06.000Z | 2017-09-08T00:44:05.000Z | **8** | 6-3          | 245          | 11/22/1982      | Texas             | ILB      | NE           | KC              | 1    | Gillette Stadium | Foxborough, MA | Outdoor     | Field Turf | Clear and warm | 63          | 77       | 8         | SW            |
| 2017090700 | 20170907000118 | away | 69.32 | 35.42 | 1.82 | 2.43 | 0.16 | 12.63       | 164.31 | 2530794 | Ron Parker      | 38           | 2017   | 35       | 1       | 14:14:00  | NE             | 3    | 2        | NE            | 0                   | 0                      | 2543773     | SHOTGUN          | 1 RB, 1 TE, 3 WR | 6                 | 2 DL, 3 LB, 6 DB | left          | 2017-09-08T00:44:06.000Z | 2017-09-08T00:44:05.000Z | **8** | 6-0          | 206          | 08/17/1987      | Newberry          | FS       | NE           | KC              | 1    | Gillette Stadium | Foxborough, MA | Outdoor     | Field Turf | Clear and warm | 63          | 77       | 8         | SW            |

</div>

The "Yards" column (in bold) is the one we're most interested in, and is considered our target. Before we do any data analysis however, it is necessary to do some exploratory data analysis (EDA). However, *even before that*, we should clean the data. Since the data comes from Kaggle, we can expect some level of cleanliness. Still, to make future analyses easier to perform, it can be helpful to transform the columns.

As always, check the Github link in the top corner to see the full code. Each transform procedure has been given its own section.

## Transformations

### Missing Data

Due to the weather columns (temperature, humidity, etc.) containing lots of missing data, the columns were straight up dropped. For the rest of the columns, there were small pockets of missing data, so those rows were dropped. The data initially had 682k rows, and after dropping, it had 673k rows.

### Data Types

Columns such as IDs and team abbreviations need to be categorical, instead of strings. The transformations were applied to make future operations easier. When we save the transformed data as a csv again, the data types unfortunately get wiped, so this was done purely for ease of use.

### Game Clock

Currently, the game clock column shows the *remaining time* left in the quarter as 14:14:00. However, the majority of file readings read this as a direct time i.e. 2:14 PM (Excel does this). To preserve the actual meaning of the column, **the game clock column was split into minutes and seconds remaining.** Moreover, since possession *does not* reset at the end of the half, the two columns show **time remaining in the half, not quarter.**

### Player Height

Currently the height is listed as "6-3" for 6 feet 3 inches. Because height is inherently a number, this was transformed to a floating point value in units of foot i.e. "6-3" = 6.25 ft.

### Offensive Personnel

This column lists how many of each type of player is present in the play. The default is 1 RB, 1 TE, 3 WR (and consequently 5 OL, and 1 QB). Teams call this "11 Personnel". However, for short yardage situations, like 3rd or 4th down and 1, teams will most likely put beefier men, such as an extra linemen or tight end in an effort to get the first down through brute strength. This leads to the row reading something like "6 OL, 1 RB, etc." Because the column doesn't have a standard, I transformed this column to have columns for the number of **running backs (RB)**, **tight ends (TE)**, **wide receivers (WR)**, **offensive linemen (OL)**, **quarterbacks (QB)**, and **any defensive players.** If you're curious about the last one, there have been plenty of instances of defensive players playing on offense (JJ Watt has a touchdown **catch** to his name by the way).

### Defensive Personnel

We follow the same logic as the offensive personnel, with **defensive linemen (DL)**, **linebackers (LB)**, and **cornerbacks (CB)**. As usual, we include offensive players on defense as well.

### Splitting of Data

In case you haven't noticed, the data has a row for *each player in each play*. However, the vast majority of columns have nothing to do with player himself, like the down, distance, yards, etc. We have about 30k plays in the data, so separating off the columns which have the same data for each player in a play and compressing it down will result in more efficient storage. For separation, we generate 3 separate tables: one for player physical characteristics (weight, height, college, etc.), one for player physical positions in each play (our original data, with most columns removed), and one for plays.

## Results

At the end of all these transformations, we produce three different files: the positional data for each player in each play, the physical characteristics of each player that has taken part in a play, and finally data about each play itself that is independent of the players.

### Positional Data

This data frame is essentially the first couple columns of our original dataset.

<div class="table-wrapper" markdown="block" style="overflow-x: scroll;">

| GameId     | PlayId         | Team | X     | Y     | S    | A    | Dis  | Orientation | Dir    | NflId   |
| ---------- | -------------- | ---- | ----- | ----- | ---- | ---- | ---- | ----------- | ------ | ------- |
| 2017090700 | 20170907000118 | away | 73.91 | 34.84 | 1.69 | 1.13 | 0.4  | 81.99       | 177.18 | 496723  |
| 2017090700 | 20170907000118 | away | 74.67 | 32.64 | 0.42 | 1.35 | 0.01 | 27.61       | 198.7  | 2495116 |
| 2017090700 | 20170907000118 | away | 74    | 33.2  | 1.22 | 0.59 | 0.31 | 3.01        | 202.73 | 2495493 |
| 2017090700 | 20170907000118 | away | 71.46 | 27.7  | 0.42 | 0.54 | 0.02 | 359.77      | 105.64 | 2506353 |
| 2017090700 | 20170907000118 | away | 69.32 | 35.42 | 1.82 | 2.43 | 0.16 | 12.63       | 164.31 | 2530794 |

</div>

The number of rows in this data is the same as the original, approximately 672k. However, instead of close to 50 columns, we only have to deal with 11.

### Player Physical Characteristics

Another data frame we have is a table of the physical characteristics of each player, as well as their positions i.e. running backs, offensive linemen, etc.

<div class="table-wrapper" markdown="block" style="overflow-x: scroll;">

| NflId   | DisplayName     | PlayerHeight | PlayerWeight | PlayerBirthDate | PlayerCollegeName | Position |
| ------- | --------------- | ------------ | ------------ | --------------- | ----------------- | -------- |
| 496723  | Eric Berry      | 6            | 212          | 12/29/1988      | Tennessee         | SS       |
| 2495116 | Allen Bailey    | 6.25         | 288          | 3/25/1989       | Miami             | DE       |
| 2495493 | Justin Houston  | 6.25         | 270          | 1/21/1989       | Georgia           | DE       |
| 2506353 | Derrick Johnson | 6.25         | 245          | 11/22/1982      | Texas             | ILB      |
| 2530794 | Ron Parker      | 6            | 206          | 8/17/1987       | Newberry          | FS       |

</div>

Notice we didn't include the team, because players can change teams every year (even during the year). It's quite possible that when a player goes to a different team and is put under a different head coach/coordinator, his performance skyrockets. However, instances like that are a tad difficult to pinpoint, and so we will be looking at those in the far future.

### Play by Play Data

Finally, we have data that only changes with the play, such as offense and defense formations and the down and distance.

<div class="table-wrapper" markdown="block" style="overflow-x: scroll;">

| GameId     | PlayId         | Season | YardLine | Quarter | Possession | Down | Distance | FieldPosition | HomeScoreBeforePlay | VisitorScoreBeforePlay | NflIdRusher | OffenseFormation | DefendersInTheBox | PlayDirection | TimeHandoff               | TimeSnap                  | Yards | HomeTeamAbbr | VisitorTeamAbbr | Week | Stadium          | Location       | Turf       | GameClockMinute | GameClockSecond | Half | RB  | TE  | WR  | OL  | QB  | DoO | DL  | LB  | DB  | OoD |
| ---------- | -------------- | ------ | -------- | ------- | ---------- | ---- | -------- | ------------- | ------------------- | ---------------------- | ----------- | ---------------- | ----------------- | ------------- | ------------------------- | ------------------------- | ----- | ------------ | --------------- | ---- | ---------------- | -------------- | ---------- | --------------- | --------------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017090700 | 20170907000118 | 2017   | 35       | 1       | NE         | 3    | 2        | NE            | 0                   | 0                      | 2543773     | SHOTGUN          | 6                 | left          | 2017-09-08 00:44:06+00:00 | 2017-09-08 00:44:05+00:00 | 8     | NE           | KC              | 1    | Gillette Stadium | Foxborough, MA | Field Turf | 29              | 14              | 1    | 1   | 1   | 3   | 5   | 1   | 0   | 2   | 3   | 6   | 0   |
| 2017090700 | 20170907000139 | 2017   | 43       | 1       | NE         | 1    | 10       | NE            | 0                   | 0                      | 2543773     | SHOTGUN          | 6                 | left          | 2017-09-08 00:44:27+00:00 | 2017-09-08 00:44:26+00:00 | 3     | NE           | KC              | 1    | Gillette Stadium | Foxborough, MA | Field Turf | 28              | 52              | 1    | 1   | 1   | 3   | 5   | 1   | 0   | 2   | 3   | 6   | 0   |
| 2017090700 | 20170907001355 | 2017   | 35       | 1       | NE         | 1    | 10       | KC            | 0                   | 0                      | 2543773     | SINGLEBACK       | 7                 | left          | 2017-09-08 00:45:17+00:00 | 2017-09-08 00:45:15+00:00 | 5     | NE           | KC              | 1    | Gillette Stadium | Foxborough, MA | Field Turf | 28              | 2               | 1    | 1   | 1   | 3   | 5   | 1   | 0   | 2   | 3   | 6   | 0   |
| 2017090700 | 20170907000345 | 2017   | 2        | 1       | NE         | 2    | 2        | KC            | 0                   | 0                      | 2539663     | JUMBO            | 9                 | left          | 2017-09-08 00:48:41+00:00 | 2017-09-08 00:48:39+00:00 | 2     | NE           | KC              | 1    | Gillette Stadium | Foxborough, MA | Field Turf | 27              | 12              | 1    | 2   | 2   | 0   | 6   | 1   | 0   | 4   | 4   | 3   | 0   |
| 2017090700 | 20170907000395 | 2017   | 25       | 1       | KC         | 1    | 10       | KC            | 7                   | 0                      | 2557917     | SHOTGUN          | 7                 | right         | 2017-09-08 00:53:14+00:00 | 2017-09-08 00:53:13+00:00 | 7     | NE           | KC              | 1    | Gillette Stadium | Foxborough, MA | Field Turf | 27              | 8               | 1    | 1   | 3   | 1   | 5   | 1   | 0   | 3   | 2   | 6   | 0   |

</div>

Notice that we now have one row per play. You can see that the 5th row is a Kansas City rush, on the next drive. Also notice that the 4th row was a 2 yard run from the 2 yard line, meaning it was actually a touchdown run. We are only given the scores before the play occurred. In total, this data frame only has about 30k rows, much less than the original 670k+ rows.

Thus, with these three data frames, we are ready to do exploratory data analysis. See you next time!