---
layout: page
permalink: /description/
title: The NFL
---

## About

The National Football Leauge (NFL) is an American football league based in the United States. It consists of two conferences, the NFC and AFC, with 4 divisions of 4 teams each. The regular season consists of 16 games for each of the 32 teams, totaling 256 games. Describing the extensive rules for this game goes slightly beyond the scope of this page. Interested readers should read the Wikipedia page [here](https://en.wikipedia.org/wiki/American_football_rules) for all the rules.

## About the Data Used

The data I have used in all NFL projects are play-by-play data. Play-by-play data has been available since the 2009 regular season through NFL's Game Center Archives. However, the data currently is in JSON format, which on its own is difficult to convert to readable .csv files. Thankfully, Github user maksimhorowitz has created an R package to scrape this data. Additionally, user ryurko has a repository that already has play-by-play data in .csv format [here](https://github.com/ryurko/nflscrapR-data). This data has a vast number of columns, which I have described below:

| Variable Name             | Description                                      | Data Type/Values                                 |
| ------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| play_id                   | ID of the play                                   | int                                              |
| game_id                   | ID of the game                                   | int                                              |
| home_team                 | The home team                                    | 3 letter team code                               |
| away_team                 | The away team                                    | 3 letter team code                               |
| posteam                   | Team with possession                             | 3 letter team code                               |
| posteam_type              | Whether posteam is home or away                  | home/away                                        |
| defteam                   | Team on defense                                  | 3 letter team code                               |
| side_of_field             | Which side of field the ball is on               | 3 letter team code                               |
| yardline_100              | Yardline given in terms of 100                   | int                                              |
| game_date                 | Date of the game                                 | date                                             |
| quarter_seconds_remaining | \# of seconds remaining in the quarter           | int                                              |
| half_seconds_remaining    | \# of seconds remaining in the half              | int                                              |
| game_seconds_remaining    | \# of seconds remaining in the game              | int                                              |
| game_half                 | Which half of the game                           | Half1/Half2                                      |
| quarter_end               | If the quarter ended on this play                | 0/1                                              |
| drive                     | drive \#                                         | int                                              |
| sp                        |                                                  | 0/1                                              |
| qtr                       | quarter \#                                       | 0/1                                              |
| down                      | down #                                           | int, N/A if downs do not apply                   |
| goal_to_go                |                                                  | 0/1                                              |
| time                      | Time on the clock                                | time                                             |
| yardln                    | Yardline as reported by NFL                      | 3 letter team code followed by int i.e. "ATL 25" |
| ydstogo                   | Yards to go on the down at the start of the play | int                                              |
| ydsnet                    | Net yards gained on the drive                    | int                                              |
| desc                      | Description of the play                          | string                                           |
| play_type                 | Type of play                                     | run/pass/kickoff/no_play/field_goal              |
| yards_gained              | Yards gained on the play                         | int                                              |
| shotgun                   | If the play was in shotgun formation             | 0/1                                              |
| no_huddle                 | If the play was done without a huddle            | 0/1                                              |
| qb_dropback               | If the QB dropped back                           | 0/1                                              |
| qb_kneel                  | If the QB kneeled on the play                    | 0/1                                              |
| qb_scramble               | If the QB scrambled on the play                  | 0/1                                              |
| pass_length               | If the pass was a short or deep pass             | short/deep, or nan if there wasn't a pass        |
| pass_location             | Location of the pass                             | left/right, or nan if no pass                    |
| air_yards                 | Yards the ball traveled through the air          | `int`, or `nan` if no pass                       |
| yards_after_catch         | Yards the receiver ran after catching the ball   | `int`, or `nan` if no pass                       |
| run_location              | Direction of the run                             | `left`/`right` or `nan` if no run                |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |
|                           |                                                  |                                                  |

## Notebooks in this Project

[Exploratory Data Analysis](EDA.md) 