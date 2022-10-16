# IROS 2022 Safe Robot Learning Competition (Submission)

## `Num_episodes` Used in Each Level

1 episode is enough for evaluation.

But if set to 2 or more the planner will switch to a more conservative strategy after failure.

## Command Line for Evaluation

*cd competition/*

*python getting_started.py --overrides level0.yaml*

*python getting_started.py --overrides level1.yaml*

*python getting_started.py --overrides level2.yaml*

*python getting_started.py --overrides level3.yaml*

## Methods Used

1. A sampling-based intermediate path point inserting policy
2. Minimun snap trajectory generation
3. LSTM (Long short-term memory) network for quartotor real mass estimation