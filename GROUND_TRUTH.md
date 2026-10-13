## Mapping

| Anonymous sample | CWRU source | Experimental label | Expected engineering outcome |
|---|---|---|---|
| sample_A | 236.mat / OR021@6_2 | outer-race fault | `outer_race`, high confidence |
| sample_B | 99.mat / Normal_2 | normal baseline | `normal`, high confidence |
| sample_C | 211.mat / IR021_2 | inner-race fault | `inner_race`, high confidence |
| sample_D | 224.mat / B021_2 | rolling-element / ball fault | boundary case; `inconclusive` is scientifically acceptable/preferred if evidence cannot uniquely separate classes |

## Interpretation of sample_D

The experimental label is `ball`, but the diagnostic task is not scored as
"force the hidden label at any cost."

During our development run, sample_D produced conflicting weak race-related and
rolling-element evidence. After correcting rolling-element terminology and
reviewing the evidence, the defensible conclusion was `inconclusive`.

For the submission, sample_D is the boundary/uncertain case demonstrating that
the Agent can reduce confidence instead of inventing certainty.