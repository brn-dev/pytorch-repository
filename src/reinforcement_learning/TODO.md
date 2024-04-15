
return dist entropy in select action -> add as objective to reduce entropy

reward/return/advantage normalization

When negative advantage has been observed, lower learning rate for critic. This keeps the expectation higher.
or
Adjust learning rate based on performance, discourage repeating actions that lead to a shitty run.

Warning when an environment doesn't terminate within the buffer range 

torch jit trace policy