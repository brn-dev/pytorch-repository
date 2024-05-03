
return dist entropy in select action -> add as objective to reduce entropy


When negative advantage has been observed, lower learning rate for critic. This keeps the expectation higher.
or
Adjust learning rate based on performance, discourage repeating actions that lead to a shitty run.

torch jit trace policy

ppo :
* fetch batch from buffer 
* use factor of old/new value estimate instead of static range
* early stopping with KL divergence
* summary info instead of last 
* save some high return action-sequences even after resetting the rollout buffer


Genetic Policy Selection:
1. create N random policies
2. train them independently for set amount of time
3. take n best policies (according to some metric)
4. create copies of them as to have N policies again 
5. optional: add noise to each policies weight -> "mutation"
6. repeat
=> Can be implemented using model db

compute score/return function without GAE
