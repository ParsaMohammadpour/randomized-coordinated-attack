# Coordinated Attack Randomized Algorithm
&emsp;  In this assignment I implemented the randomized algorithm for coordinated attack problem. This algorithm trys to solve the consensus problem for link failure. This algorithm work when:
##### - each process knows entire graph(completed graph)
##### - communication graph is undirected
##### - any node decision is zero or one
although a **strongly-connected** network with r greater than or equal to the diameter also works.((r, is number of rounds that algorithm has))
<br/>
<br/>
Three main requirements are as follows:
<br/>
### safety:
#### - agreement:
&emsp; no two process decide on different values
#### - Validation;
&emsp;  If all processes start with zero, then zero is the only possible decision.
<br/>
&emsp;  If all processes start with one, and we had no failures, then one is the only possible decision.
### liveness:
#### - Termination: 
&emsp;  This requirement specifies  that all processes should eventually decide.
<br/>
<br/>
We should note that no deterministic consensus protocol provides all three of safety, liveness,
and fault tolerance in an asynchronous system.
<br/>
<br/>
<br/>
<br/>
## algorithm
### description
&emsp; In this algorithm, the leader randomly select a key between [1, r] , where r is the number of rounds that algorithm has.(for simplification, we consider process number 1 as the leader in which generates the key).
<br/>
Each process stores two vectors of size n, related to each process input and information level.
<br/>
every process sends it's entire states at each round to every other process.
<br/>
Each process information level, can be computed by finding the min value of the information level vector +1. Each process upon receiving the information level from another process, updates its own information level by setting the max of its current information level of process i (index i of information level vector) and the incoming information level from the sender process. After it had received all messages and updates the information level vector, the process updates its own information level by the mentioned function. (find min value of its own information level vector and add this value with 1).
<br/>
<br/>
### decision rule
&emsp; If we were at rth round, then each process has to decide by the following rule:
if the key was not undefined, and level (information level) was greater than or equal to the key, and all values of all processes was one, then process decides one. Otherwise, process decides zero.