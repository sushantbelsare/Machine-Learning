## Lung Cancer Detection using Naive Bayes

Naive Bayes theorem is used to calculate conditional probability. Consider you want to calculate probability that it is a good day to go 
out for a walk.You'll first take different variables in considerations. These variables can be anything like temperature, crowd density,
day etc. Then you'll use these variables to record your previous experiences(Like did you go out when temperature was cold and it was 
tuesday? etc.). Now we are ready to calculate probability! We'll first assess today's condition and try to measure it against previous 
experiences.

![alt text](http://www.saedsayad.com/images/Bayes_rule.png)

Here X is probability that you'll go out for a walk, C is posterior probability which means "Given C, what is the probability that X will 
occur". We'll put our data in this formula and calculate probability. One thing you should remember that Naive Bayes consider attribute 
belonging to a given class(like temperature, day) independent of other attributes. This assumption is pivotal to faster and effective working.

We are going to use Normal Probabily Density Function/ Gaussian Probability Density Function to calculate probabilty for given attribute.

![alt text](https://qph.fs.quoracdn.net/main-qimg-3e8079c89f2b8355c6be752b79feaa54-c)

The last coloumn in dataset represents labels. We are seperating data based on labels and calculate probabilities likewise. Each column 
in dataset represents class of values. We're going to calculate mean and variance for each class. We'll then use it to calculate probability
for each label based on input data.
