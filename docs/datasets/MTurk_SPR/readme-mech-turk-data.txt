Variables for Collecting Response Time Data Using Amazon Mechanical Turk

Filename: dataExp1.csv
Description: Experiment 1
34 adults
48 test items, all ungrammatical wh-questions
72 fillers
Presented in randomized order

Variables:

Count - counter  
X.1 - counter
X - counter
fullItem - full item code before it is split
response - participant response (only for comprehension questions, 0 = incorrect, 1 = correct)
RT - response time, in ms
participant - participant ID number
item - item number
condition - test condition (CondA = singular subject, singular wh, DP; CondB = singular subject, singular wh, pronoun; CondC = plural subject, singular wh, DP; CondD = plural subject, singular wh, pronoun; CondE = singular subject, plural wh, DP; CondF = singular subject, plural wh, pronoun; CondG = plural subject, plural wh, DP; CondF = plural subject, plural wh, pronoun)
region
Pronoun - no (0) or yes (1)
NumberWh - singular (0) or plural (1)
NumberSu - singular (0) or plural (1)
remove - used to remove items with incorrect comprehension questions
word - word read by participant
regionTime - response time aggregated by region
wordLength - number of characters in word
regionLength - number of characters in region
logRT - log of region time
exp - expected reading time
resid - residual reading time


Filename: dataExp2.csv
Description: Experiment 2
82 participants
16 test items, half ungrammatical
96 fillers

Variables:
Count - counter
X - counter	
fullItem - full item code before it is split	
response - participant response (only for comprehension questions, 0 = incorrect, 1 = correct)	
RT - response time, in ms	
participant - participant ID number	
item - item number	
condition - test condition (CondA = singular attractor, grammatical; CondB = plural attractor, grammatical; CondC = singular attractor, ungrammatical; CondD = singular attractor, ungrammatical)	
word - word read by participant	
region	- Sentence region in numerical order
ungrammatical - no (0) or yes (1)	
mismatch - number mismatch between subject and attractor, no (0) or yes (1)	
remove	- used to remove items with incorrect comprehension questions
wordLength - number of characters in word 	
logRT - log of response time	
exp - expected reading time	
resid - residual reading time

Filename: dataExp3.csv
Description: Experiment 3
60 participants
24 test items, half ungrammatical
72 fillers

Count - counter
X - counter	
fullItem - full item code before it is split	
response - participant response (only for comprehension questions, 0 = incorrect, 1 = correct)	
RT - response time, in ms	
participant - participant ID number	
item - item number	
condition - test condition (CondA = singular attractor, grammatical; CondB = plural attractor, grammatical; CondC = singular attractor, ungrammatical; CondD = singular attractor, ungrammatical)	
word - word read by participant	
region	- sentence region in numerical order	
ungrammatical - no (0) or yes (1)
mismatch - number mismatch between subject and attractor, no (0) or yes (1)	
wordLength - number of characters in word 	

