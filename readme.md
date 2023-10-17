## Introduction to Advanced Math Word Problem Solver

This software uses Natural Language Processing to solve Math word problems. Our approach forms an English corpus of arithmetic word problems, creates equation templates, performs normalizing equations, and experimentally evaluates T-RNN and retrieval baselines.

### Contributions
- Creation of 'Dolphin300', a unique English corpus of arithmetic word problems.
- Development of equation templates and normalizing equations on par with the Math23K dataset [1].
- Evaluation of T-RNN and retrieval baselines on Math23K, Dolphin300 and Dolphin1500.

### Data Processing Examples:
Our software transforms sentences into concise math problems. For instance:
- What is the value of five times the sum of twice of three-fourths and nine?
- help!!!!!!!(please) i can't figure this out! what is the sum of 4 2/5 and 17 3/7 ?

### Folder Structure:
- Web scraping: Contains code to scrap and clean math word problems from the Internet.
- Data_Cleaning: Houses the code for data cleaning including the transformation logic.
- T-RNN