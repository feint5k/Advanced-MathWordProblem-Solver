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
- T-RNN and baselines: Contains the code for T-RNN and baseline models.

### Implementation:
The project is implemented in a Python 3.6 or above environment using Pytorch. T-RNN code is replicated and further implementations for Math23K are added. Data replication and raw Dolphin18k data processing have been tackled as well.

## References:
[1] Lei Wang, Dongxiang Zhang, Jipeng Zhang, Xing Xu, Lianli Gao, Bingtian Dai, and Heng Tao Shen. Template-based math word problem solvers with recursive neural networks. 2019.
[2] Yan Wang, Xiaojiang Liu, and Shuming Shi. Deep neural solver for math word problems. In Proceedings of the 2017 Conference on Empirical Methods in Natural  Language Processing, pages 845–854, 2017.