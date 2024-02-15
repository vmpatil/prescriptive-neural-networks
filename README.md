# Prescriptive_Neural_Networks
An associated Github repo for our paper 'Applications of 0-1 Neural Networks in Prescription and Prediction'

Authors: Vrishabh Patil (vmpatil@wisc.edu), Yonatan Mintz (ymintz@wisc.edu)

**Under Review**

### Abstract ###

A key challenge in medical decision making is learning treatment policies for patients with limited data. This challenge is particularly evident in personalized healthcare decision-making, where models need to take into account the intricate relationships between patient characteristics, treatment options, and health outcomes. To address this, we introduce prescriptive networks (PNNs), shallow 0-1 neural networks trained with mixed integer programming that can be used with counterfactual estimation to optimize policies in medium data settings. These models offer greater interpretability than deep neural networks and can encode more complex policies than common models such as decision trees. We show that PNNs can outperform existing methods in both synthetic data experiments and in a case study of assigning treatments for postpartum hypertension. In particular, PNNs are shown to produce policies that could reduce peak blood pressure by 5.47 mm Hg over existing clinical practice, and by 2 mm Hg over the next best prescriptive modeling technique. Moreover PNNs were more likely then all other models to correctly identify clinically significant features while existing models relied on potentially dangerous features such as patient insurance information and race that could lead to bias in treatment.

### Code and Experiments

```
hypertension
  experiments associated with treatment assignment problem for post-partum hypertension patients. directory includes python scripts and data for the prescriptive tree models, causal forest model, and prescriptive neural network.

synthetic
  experiments associated with simulated data as discussed in [1]. directory includes python scripts and data for the prescriptive tree models, causal forest model, and prescriptive neural network.

warfarin
  experiments associated with learning personalized warfarin dosing [2]. directory includes python scripts and data for the prescriptive tree models, causal forest model, and prescriptive neural network.
```
[1] Athey, S., Imbens, G.: Recursive partitioning for heterogeneous causal effects. Proceedings of the National Academy of Sciences 113(27), 7353–7360 (2016)

[2] Consortium, I.W.P.: Estimation of the warfarin dose with clinical and pharmacogenetic data. NEJM 360(8), 753–764 (2009)
