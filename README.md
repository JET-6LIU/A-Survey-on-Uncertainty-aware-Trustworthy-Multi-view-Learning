# A Survey on Uncertainty-aware Trustworthy Multi-view Learning
Repository for the survey "Uncertainty-aware Trustworthy Multi-view Learning", including links to referenced literature.

## Table of Contents

- [Uncertainty Estimation Methods for Deep Learning](#uncertainty-estimation-methods-for-deep-learning)
  - [Uncertainty Estimation Methods](#uncertainty-estimation-methods)
    - [Single Deterministic Methods](#single-deterministic-methods)
    - [Bayesian Methods](#bayesian-methods)
    - [Ensemble Methods](#ensemble-methods)
    - [Post-hoc Methods](#post-hoc-methods)
  - [EDL in Multi-view Learning](#edl-in-multi-view-learning)
- [Methodological Improvement](#methodological-improvement)
  - [Evidence Collection](#evidence-collection)
  - [Uncertainty Estimation](#uncertainty-estimation)
  - [Opinion Aggregation](#opinion-aggregation)
- [Challenging Scenarios](#challenging-scenarios)
  - [Conflicting Views](#conflicting-views)
  - [Incomplete Views](#incomplete-views)
  - [Imbalanced Problems](#imbalanced-problems)
  - [Adversarial Attacks](#adversarial-attacks)
  - [Noisy Labels](#noisy-labels)
  - [Open-set Scenarios](#open-set-scenarios)
- [Beyond Classification](#beyond-classification)
  - [Semi-supervised Learning](#semi-supervised-learning)
  - [Zero-shot Learning](#zero-shot-learning)
  - [Prompt Learning](#prompt-learning)
  - [Representation Learning](#representation-learning)
  - [Clustering](#clustering)
 - [Application](#Application)
   - [Medical Science](#Medical Science)
   - [Medical Image Segmentation](#Medical Image Segmentation)
   - [Other Specific Tasks](#Other Specific Tasks)
  - [Others](#Others]
    - [Fake News Detection](#Fake News Detection]
    - [Sentiment Analysis](#Sentiment Analysis]
    - [Autonomous Driving](#Autonomous Driving]

## Uncertainty Estimation Methods for Deep Learning

### Uncertainty Estimation Methods

#### Single Deterministic Methods

<table>
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th>Year</th>
      <th>Paper</th>
      <th>Venue</th>
      <th>Code</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td rowspan="11">Internal Methods</td>
      <td rowspan="4">Prior Networks</td>
      <td>2018</td>
      <td><a href="https://proceedings.neurips.cc/paper/2018/file/3ea2db50e62ceefceaf70a9d9a56a6f4-Paper.pdf">Predictive uncertainty estimation via prior networks</a></td>
      <td>NeurIPS</td>
      <td><a href="https://github.com/KaosEngineer/PriorNetworks-OLD">Code</a></td>
    </tr>
    <tr>
      <td>2019</td>
      <td><a href="https://proceedings.neurips.cc/paper/2019/file/7dd2ae7db7d18ee7c9425e38df1af5e2-Paper.pdf">Reverse kl-divergence training of prior networks: Improved uncertainty and adversarial robustness</a></td>
      <td>NeurIPS</td>
      <td><a href="https://github.com/KaosEngineer/PriorNetworks">Code</a></td>
    </tr>
    <tr>
      <td>2020</td>
      <td><a href="https://proceedings.neurips.cc/paper_files/paper/2020/file/68d3743587f71fbaa5062152985aff40-Paper.pdf">Towards maximizing the representation gap between in-domain & out-of-distribution examples</a></td>
      <td>NeurIPS</td>
      <td><a href="https://github.com/jayjaynandy/maximize-representation-gap">Code</a></td>
    </tr>
    <tr>
      <td>2021</td>
      <td><a href="https://ieeexplore.ieee.org/document/9414153">Failure prediction by confidence estimation of uncertainty-aware Dirichlet networks</a></td>
      <td>ICASSP</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td rowspan="5">Posterior-based</td>
      <td>2018</td>
      <td><a href="https://proceedings.neurips.cc/paper/2018/file/a981f2b708044d6fb4a71a1463242520-Paper.pdf">Evidential deep learning to quantify classification uncertainty</a></td>
      <td>NeurIPS</td>
      <td><a href="https://muratsensoy.github.io/uncertainty.html">Code</a></td>
    </tr>
    <tr>
      <td>2020</td>
      <td><a href="https://proceedings.neurips.cc/paper_files/paper/2020/file/0eac690d7059a8de4b48e90f14510391-Paper.pdf">Posterior network: Uncertainty estimation without ood samples via density-based pseudo-counts</a></td>
      <td>NeurIPS</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2020</td>
      <td><a href="https://proceedings.neurips.cc/paper_files/paper/2020/file/aab085461de182608ee9f607f3f7d18f-Paper.pdf">Deep evidential regression</a></td>
      <td>NeurIPS</td>
      <td><a href="https://github.com/aamini/evidential-deep-learning">Code</a></td>
    </tr>
    <tr>
      <td>2023</td>
      <td><a href="https://proceedings.mlr.press/v202/deng23b/deng23b.pdf">Uncertainty estimation by fisher information-based evidential deep learning</a></td>
      <td>ICML</td>
      <td><a href="https://github.com/danruod/IEDL">Code</a></td>
    </tr>
    <tr>
      <td>2024</td>
      <td><a href="https://openreview.net/pdf?id=Si3YFA641c">R-edl: Relaxing nonessential settings of evidential deep learning</a></td>
      <td>ICLR</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td rowspan="2">Distance-aware</td>
      <td>2020</td>
      <td><a href="https://proceedings.mlr.press/v119/van-amersfoort20a/van-amersfoort20a.pdf">Uncertainty estimation using a single deep deterministic neural network</a></td>
      <td>ICML</td>
      <td><a href="https://github.com/y0ast/deterministic-uncertainty-quantification">Code</a></td>
    </tr>
    <tr>
      <td>2020</td>
      <td><a href="https://proceedings.neurips.cc/paper/2020/file/543e83748234f7cbab21aa0ade66565f-Paper.pdf">Simple and principled uncertainty estimation with deterministic deep learning via distance awareness</a></td>
      <td>NeurIPS</td>
      <td><a href="https://github.com/google/uncertainty-baselines/tree/master/baselines">Code</a></td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <td rowspan="3">External Methods</td>
      <td>Additional Network</td>
      <td>2020</td>
      <td><a href="https://arxiv.org/pdf/1908.07235">Density estimation in representation space to predict model uncertainty</a></td>
      <td>EDSMLS</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td rowspan="2">Gradient-based Analysis</td>
      <td>2018</td>
      <td><a href="https://arxiv.org/pdf/1805.08440">Classification uncertainty of deep neural networks based on gradient information</a></td>
      <td>IAPR</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2020</td>
      <td><a href="">Gradients as a measure of uncertainty in neural networks</a></td>
      <td>ICIP</td>
      <td><a href=""></a></td>
    </tr>
  </tbody>
</table>

#### Bayesian Methods
<table>
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th>Year</th>
      <th>Paper</th>
      <th>Venue</th>
      <th>Code</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td rowspan="11">Distributional Approximation</td>
      <td rowspan="7">Variational Inference</td>
      <td>2011</td>
      <td><a href="https://proceedings.neurips.cc/paper/2011/file/7eb3c8be3d411e8ebfab08eba5f49632-Paper.pdf">Practical variational inference for neural networks</a></td>
      <td>NeurIPS</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2015</td>
      <td><a href="https://proceedings.mlr.press/v37/blundell15.pdf">Weight uncertainty in neural network</a></td>
      <td>ICML</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2015</td>
      <td><a href="https://proceedings.neurips.cc/paper/2015/file/bc7316929fe1545bf0b98d114ee3ecb8-Paper.pdf">Variational dropout and the local reparameterization trick</a></td>
      <td>NeurIPS</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2015</td>
      <td><a href="https://proceedings.mlr.press/v37/rezende15.pdf">Variational inference with normalizing flows</a></td>
      <td>ICML</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2016</td>
      <td><a href="https://proceedings.mlr.press/v48/louizos16.pdf">Structured and efficient variational deep learning with matrix gaussian posteriors</a></td>
      <td>ICML</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2017</td>
      <td><a href="https://proceedings.mlr.press/v54/sun17b/sun17b.pdf">Learning structured weight uncertainty in Bayesian neural networks</a></td>
      <td>AISTATS</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2017</td>
      <td><a href="https://proceedings.mlr.press/v70/louizos17a/louizos17a.pdf">Multiplicative normalizing flows for variational bayesian neural networks</a></td>
      <td>ICML</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td rowspan="4">Laplace Approximation</td>
      <td>2018</td>
      <td><a href="https://discovery.ucl.ac.uk/id/eprint/10080902/1/kflaplace.pdf">A scalable laplace approximation for neural networks</a></td>
      <td>ICLR</td>
      <td><a href="https://github.com/BB-UCL/Lasagne">Code</a></td>
    </tr>
    <tr>
      <td>2020</td>
      <td><a href="https://proceedings.mlr.press/v119/kristiadi20a/kristiadi20a.pdf">Being bayesian, even just a bit, fixes overconfidence in relu networks</a></td>
      <td>ICML</td>
      <td><a href=""></a></td>
    </tr>    
    <tr>
      <td>2021</td>
      <td><a href="https://proceedings.mlr.press/v161/kristiadi21a/kristiadi21a.pdf">Learnable uncertainty under laplace approximations</a></td>
      <td>UAI</td>
      <td><a href="https://github.com/wiseodd/lula">Code</a></td>
    </tr>
    <tr>
      <td>2022</td>
      <td><a href="https://proceedings.mlr.press/v180/hobbhahn22a/hobbhahn22a.pdf">Fast predictive uncertainty for classification with Bayesian deep networks</a></td>
      <td>UAI</td>
      <td><a href="https://github.com/mariushobbhahn/LB_for_BNNs_official">Code</a></td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <td rowspan="7">Sampling-based</td>
      <td rowspan="2">Markov Chain Monte Carlo</td>
      <td>1992</td>
      <td><a href="https://glizen.com/radfordneal/ftp/bbp.pdf">Bayesian training of backpropagation networks by the hybrid Monte Carlo method</a></td>
      <td></td>
    </tr>
    <tr>
      <td>2011</td>
      <td><a href="https://arxiv.org/pdf/1206.1901">MCMC using Hamiltonian dynamics</a></td>
      <td></td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td rowspan="2">Stochastic Gradient Langevin Dynamics</td>
      <td>2011</td>
      <td><a href="https://icml.cc/2011/papers/398_icmlpaper.pdf">Bayesian learning via stochastic gradient Langevin dynamics</a></td>
      <td>ICML</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2021</td>
      <td><a href="https://www.tandfonline.com/doi/pdf/10.1080/01621459.2020.1847120">Stochastic gradient markov chain monte carlo</a></td>
      <td>JASA</td>
      <td><a href=""></a></td>
    </tr> 
    <tr>
      <td rowspan="2">Monte Carlo Dropout</td>
      <td>2016</td>
      <td><a href="https://proceedings.mlr.press/v48/gal16.pdf">Dropout as a bayesian approximation: Representing model uncertainty in deep learning</a></td>
      <td>ICML</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2017</td>
      <td><a href="https://proceedings.mlr.press/v70/li17a/li17a.pdf">Dropout inference in bayesian neural networks with alpha-divergences</a></td>
      <td>ICML</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td rowspan="1"> Adaptive Bayesian Neural Network (ABNN)</td>
      <td>2024</td>
      <td><a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Franchi_Make_Me_a_BNN_A_Simple_Strategy_for_Estimating_Bayesian_CVPR_2024_paper.pdf">Make me a bnn: A simple strategy for estimating bayesian uncertainty from pre-trained models</a></td>
      <td>CVPR</td>
      <td><a href="https://github.com/torch-uncertainty/torch-uncertainty">Code</a></td>
    </tr>
  </tbody>
</table>

#### Ensemble Methods
<table>
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th>Year</th>
      <th>Paper</th>
      <th>Venue</th>
      <th>Code</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td rowspan="4">Distributional Approximation</td>
      <td rowspan="1">Standard Ensembles</td>
      <td>2017</td>
      <td><a href="https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf">Simple and scalable predictive uncertainty estimation using deep ensembles</a></td>
      <td>NeurIPS</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td rowspan="2">Parameter-sharing Ensembles</td>
      <td>2019</td>
      <td><a href="https://arxiv.org/pdf/1910.08168">Deep sub-ensembles for fast uncertainty estimation in image classification</a></td>
      <td>arXiv</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2020</td>
      <td><a href="https://arxiv.org/pdf/2002.06715">Batchensemble: an alternative approach to efficient ensemble and lifelong learning</a></td>
      <td>ICLR</td>
      <td><a href="https://github.com/google/edward2">Code</a></td>
    </tr>
    <tr>
      <td rowspan="1">Distilled Ensembles</td>
      <td>2020</td>
      <td><a href="https://openreview.net/pdf?id=BygSP6Vtvr">Ensemble distribution distillation</a></td>
      <td>ICLR</td>
      <td><a href=""></a></td>
    </tr>
  </tbody>
</table>

#### Post-hoc Methods
<table>
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th>Year</th>
      <th>Paper</th>
      <th>Venue</th>
      <th>Code</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td rowspan="4">Calibration-based</td>
      <td rowspan="2">Temperature Scaling</td>
      <td>2019</td>
      <td><a href="https://arxiv.org/pdf/1909.13550">Well-calibrated model uncertainty with temperature scaling for dropout variational inference</a></td>
      <td>arXiv</td>
      <td><a href="https://github.com/mlaves/bayesian-temperature-scaling">Code</a></td>
    </tr>
    <tr>
      <td>2024</td>
      <td><a href="https://link.springer.com/content/pdf/10.1007/s00521-024-09505-4.pdf">Adaptive temperature scaling for robust calibration of deep neural networks</a></td>
      <td>NCA</td>
    </tr>
    <tr>
      <td rowspan="2">Conformal Prediction</td>
      <td>2022</td>
      <td><a href="https://www.nature.com/articles/s41467-022-34945-8.pdf">Estimating diagnostic uncertainty in artificial intelligence assisted pathology using conformal prediction</a></td>
      <td>Nature Communications</td>
      <td><a href="https://github.com/heolss/Conformal_analyses">Code</a></td>
    </tr>
    <tr>
      <td>2023</td>
      <td><a href="https://ojs.aaai.org/index.php/AAAI-SS/article/download/27492/27265">Quantifying deep learning model uncertainty in conformal prediction</a></td>
      <td>AAAI</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td rowspan="2">Augmentation-based</td>
      <td rowspan="2">Test-Time Augmentation</td>
      <td>2020</td>
      <td><a href="https://proceedings.mlr.press/v124/lyzhov20a/lyzhov20a.pdf">Greedy policy search: A simple baseline for learnable test-time augmentation</a></td>
      <td>UAI</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2021</td>
      <td><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf">Better aggregation in test-time augmentation</a></td>
      <td>ICCV</td>
      <td><a href=""></a></td>
    </tr>
  </tbody>
</table>

### EDL in Multi-view Learning

| Year |                                                                                                   Paper                                                                                                    | Venue |                                  Code                                   |
|:----:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----:|:-----------------------------------------------------------------------:|
| 2022 |   [Uncertainty estimation for multi-view data: The power of seeing the whole picture](https://proceedings.neurips.cc/paper_files/paper/2022/file/2ab3163ee384cd46baa7f1abb2b1bf19-Paper-Conference.pdf)    | NeurIPS | [Code](https://github.com/davidmcjung/multiview_uncertainty_estimation) |
| 2023 |  [Beyond unimodal: Generalising neural processes for multimodal uncertainty estimation](https://proceedings.neurips.cc/paper_files/paper/2023/file/839e23e5b1c52cfd1268f4023a3af0d6-Paper-Conference.pdf)  | NeurIPS  |                                                                         |
| 2024 |                                                                       [Predictive dynamic fusion](https://arxiv.org/pdf/2406.04802)                                                                        | ICML  |   [Code](https://github.com/liuyuxiang1021/Predivtive-Dynamic-Fusion)   |


## Methodological Improvement

### Evidence Collection
<table>
  <thead>
    <tr>
      <th></th>
      <th>Year</th>
      <th>Paper</th>
      <th>Venue</th>
      <th>Code</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td rowspan="3">Special Data Types</td>
      <td>2025</td>
      <td><a href="https://arxiv.org/pdf/2408.07331?">RSEA-MVGNN: Multi-view graph neural network with reliable structural enhancement and aggregation</a></td>
      <td>IF</td>
      <td><a href="https://github.com/junyu000/RSEA-MVGNN">Code</a></td>
    </tr>
    <tr>
      <td>2025</td>
      <td><a href="https://ieeexplore.ieee.org/document/10851430">Multiview Uncertainty-Aware Fusion for Human Activity Recognition via Dempster–Shafer Theory</a></td>
      <td>TII</td>
    <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2025</td>
      <td><a href="https://hal.science/hal-05001673v2/file/MvEK-NN.pdf">Multi-view evidential K-NN classification</a></td>
      <td>IF</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td rowspan="3">High Quality of evidence</td>
      <td>2024</td>
      <td><a href="https://pdf.sciencedirectassets.com/272144/1-s2.0-S1566253523X00116/1-s2.0-S1566253523004293/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQD0MLoiZhYaaiFTUEPj0%2FFNWhgVwrnp8GZ%2B%2Fdtw16IRUAIhALYczhDh%2F3W%2Fx2Pk5Z1f6wV9OvB%2BNMsiX8cCrEZdC%2BpRKrsFCLz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgwEkjXsigbg%2B5uYo44qjwULgM1%2B6Q670jNEswnq1hgwvNVK%2FpWv82P5PkYj%2BCkTu2fPRRhjIt3jjwo5tGSTKQ6R3ZdoyAWnXRx5LDYP0q7fEtcxj3kr4rzWLnX%2F%2FCvDG%2BASWhMVTHLnRgAbHXm%2B0pAexaCqFIyIgTmOdNT1gmE1FeoBPLQElMZytoLvtTonHsfi7%2FKds7Vrm2uFoa%2F7wt4WIgMOJjTcbtoWwDuNLWCI6bQUCe%2B4PLxfYI6MXd2AEN916644NukqIKFZ2mKDnVfDQYBxZh9wL2iSMkc0yAriWDfeKOt717NbtNtOX7A%2BkRDujAQOgeUJIUnRQ7jInmw9dn%2B1OikncmeyVVCLq3pd%2BuSjtFN1dz%2Fn83F%2BOOw1HiRsimakHeJQ91kMZEZHVei%2BDjxhU2XnT1Q52K97OBev7zsUuS1aF8d8ZwnFLZnlXql730PSMb2CPHoLDnh9xEim7ihZaL9I4Z7urYFpl%2Bb2yXy5Af5JvikZJT1Kx2anz3gZiwcJolp454pJk5gjmdhmXfergJUMLB1lHuDQcWz7AMA%2B15bjBYmjratjz9ye0FEQomBW32fljQ3X7Z%2F5RFclBGiqIKqinUA9Tl1uKBb6iddLE%2BosBygpyg2nCi5rtaM%2BElJgokFSjNRzNr9TZ24bbUuUsX%2FLkMo%2Bs%2Blh9gcnNETG%2FPfSDSwnv8Lbuhu19NYSYn1Z4%2BkWJdwrU7akkGokLdvt%2BB6sbOtIw6Mi0qjxXkg7GYBa0uZbHXHdYd1g1nJQ8wmMFX8nB%2B0mKsHnHDFFJPzGSBN22PNvX5zxgAcFd3LBRxAXfmgoDmivCR0nV09TjadeIxkQx9gzn%2BGDVpwNs4AfGL3Aewaa9vv00xYZYhL194wp5dqsTeIU4DbEMOv7wMsGOrABgKEk4dZNr2FFKnLnn2ZOrI9x7LKw4zks8Dp5%2BI1QlOpyEP%2BCBbN%2F2Am3qUbEN0QP0HBxZcGf7agCX%2BEMkdOdUISirfol8SPcX%2FzvX0eiTqNvw96XcR6SK9QLGIhLVe0KebJrAKi2ued8bMqubwcWOLR3q7ZxnMjI8W%2Fc2UALbv2P2eqj%2F0AOLUZPzrNUoMxE7FvrNKWZ2OrhH4y3A7rE%2FwtPuCDnwiWdv2M2hP6kpUo%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260121T032537Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYYTRABGEL%2F20260121%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=7ce7fb94f94b78a250b22cd92f7b4c5a10e342b7ebd2ea6baf02c24e36a006d1&hash=488577afdbf47492a482c28aaf520cde18d45df771aadd9925f2b0622fbe2f43&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1566253523004293&tid=spdf-6f9cf7fd-a294-48a4-82d1-0e05c78e784c&sid=a22f34a46dce7640c06b1829322b16a03d54gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0d0e560455530d52&rr=9c13b4968c89772b&cc=cn&kca=eyJrZXkiOiI1bk0yUEpHNml6eFFLcXhLU3JZYnRXS0xRYzhUdWNKZ3VTblNOeW9HbHhwcWdiNWdNMmcxbHFOOHJ3ZHZ4Y3cxaUh0UldTY3k4YTluWm1yS3FpTkxYWDQzbm9LbHgvczllSHkzNmM5eHBEbks2cEVOM0xMdlFPNXRGMC9CQ0VFNmRFWWgva2RIUXU1bDAyTWVBbkRHRjFzblpReUdDMWtZQUhocnAzVFJNalVGQlVVcSIsIml2IjoiZTFlODZiNWExYmRkZTU2MDFmMGQ5NmI2Mzg2NjJkOWIifQ==_1768965950560">Dual-level Deep Evidential Fusion: Integrating multimodal information for enhanced reliable decision-making in deep learning</a></td>
      <td>IF</td>
      <td><a href="https://github.com/Zhimin00/DDEF">Code</a></td>
    </tr>
    <tr>
      <td>2024</td>
      <td><a href="https://dl.acm.org/doi/pdf/10.1145/3664647.3681404">Dynamic evidence decoupling for trusted multi-view learning</a></td>
      <td>ACM MM</td>
      <td><a href="https://github.com/Lihong-Liu/CCML">Code</a></td>
    </tr>
    <tr>
      <td>2025</td>
      <td><a href="https://ieeexplore.ieee.org/abstract/document/11209409">Trustworthy Localized Corrections-guided Mutual Learning for Multi-View Learning</a></td>
      <td>ICME</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td rowspan="4">New Pseudo View</td>
      <td>2022</td>
      <td><a href="https://ieeexplore.ieee.org/document/9767662">Trusted multi-view classification with dynamic evidential fusion</a></td>
      <td>TPAMI</td>
      <td><a href="https://github.com/Han-Zongbo/TMC">Code</a></td>
    </tr>
    <tr>
      <td>2025</td>
      <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S1566253525004506">Trustworthy multimodal feature-enhanced fusion network for non-contact rotating machinery fault diagnosis</a></td>
      <td>IF</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2025</td>
      <td><a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4862049">Hierarchically trusted evidential fusion method with consistency learning for multimodal language understanding</a></td>
      <td>KBS</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2025</td>
      <td><a href="https://openreview.net/pdf?id=M3kBtqpys5">Trusted multi-view classification via evolutionary multi-view fusion</a></td>
      <td>ICLR</td>
      <td><a href="https://github.com/fupinhan123/TEF">Code</a></td>
    </tr>
  </tbody>
</table>

### Uncertainty Estimation

<table>
  <thead>
    <tr>
      <th></th>
      <th>Year</th>
      <th>Paper</th>
      <th>Venue</th>
      <th>Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">Calculation</td>
      <td>2025</td>
      <td><a href="https://dl.acm.org/doi/pdf/10.1145/3701716.3715534">Refining Confusion and Ignorance in Trusted Multi-View Classification</a></td>
      <td>WWW</td>
      <td><a href="https://github.com/muxixi727/RTMC">Code</a></td>
    </tr>
    <tr>
      <td>2025</td>
      <td><a href="https://openreview.net/pdf?id=ZzuaeYvLsJ">Deep Fuzzy Multi-view Learning for Reliable Classification</a></td>
      <td>ICML</td>
      <td><a href="https://github.com/siyuancncd/FUML">Code</a></td>
    </tr>
    <tr>
      <td>2025</td>
      <td><a href="https://openreview.net/pdf?id=U64wEbM7NB">Trusted Multi-View Classification with Expert Knowledge Constraints</a></td>
      <td>ICML</td>
      <td><a href="https://github.com/jie019/TMCEK_ICML2025">Code</a></td>
    </tr>
    <tr>
      <td>2025</td>
      <td><a href="https://ieeexplore.ieee.org/document/11045813">Uncertainty Quantification for Incomplete Multi-View Data Using Divergence Measures</a></td>
      <td>TIP</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td rowspan="2">Calibration</td>
      <td>2024</td>
      <td><a href="https://dl.acm.org/doi/pdf/10.1145/3664647.3681297">Building trust in decision with conformalized multi-view deep classification</a></td>
      <td>ACM MM</td>
      <td><a href=""></a></td>
    </tr>
    <tr>
      <td>2025</td>
      <td><a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Enhancing_Testing-Time_Robustness_for_Trusted_Multi-View_Classification_in_the_Wild_CVPR_2025_paper.pdf">Enhancing Testing-Time Robustness for Trusted Multi-View Classification in the Wild</a></td>
      <td>CVPR</td>
      <td><a href=""></a></td>
    </tr>
  </tbody>
</table>

### Opinion Aggregation

| Year |                                                           Paper                                                           | Venue |                                  Code                                   |
|:----:|:-------------------------------------------------------------------------------------------------------------------------:|:-----:|:-----------------------------------------------------------------------:|
| 2022 | [Trusted multi-view deep learning with opinion aggregation](https://ojs.aaai.org/index.php/AAAI/article/view/20724/20483) | AAAI  |   [Code](https://github.com/sunshiding/TMDLO_AAAI)   |
| 2022 | [Uncertainty-aware multiview deep learning for internet of things applications](https://ieeexplore.ieee.org/document/9906001) |  TII  |   [Code](https://github.com/xdmvteam/EMDL)   |
| 2023 | [Safe multi-view deep classification](https://ojs.aaai.org/index.php/AAAI/article/view/26066/25838) | AAAI  |      |
| 2024 | [Generalized Trusted Multi-view Classification Framework with Hierarchical Opinion Aggregation](https://arxiv.org/pdf/2411.03713?) | arXiv |   [Code](https://github.com/lshi91/GTMC-HOA)   |
| 2025 | [Trusted unified feature-neighborhood dynamics for multi-view classification](https://ojs.aaai.org/index.php/AAAI/article/download/33914/36069) | AAAI  |   [Code](https://github.com/JethroJames/TUNED)   |
| 2024 | [Enhancing Multi-View Classification Reliability with Adaptive Rejection](https://ojs.aaai.org/index.php/AAAI/article/view/34088/36243) | AAAI  |     |


## Challenging Scenarios

### Conflicting Views

| Year |                                                                      Paper                                                                       |  Venue  |                          Code                          |
|:----:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:-------:|:------------------------------------------------------:|
| 2024 |                     [Reliable conflictive multi-view learning](https://ojs.aaai.org/index.php/AAAI/article/view/29546/30911)                     |  AAAI   |        [Code](https://github.com/jiajunsi/RCML)        |
| 2025 |            [Multimodal Learning with Uncertainty Quantification based on Discounted Belief Fusion](https://arxiv.org/pdf/2412.18024)             | AISTATS | [Code](https://github.com/bezirganyan/DBF_uncertainty) |
| 2025 |                         [Navigating Conflicting Views: Harnessing Trust for Learning](https://arxiv.org/pdf/2406.00958)                          |  ICML   | [Code](https://github.com/OverfitFlow/Trust4Conflict)  |
| 2025 |                      [Deep Fuzzy Multi-view Learning for Reliable Classification](https://openreview.net/pdf?id=ZzuaeYvLsJ)                      |  ICML   |       [Code](https://github.com/siyuancncd/FUML)       |
| 2023 |                   [Rtmc: A rubost trusted multi-view classification framework](https://ieeexplore.ieee.org/document/10220047)                    |  ICME   |                                                        |
| 2024 | [Uvat: Uncertainty incorporated view-aware transformer for robust multi-view classification](https://ieeexplore.ieee.org/document/10666988)     |   TIP   |       [Code](https://github.com/li-yapeng/UVaT)        |
    
### Incomplete Views

| Year |                                                                                         Paper                                                                                         | Venue |                     Code                     |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----:|:--------------------------------------------:|
| 2023 |  [Exploring and exploiting uncertainty for incomplete multi-view classification](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_Exploring_and_Exploiting_Uncertainty_for_Incomplete_Multi-View_Classification_CVPR_2023_paper.pdf)    | CVPR  | [Code](https://github.com/xiexiaoxie41/UIMC) |
| 2024 | [Uvat: Uncertainty incorporated view-aware transformer for robust multi-view classification](https://ieeexplore.ieee.org/document/10666988)     |  TIP  |  [Code](https://github.com/li-yapeng/UVaT)   |
| 2024 | [Towards robust uncertainty-aware incomplete multi-view classification](https://arxiv.org/pdf/2409.06270?)     | arXiv |          |
| 2024 | [Evidential deep partial multi-view classification with discount fusion](https://arxiv.org/pdf/2408.13123?)     | arXiv |          |
| 2025 | [Uncertainty Quantification for Incomplete Multi-View Data Using Divergence Measures](https://ieeexplore.ieee.org/document/11045813)     |  TIP  |   |
| 2025 | [Trusted Cross-view Completion for incomplete multi-view classification](https://www.sciencedirect.com/science/article/abs/pii/S0925231225003947)     |  Neurocomputing  |   |
| 2026 | [Trustworthy data recovery for incomplete multi-view learning](https://pdf.sciencedirectassets.com/271605/1-s2.0-S0165168425X00098/1-s2.0-S0165168425002609/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEA8aCXVzLWVhc3QtMSJGMEQCIENfA4vowli2LzSDgdFyp1d%2Fd4AUZiT1HbAb%2F4qaG1%2BwAiBQi2Eoni8kJsnnqRQHQM7NVOB8ucsaZ%2FbhByQ7w1yXNyq7BQjX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMXxahaL3NbFw8etrBKo8FOfEb%2F77gyT3CYEDXgdp%2BN9TmtV8vFmOOX8gmF%2Fyh9n1G6TAFexiPFePdNCq0%2BVJhjlIedbiHSsZ5abpI211ND0tdkwDzkYxbkhr2aMu%2FrZtlHH5l73mgdzEil%2BSQlMh8bmHz9h2M%2BxneT1QhQRJmJKTu67HN2pmt6Ejwy06UW5WpwYssB9popYwCAu%2FoqeMa%2BOBTL48UQ4yUMTrX1ZtM3hT3I%2B9jias2nN4i8Gvt%2BWicYNJQqXq%2BsFhefqf81h6MV74X5FIXUS04gJh4bdU%2FRGUUuMJqG7v8PsJZlPsHEQohoRN9cX%2B1avLXMAFtIVQxXWK%2FUYYxITrE43gbfdWXoX%2B3WmhmSD%2FApQ5CfFQH2sk%2BcuZLGg42O12jWkyfyP16hXrzSaoqaGgpbU07oWO8ctDraPeUWTkMgFR9g54wn%2By7dBm%2Fp9Jx%2FxznQap8lH6ujWYcOeGtcDtj6h80SiluC%2BNu50H6x%2FEKmMeA7R2W7CK2rQpi%2BrBaXqExc6YII3acrmsec4RuL7GHJiiBltpg0%2BNu1a%2BRjMBeDSuxJm9d5M9yaLfGCoq%2FYqcyGacL3U7BJyfJ1pOCdgxHGnhxHZtIr3rd3KVmaHoIQ0zlML3gcECKhxTuuo666pa4j9rINfDevy4HOkXmA7izencVkjatKgNs%2B56PydXhfpwQFqi2D2%2BwE4WiwQbDyqzyzcRADbXkPLgPyR9p3bs2DiKh%2FI8dwIPPlXC1%2FROPEBEsak9Ahle4tAdikb4BFZhZRV1cPPHIyhv39WImlW4xbErV1DbW6eiRdRiV3aVsjCfxMa6ijdjdNoP5x%2FigjJOtaTlkxkybXa67Vurv6mUCRCYuCJRGgD9JoU0CUsDaMXw7rNhw4TCOgMfLBjqyAR5NCpxmMiA88%2FIYhGVpINklMjGfNNE40lDpHXAEBU9BZUwV%2BdZQn4n9CEF9fsCj5Hr%2B4N5p9rNKeC%2BH0GwwDWQpqhIpbQ6kUhu9hbIQVlCTN2ZZA9nA5hf93CHPFBCjDJXyuOcXrol9Oh7g6%2By7Lx9tsibZLijRCv9wAgITQZ%2FfqJ5%2BVMtjV%2F7WJiAo8ZjWMlarMigbPCvQRtaLe0Pgy6aRfFSo0oUJ%2FOYhwC1HbEndwmQ%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260122T061948Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYV2DCDIYJ%2F20260122%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=78eb31f052f8238264c36238b8086077881d31d6025e1cd8fa27b2d2019e22d3&hash=680596a26b7a8072f6b1670a7e81c0fec46f46b7bd673becf9ac17e79ed3b10b&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0165168425002609&tid=spdf-6b73be88-7bc2-4389-a786-fe9d688c2969&sid=7e6e61945578c7461388b7d98ebbd47f9abegxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1b13560455030957&rr=9c1cf11b5eb0db4f&cc=us)     |  SP   | [Code](https://github.com/ding6ding/TDR-IMV) |

### Imbalanced Problems

| Year |                                                                                         Paper                                                                                         | Venue  |                    Code                    |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------:|:------------------------------------------:|
| 2025 | [Fairness-Aware Multi-view Evidential Learning with Adaptive Prior](https://arxiv.org/pdf/2508.12997)     | arXiv  | [Code](https://github.com/li-yapeng/UVaT)  |
| 2025 | [Beyond Equal Views: Strength-Adaptive Evidential Multi-View Learning](https://dl.acm.org/doi/pdf/10.1145/3746027.3755092)     | ACM MM | [Code](https://github.com/Wednesque/SAEML) |
| 2026 |  [Trusted Multi-view Learning for Long-tailed Classification](https://arxiv.org/pdf/2511.09138)    |  AAAI  | [Code](https://github.com/cncq-tang/TMLC)  |

### Adversarial Attacks

| Year |                                                                                         Paper                                                                                         | Venue |                    Code                    |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----:|:------------------------------------------:|
| 2025 | [Reliable Disentanglement Multi-view Learning Against View Adversarial Attacks](https://www.ijcai.org/proceedings/2025/0720.pdf)     | IJCAI | [Code](https://github.com/Willy1005/2025-IJCAI-RDML)  |
| 2025 | [Evidential dissonance measure in robust multi-view classification to resist adversarial attack](https://www.sciencedirect.com/science/article/abs/pii/S156625352400383X)     |  IF   | [Code](https://github.com/Wednesque/SAEML) |

### Noisy Labels

| Year |                                                                                         Paper                                                                                         | Venue |                    Code                    |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----:|:------------------------------------------:|
| 2025 | [Trusted multi-view learning with label noise](https://www.ijcai.org/proceedings/2024/0582.pdf)     | IJCAI | [Code](https://github.com/YilinZhang107/TMNR)  |

### Open-set Scenarios

| Year |                                                                                        Paper                                                                                         | Venue  |                    Code                    |
|:----:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------:|:------------------------------------------:|
| 2025 |  [Trusted Open-World Multi-View Classification with Dynamic Opinion Aggregation](https://dl.acm.org/doi/pdf/10.1145/3746027.3755015)  | ACM MM |   |

## Beyond Classification

### Semi-supervised Learning

| Year |                                                                                        Paper                                                                                         | Venue |                    Code                    |
|:----:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----:|:------------------------------------------:|
| 2024 |  [Trusted Semi-Supervised Multi-View Classification With Contrastive Learning](https://ieeexplore.ieee.org/abstract/document/10475541)  |  TMM  |   |

### Zero-shot Learning

| Year |                                                                                        Paper                                                                                         | Venue  |                     Code                     |
|:----:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------:|:--------------------------------------------:|
| 2024 |  [Crest: Cross-modal resonance through evidential deep learning for enhanced zero-shot learning](https://dl.acm.org/doi/pdf/10.1145/3664647.3681629)  | ACM MM | [Code](https://github.com/JethroJames/CREST) |

### Prompt Learning

| Year |                                                                                        Paper                                                                                        | Venue  |                     Code                     |
|:----:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------:|:--------------------------------------------:|
| 2024 |  [EPE-P: Evidence-based Parameter-efficient Prompting for Multimodal Learning with Missing Modalities](https://arxiv.org/pdf/2412.17677)              | ICASSP | [Code](https://github.com/Boris-Jobs/EPE-P_MLLMs-Robustness) |

### Representation Learning

| Year |                                                                                        Paper                                                                                        | Venue |                    Code                    |
|:----:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----:|:------------------------------------------:|
| 2024 |  [Multi-trusted cross-modal information bottleneck for 3d self-supervised representation learning](https://www.sciencedirect.com/science/article/abs/pii/S095070512300967X)              |  KBS  |                                            |
| 2024 |  [Trusted 3D self-supervised representation learning with cross-modal settings](https://link.springer.com/article/10.1007/s00138-024-01556-w)              |  MVA  |                                            |
| 2025 |  [Uncertainty Quantification via Hölder Divergence for Multi-View Representation Learning](https://arxiv.org/pdf/2411.00826)              |  TMM  | [Code](https://github.com/wmh12138/HDMVL.) |

### Clustering

| Year |                                                                                        Paper                                                                                        | Venue |                       Code                        |
|:----:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----:|:-------------------------------------------------:|
| 2023 |  [Adaptive weighted multi-view evidential clustering](https://www.researchgate.net/profile/Zhe-Liu-107/publication/372242432_Adaptive_Weighted_Multi-View_Evidential_Clustering/links/64ac33d08de7ed28ba8a0d29/Adaptive-Weighted-Multi-View-Evidential-Clustering.pdf)              | ICANN |  [Code](https://github.com/MAMBA126JAMES/WMVEC)   |
| 2024 |  [Adaptive weighted multi-view evidential clustering with feature preference](https://link.springer.com/article/10.1007/s00138-024-01556-w)              |  KBS  | [Code](https://github.com/MAMBA126JAMES/WMVEC-FP) |
| 2025 |  [How to characterize imprecision in multi-view clustering?](https://arxiv.org/pdf/2404.04970?)              | TETCI |    [Code](https://github.com/JinyiXUres/MvLRECM)     |
| 2025 |  [Self-supervised Trusted Contrastive Multi-view Clustering with Uncertainty Refined](https://ojs.aaai.org/index.php/AAAI/article/view/33902/36057)              | AAAI  |    [Code](https://github.com/ShizheHu)     |

## Application

### Medical Science

<table>
  <thead>
    <tr>
      <th></th>
      <th>Year</th>
      <th>Paper</th>
      <th>Venue</th>
      <th>Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="20">Medical Image Classification</td>
      <td>2022</td>
      <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S0888613X22001256">Deep evidential fusion network for medical image classification</a></td>
      <td>IJAR</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://arxiv.org/pdf/2405.02918?">MERIT: Multi-view evidential learning for reliable and interpretable liver fibrosis staging</a></td>
      <td>MedIA</td>
      <td><a href="https://github.com/HenryLau7/MERIT">Code</a></td>
    </tr>
    <tr>      
      <td>2024</td>
      <td><a href="https://ieeexplore.ieee.org/abstract/document/10822671">Clinical Insight-Augmented Multi-View Learning for Alzheimer's Detection in Retinal OCTA Images</a></td>
      <td>BIBM</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2023</td>
      <td><a href="https://arxiv.org/pdf/2307.04981">A Multi-view Impartial Decision Network for Frontotemporal Dementia Diagnosis</a></td>
      <td>arXiv</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2024</td>
      <td><a href="https://arxiv.org/pdf/2405.18167">Confidence-aware multi-modality learning for eye disease screening</a></td>
      <td>MedIA</td>
      <td><a href="https://github.com/Cocofeat/EyeMoSt">Code</a></td>
    </tr>
    <tr>      
      <td>2023</td> 
      <td><a href="https://arxiv.org/pdf/2303.09790">Reliable multimodality eye disease screening via mixture of student'st distributions</a></td>
      <td>MICCAI</td>
      <td><a href="https://github.com/Cocofeat/EyeMoSt">Code</a></td>
    </tr>
    <tr>      
      <td>2023</td>
      <td><a href="https://www.nature.com/articles/s41467-023-42444-7.pdf">Uncertainty-inspired open set learning for retinal anomaly identification</a></td>
      <td>Nature Communications</td>
      <td><a href="https://github.com/LooKing9218/UIOS">Code</a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S0957417425018457?via%3Dihub">Glaucoma progression prediction using multi-modal data and trusted multi-view learning</a></td>
      <td>ESWA</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S0895611124001563">Adaptive fusion of dual-view for grading prostate cancer</a></td>
      <td>CMIG</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2024</td>
      <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S1566253524003701">TDF-Net: Trusted Dynamic Feature Fusion Network for breast cancer diagnosis using incomplete multimodal ultrasound</a></td>
      <td>IF</td>
      <td><a href=""></a></td>
    </tr>    
    <tr>      
      <td>2024</td>
      <td><a href="https://papers.miccai.org/miccai-2024/paper/2652_paper.pdf">Uncertainty-aware multi-view learning for prostate cancer grading with dwi</a></td>
      <td>MICCAI</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2024</td>
      <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S1476927124001907">TMODINET: A trustworthy multi-omics dynamic learning integration network for cancer diagnostic</a></td>
      <td>Comput. Biol. Chem.</td>
      <td><a href="https://github.com/isGao1109/TMODINET/tree/master">Code</a></td>
    </tr>
    <tr>      
      <td>2024</td>
      <td><a href="https://watermark02.silverchair.com/btae159.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA3EwggNtBgkqhkiG9w0BBwagggNeMIIDWgIBADCCA1MGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMJ5oD2Xw2lRVoJnMRAgEQgIIDJC70TbMi0B3sMFdCrH5CfBMAo65PZVJtZiAstfQjqoi_W1dW472hSjOYXcOYWxdEt7B9rEO51surKSRCW26mtfXOyt6upevJ264GKxUOFlKJGfsSyo4yt68jo5KDmN1yhkknojiEvrXSZ_9mTy_PbtagpV5Mo8zMgH6ustvQyGycp7_k-l80HbxNFj1-OlhJYS8qbWMZs36rXslA0Pr0jq_7Ois9CdnlL5brvxILUQf1Jv1cCZHu6z0Eo6uCHV_ykYDZAH7pZDOYcNlP2xSkxEW5mvlijRVVYzrt86eC8-tNFW2PithYwivBmoam51zCY2bjLx2JbeBozJpI1Vk4jvmhT5gk-sD-UQvFSjmXHpPRZ9MruMBlWIHhcG0ZpL3b2wPx3kRDacKlWbdaI0oiiGXyaZjWWp9q6l1XczD53_PGmQBwVlETSmAjudAT7kvgNp9ox88OQMIHc_4LhZtWPWxJyPbmxYYBpOGv9IY2-jfhGc-OxcUgMBiOEL1ebg3alTckMtvI_8HJaQ81UMDG75cZNCXhiriiMath8EXusBCi5-OYTVSnsevNHDNbVBlBzZLn3ULNV9PkoS0nJj1-eeY7umsPRBTAF8uX3e-upNBQu7X4zihBbhDe_TrN7Cjw70v93QxLR-ctW-rbD5417368oFNpDhqO1oKQVaY8VPqQEcLYvsW2ZQWznoNVmPNCIvMPjpQD4v9v4jvV-0Vh-_bhPETPzp0T9_ZsQAbNR9q_ZfzYvlHx1_A0j-8ZKCWXZlx6cbSdsXRlyDl1M-aojxNDUEa3MTQ8s3ol_Y0Kgfxhghg-qQUxxHQ_iCegX1BcL6RVm8crazGcEyEyWZi5V7o_Rh6vUMIhbOEUMiXTWIYfkfqI8PPn0mbyzhsdKXDv73eYUkK5PpCbP5XKzFzRjn9GcgPQtpVu95iPURIlWmWvI-j7koLqDs0YhRXSJ1oalBF_b6bffriZ9oNFF1dzwrdnpHxO16WRMFvtdjjMVN22QRT2povQHGhQlLT4XZbd8j89ndrordEC8OnYBJA_UvCp85RL_et1kjxLD6M5i8vFrQTtfQ">HyperTMO: a trusted multi-omics integration framework based on hypergraph convolutional network for patient classification</a></td>
      <td>Bioinformatics</td>
      <td><a href="https://github.com/ippousyuga/HyperTMO">Code</a></td>
    </tr>
    <tr>      
      <td>2023</td>
      <td><a href="https://ieeexplore.ieee.org/abstract/document/10385301?casa_token=C3JhJy850DIAAAAA:lG4JvcrQunHG-4AlctZYYFpynraem7IsTu3MUgBEiKXs-TmOsjT3tHGEpVkqcKA0C1VineGFCQ">Trusted Fine-grained Medical Image Classification through Multiple Evidence Fusion</a></td>
      <td>BIBM</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2024</td>
      <td><a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10528291">A multi-view deep evidential learning approach for mammogram density classification</a></td>
      <td>IEEE Access</td>
      <td><a href="https://github.com/MultiCancerResearch/MV-DEFEAT">Code</a></td>
    </tr>
    <tr>      
      <td>2024</td>
      <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S1361841524002196">E2-MIL: An explainable and evidential multiple instance learning framework for whole slide image classification</a></td>
      <td>MedIA</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2024</td>
      <td><a href="https://papers.miccai.org/miccai-2024/paper/0746_paper.pdf">Correlation-Adaptive Multi-view CEUS Fusion for Liver Cancer Diagnosis</a></td>
      <td>MICCAI</td>
      <td><a href="https://github.com/shukangzh/CAMVF">Code</a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://ojs.aaai.org/index.php/AAAI/article/download/32642/34797">A Trusted Lesion-assessment Network for Interpretable Diagnosis of Coronary Artery Disease in Coronary CT Angiography</a></td>
      <td>AAAI</td>
      <td><a href="https://github.com/PerceptionComputingLab/CAD_Diagnosis/">Code</a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S002002552401613X?casa_token=Ael2g6bDC7IAAAAA:JnLN41h5C464bYh2zr8df1Op6mMiunarTgaYhlS6rwRNhEW2U6T2AQgkI0_BDgnfQumrNIQ5kS0">Trust EEG epileptic seizure detection via evidential multi-view learning</a></td>
      <td>IS</td>
      <td><a href="https://github.com/Wednesque/Trust-EEG-Epileptic-Seizure-Detection-via-Evidential-Multi-view-Learning">Code</a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://www.sciencedirect.com/science/article/abs/pii/S0957417425030696">Reliable Decision Making on Clinical EEG: Trusted Multi-View Learning with Subjective Logic for Uncertainty Quantification</a></td>
      <td>ESWA</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td rowspan="6">Medical Image Segmentation</td>
      <td>2022</td>
      <td><a href="https://arxiv.org/pdf/2206.09309">Tbrats: Trusted brain tumor segmentation</a></td>
      <td>MICCAI</td>
      <td><a href="https://github.com/Cocofeat/TBraTS">Code</a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://arxiv.org/pdf/2406.18327?">Multi-modal evidential fusion network for trustworthy PET/CT tumor segmentation</a></td>
      <td>KBS</td>
      <td><a href="https://github.com/QPaws/MEFN">Code</a></td>
    </tr>
    <tr>      
      <td>2024</td>
      <td><a href="https://arxiv.org/pdf/2311.06400">EviPrompt: A training-free evidential prompt generation method for adapting segment anything model in medical images</a></td>
      <td>TIP</td>
      <td><a href="https://github.com/SPIresearch/EviPrompt">Code</a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://ojs.aaai.org/index.php/AAAI/article/download/33911/36066">Multi-view Evidential Learning-based Medical Image Segmentation</a></td>
      <td>AAAI</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://hal.science/hal-04681852/document">Deep evidential fusion with uncertainty quantification and reliability learning for multimodal medical image segmentation</a></td>
      <td>IF</td>
      <td><a href="https://github.com/iWeisskohl/Deep-evidential-fusion">Code</a></td>
    </tr>
    <tr>      
      <td>2023</td>
      <td><a href="https://arxiv.org/pdf/2303.10049">Uncertainty-informed mutual learning for joint medical image classification and segmentation</a></td>
      <td>MICCAI</td>
      <td><a href="https://github.com/KarryRen/UML">Code</a></td>
    </tr>
    <tr>   
      <td rowspan="3">Other Specific Tasks</td>
      <td>2025</td>
      <td><a href="https://www.nature.com/articles/s41598-025-93770-3.pdf">Multimodal multi-instance evidence fusion neural networks for cancer survival prediction</a></td>
      <td>SR</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://arxiv.org/pdf/2412.01215?">EsurvFusion: An evidential multimodal survival fusion model based on Gaussian random fuzzy numbers</a></td>
      <td>TFS</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://ieeexplore.ieee.org/abstract/document/11239058/">Evidence-Based Multidisease Prediction via Dynamic Knowledge Evolution and Multimodal Reasoning in EMRs</a></td>
      <td>CME</td>
      <td><a href=""></a></td>
    </tr>
  </tbody>
</table>

### Others

<table>
  <thead>
    <tr>
      <th></th>
      <th>Year</th>
      <th>Paper</th>
      <th>Venue</th>
      <th>Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">Fake News Detection</td>
      <td>2024</td>
      <td><a href="https://aclanthology.org/2024.acl-long.316.pdf">Event-radar: Event-driven multi-view learning for multimodal fake news detection</a></td>
      <td>ACL</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://ieeexplore.ieee.org/abstract/document/11198721">Multimodal Fake Information Detection Based on Evidential Deep Learning</a></td>
      <td>AIoTC</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://ieeexplore.ieee.org/abstract/document/10858867">TMEF-BI: Trusted Multimodal Evidential Fusion Considering Behavior Information for Fake News Detection</a></td>
      <td>TCSS</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://ieeexplore.ieee.org/document/11229212">Multi-View Graph Learning with Dynamic Evidential Fusion for Response Forecasting</a></td>
      <td>IJCNN</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td rowspan="7">Sentiment Analysis</td>
      <td>2022</td>
      <td><a href="https://ieeexplore.ieee.org/document/9862943">Deep Tensor Evidence Fusion Network for Sentiment Classification</a></td>
      <td>TCSS</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://www.researchgate.net/profile/Shuai-Wang-286/publication/387092899_UEFN_Efficient_uncertainty_estimation_fusion_network_for_reliable_multimodal_sentiment_analysis/links/678721aff8b7bf1abcbba494/UEFN-Efficient-uncertainty-estimation-fusion-network-for-reliable-multimodal-sentiment-analysis.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uRG93bmxvYWQiLCJwcmV2aW91c1BhZ2UiOiJwdWJsaWNhdGlvbiJ9fQ">UEFN: efficient uncertainty estimation fusion network for reliable multimodal sentiment analysis</a></td>
      <td>Applied Intelligence</td>
      <td><a href="https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK">Code</a></td> 
    </tr>
    <tr>      
      <td>2024</td>
      <td><a href="https://www.mdpi.com/2079-9292/13/3/662">Hybrid uncertainty calibration for multimodal sentiment analysis</a></td>
      <td>MDPI</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://www.xoveexu.com/file/paper/25-06-ICME-UniMSA.pdf">Decoding Emotional Silences: Reliable Multimodal Sentiment Analysis with Bipolar Uncertainty</a></td>
      <td>ICME</td>
      <td><a href="https://github.com/SuperPower97/UniMSA">Code</a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://dl.acm.org/doi/pdf/10.1145/3746252.3761430">ESED: Emotion-Specific Evidence Decomposition for Uncertainty-Aware Multimodal Emotion Recognition in Conversation</a></td>
      <td>CIKM</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://djingwang.github.io/works/TACL-%20A%20Trusted%20Action-enhanced%20Curriculum%20Learning%20Approach%20to%20Multimodal%20Affective%20Computing.pdf">TACL: A Trusted Action-enhanced Curriculum Learning Approach to Multimodal Affective Computing</a></td>
      <td>Neurocomputing</td>
      <td><a href="https://github.com/huggingface/transformers/tree/main/src/transformers/models/t5">Code</a></td>
    </tr>
    <tr>      
      <td rowspan="8">Autonomous Driving</td>
      <td>2025</td>
      <td><a href="https://arxiv.org/pdf/2508.20066">PAUL: Uncertainty-Guided Partition and Augmentation for Robust Cross-View Geo-Localization under Noisy Correspondence</a></td>
      <td>arXiv</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://ieeexplore.ieee.org/abstract/document/11024149">Confidence Fusion with  Representation Distribution  and Mixture of Experts for  Multi-modal Radar Target  Recognition</a></td>
      <td>TAES</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2024</td>
      <td><a href="https://arxiv.org/pdf/2410.08739?">MMLF: Multi-modal Multi-class Late Fusion for Object Detection with Uncertainty Estimation</a></td>
      <td>arXiv</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2024</td>
      <td><a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10606099">EVORA: Deep Evidential Traversability Learning for  Risk-Aware Off-Road Autonomy</a></td>
      <td>T-RO</td>
      <td><a href="https://xiaoyi-cai.github.io/evora/">Code</a></td>
    </tr>
    <tr>      
      <td>2024</td>
      <td><a href="https://ieeexplore.ieee.org/abstract/document/10540258">“Where Does the Devil Lie?”: Multimodal Multitask  Collaborative Revision Network for Trusted  Road Segmentation</a></td>
      <td>TMM</td>
      <td><a href=""></a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://ieeexplore.ieee.org/abstract/document/10971954">Human-Centric Context and Self-Uncertainty-Driven Multi-Modal Large Language Model for Training-Free Vision-Based Driver State Recognition</a></td>
      <td>TITS</td>
      <td><a href="https://github.com/w64228013/HSUM">Code</a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://ieeexplore.ieee.org/abstract/document/10927027">Trustworthy Driver State Perception via Contextual Interaction-Driven Evidential Vision-Language Fusion in Vehicular Cyber-Physical Systems</a></td>
      <td>TITS</td>
      <td><a href="https://github.com/w64228013/TDSP">Code</a></td>
    </tr>
    <tr>      
      <td>2025</td>
      <td><a href="https://ieeexplore.ieee.org/abstract/document/11073807">Evidential Multimodal Fusion Network for Trusted Pedestrian Crossing Intent Prediction</a></td>
      <td>TCSS</td>
      <td><a href=""></a></td>
    </tr>
  </tbody>
</table>

