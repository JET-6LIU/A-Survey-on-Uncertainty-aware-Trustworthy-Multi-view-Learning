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
- [THEORETICAL FOUNDATIONS AND TAXONOMY](#theoretical-foundations-and-taxonomy)
  - [Theoretical Foundations](#theoretical-foundations)

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


    <tr>
      <td></td>
      <td><a href=""></a></td>
      <td></td>
      <td><a href="">Code</a></td>
    </tr>
