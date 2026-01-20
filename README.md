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


## THEORETICAL FOUNDATIONS AND TAXONOMY

### Theoretical Foundations

