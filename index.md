---
layout: project_page
permalink: /

title: "WaveMind: Towards a Generalist EEG Foundation Model Aligned to Textual and Visual Modalities"
authors:
    Ziyi Zeng$^1$,Zhenyang Cai$^1$,Yixi Cai$^1$,Xidong Wang$^1$,  <br>
    Rongsheng Wang$^1$, Siqi Cai$^2$, Haizhou Li$^1$,Benyou Wang$^1$  
affiliations:
    $^1$ The Chinese University of Hong Kong, Shenzhen  <br>
    $^2$ Harbin Institute of Technology, Shenzhen
paper: https://www.overleaf.com/6791821851jzqdxbyycrrh#d3b960
video: 
code: 
data: 
---

<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
While electroencephalography (EEG) interpretation using multimodal large language models (MLLMs) offers a novel approach for analyzing brain signals, the inherent complexity of brain activity poses significant challenges. This complexity stems from concurrent cognitive functions associated with consciousness and non-cognitive processes involved in homeostasis, generating distinct supervisory modalities during model training. To address these limitations, we propose WaveMind, an EEG alignment framework designed for EEG-MLLM training that projects EEG data into a shared semantic space across different supervisory modalities. To develop a cross-task EEG interpretation chatbot, we further contribute WaveMind-Instruct, comprising 338k GPT-assisted synthesized instruction pairs for fine-tuning. The resulting chatbot demonstrates robust classification performance and enables flexible, open-ended conversations covering four downstream tasks. Ablative analysis reveals the complementary relationship between diverse brain activities and supervision modalities, providing valuable insights for both neuroscience and the development of general-purpose EEG interpretation systems.
        </div>
    </div>
</div>

<div style="text-align: center; margin: 20px 0;">
<img src="static\image\Graphic_abstract.png" width="600" height="400">
</div>
## Background

## Objective
The main objective of this paper was to prove that singular type of brain activity limits the ability to generalize EEG_MLLM. We propose a multimodal EEG alignment framework named `WaveMind` for EEG-MLLM to solve the limitation.




## Methodology
### Architecture
Decode the raw EEG signals $X_e \in \mathbb{R}^{T \times C}$ into neural language tokens  $W = \{w_1, \dots, w_N\}$ via LLM backbone, as shown in the *Figure 1* below. As maany works have shown that inserting category information is beneficial for model generation. Therefore, WaveMind incorporates Retrieval-Augmented Generation (RAG) module that stores multimodal supervision's features (i.e. $\hat{Z}^I$ and $\hat{Z}^T$) with their category.

<div style="text-align: center; margin: 20px 0;">
  <img src="static\image\Architecture-1.png" width="900" height="500">
</div>
*Figure 1*: **The overall architecture of WaveMind.** Left: three-stage training procedure. Right: inference procedure of WaveMind. The system projects EEG data into a unified semantic space and integrates retrieval-augmented generation (RAG) for robust language generation.

### Training Paradigm
#### Stage 1: **Dual-Supervision Representation Alignment**
We align EEG features into a unified space using a dual-supervised CLIP framework:

• CLIP-ViT extracts image-guided features : $\mathbf{Z}_I \in \mathbb{R}^{768}$ 

• CLIP-BERT produces semantic features: $\mathbf{Z}_T \in \mathbb{R}^{768}$
After L2 normalization, both are mapped into the same CLIP space.

The objective function combines two InfoNCE losses: $$\mathcal{L} = \lambda \mathcal{L}_{\text{img}} + (1 - \lambda)\mathcal{L}_{\text{txt}}$$.  We train on 1.2M EEG pairs, outperforming 7 baseline encoders. Adding an auxiliary classification loss $$\mathcal{L}_{cls}$$ showed no performance gain

#### Stage 2：**Cold-start Training for the Adapter**
We propose pre-training the adapter on image-domain data $\hat{Z_I}$ (sharing CLIP space with EEG features $\hat{Z_e}$) using LLaVA-Pretrain-558k before EEG instruction-tuning. This aligns the MLLM with CLIP space and initializes EEG-domain tuning.

##### Stage 3: **EEG Instruction Tuning**
At this stage, we perform instruction tuning using the *WaveMind\_Instruct-338K*. In this stage, LoRA module and modality-adapter are unfrozen during training, while ATMM is frozen during training.

<div style="text-align: center; margin: 20px 0;">
<img src="static\image\Instruction.png" width="900" height="500">
</div>

*Figure 2*: **Instruction construction pipeline of WaveMind.** The raw signals are first pre-processed
into segments with the same configuration, then executed with different instruction synthesis processes
depending on the type of supervision. We have constructed four types of instructions to ensure the model learns diverse knowledge.



## Result 
### **Classification Evaluation**
We use *WaveMind-Bench* to evaluate the MLLM’s ability to recognize objects in visual stimuli and annotation categories represented in EEG. Table 4 presents MCQ classification results. WaveMind’s performance was assessed across three distinct methods: random/real EEG and with the RAG module. Crucially, when predicting with real EEG data, WaveMind significantly outperforms the random input baseline across all evaluated datasets. The RAGmodule substantially improves classification results across most EEG datasets. Notably, cognitive task classification and MCQs with many options show the most significant gains: THING-EEG’s 40-way accuracy doubled from 0.122 to 0.250, while ImageNet-EEG increased from 0.574 to 0.603. Non-cognitive tasks also benefited, with TUEV improving from 0.888 to 0.904 and TUAB reaching 0.575. Only SEED showed a slight decrease.

<table border="1" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th rowspan="2"><b>Dataset</b></th>
      <th rowspan="2"><b>Evaluation Protocol</b></th>
      <th rowspan="2"><b>k</b></th>
      <th colspan="3"><b>Random EEG</b></th>
      <th colspan="3"><b>Real EEG</b></th>
      <th colspan="3"><b>EEG w/RAG<sup>†</sup></b></th>
    </tr>
    <tr>
      <th>2</th><th>4</th><th>k</th>
      <th>2</th><th>4</th><th>k</th>
      <th>2</th><th>4</th><th>k</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TUEV</td><td>SI</td><td>6</td>
      <td>0.434</td><td>0.240</td><td>0.159</td>
      <td>0.940</td><td>0.867</td><td>0.888</td>
      <td>0.925</td><td>0.890</td><td>0.904</td>
    </tr>
    <tr>
      <td>TUAB</td><td>SI</td><td>2</td>
      <td>0.501</td><td>/</td><td>/</td>
      <td>0.736</td><td>/</td><td>/</td>
      <td>0.741</td><td>/</td><td>/</td>
    </tr>
    <tr>
      <td>SEED</td><td>SD</td><td>4</td>
      <td>0.515</td><td>/</td><td>0.335</td>
      <td>0.684</td><td>/</td><td>0.543</td>
      <td>0.676</td><td>/</td><td>0.529</td>
    </tr>
    <tr>
      <td>ImageNet-EEG</td><td>SD</td><td>40</td>
      <td>0.507</td><td>0.244</td><td>0.021</td>
      <td>0.914</td><td>0.853</td><td>0.574</td>
      <td>0.937</td><td>0.887</td><td>0.603</td>
    </tr>
    <tr>
      <td>THING-EEG</td><td>SD</td><td>40</td>
      <td>0.474</td><td>0.243</td><td>0.027</td>
      <td>0.760</td><td>0.554</td><td>0.122</td>
      <td>0.869</td><td>0.721</td><td>0.250</td>
    </tr>
  </tbody>
</table>
<p><sup>†</sup> RAG: Retrieval-Augmented Generation</p>

*Table1* : **Averaged Classification Result on WaveMind-Bench**. The weight accuracy over *k* options is reported where each question consists of 1 correct and *k-1* wrong options. The model is asked to output the letter represented by the correct options


### **Generalization Evaluation**
For cognitive tasks using the THING-EEG dataset, we additionally assess closed-set and Subject-Dependent conditions. As shown in Table5, it maintained consistent and accurate decoding performance whether encountering unseen object categories or untrained subjects.

<table border="1" cellspacing="0" cellpadding="6">
  <thead>
    <tr>
      <th rowspan="2"></th>
      <th rowspan="2"></th>
      <th colspan="3">Closed-set (1573 class)</th>
      <th colspan="3">Zero-shot (200 clsss)</th>
    </tr>
    <tr>
    </tr>
    <tr>
      <th rowspan=""></th>
      <th colspan="1">k</th>
      <th>2</th><th>4</th><th>40</th>
      <th>2</th><th>4</th><th>40</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1">Group</td>
      <td rowspan="1">Chances</td>
      <td>0.500</td><td>0.250</td><td>0.033</td>
      <td>0.500</td><td>0.250</td><td>0.033</td>
    </tr>
    <tr>
      <td rowspan="2">Real EEG</td>
      <td><b>SD</b></td>
      <td>0.728</td><td>0.504</td><td>0.096</td>
      <td>0.756</td><td>0.574</td><td>0.128</td>
    </tr>
    <tr>
      <td><b>SI</b></td>
      <td>0.680</td><td>0.419</td><td>0.074</td>
      <td>0.689</td><td>0.442</td><td>0.058</td>
    </tr>
    <tr>
      <td rowspan="2">EEG w/RAG</td>
      <td><b>SD</b></td>
      <td>0.786</td><td>0.627</td><td>0.182</td>
      <td>0.862</td><td>0.732</td><td>0.243</td>
    </tr>
    <tr>
      <td><b>SI</b></td>
      <td>0.698</td><td>0.492</td><td>0.108</td>
      <td>0.761</td><td>0.578</td><td>0.159</td>
    </tr>
  </tbody>
</table>

*Table2*: **K-way Classification Performance on THING-EEG Dataset** Weighted accuracy is reported as the class imbalance in the closed-set evaluation.

## Contributions

• **Unified EEG Alignment Framework**: We propose WaveMind, a novel alignment framework that projects EEG signals paired with diverse modalities into a shared semantic space.

• **Comprehensive Dataset and Benchmark**: We synthesized WaveMind-Instruct, the first cross-task instruction dataset comprising 4 instruction-tuning types and 2 chat scenarios, along with WaveMind-Bench, which contains 12K MCQs, to facilitate evaluation of EEGMLLMs.

• **Multi-Stage Training and Performance**: We propose a three-stage training scheme to fully unlock the model’s ability to recognize and understand EEG. The model performs well in classification tasks and has initially acquired the ability to open question answering.
