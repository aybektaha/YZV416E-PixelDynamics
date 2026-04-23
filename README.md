# YZV416E Computer Vision
**Istanbul Technical University**  
**Department of AI and Data Engineering**  
**Final Project Proposal Submission - Spring 2026 Term**  
**Adapting Vision Models to Domain-Specific Tasks using Computer Vision Techniques**  

**Due:** April 23, 2026 &nbsp;&nbsp;&nbsp;&nbsp; **Weight:** 30% of final grade

---

## 1. Topic Selection
[ G ] Motion-Based Object Segmentation via Region Growing | FlowFormer | DAVIS / SegTrack v2

**Selected Topic:** G – Motion-Based Object Segmentation via Region Growing

## 2. Team Information 
- **Team name:** FlowMakers
- **Student 1 (major / role):** Aybek Taha (AI & Data Eng. / Role: FlowFormer Pipeline & Extraction)
- **Student 2 (major / role):** Omar Qawasmi (AI & Data Eng. / Role: Region Growing Logic Development)
- **Student 3 (major / role):** [Student 3 Name] (AI & Data Eng. / Role: Dataset Setup & Metric Evaluation)

**Task Division:**
Student 1 will handle the deep learning side by setting up FlowFormer to extract high-quality optical flow maps. Student 2 is going to take those flow maps and build the actual Region Growing algorithm to cluster the pixels. Student 3 will manage the dataset files, handle all the image resizing, and calculate our final accuracy scores.

## 3. Project Theme and Motivation
- **What Computer Vision problem are you tackling?** 
We are focusing on Moving Object Segmentation (MOS). We aim to perform accurate motion-based segmentation on video sequences by integrating optical flow features with classical region growing algorithms to robustly segment moving objects from their backgrounds.
- **Why is this problem challenging?**
Regular image segmentation usually just looks at colors or textures. The problem is, this completely fails if the object and the background look similar—like a grey car driving on a grey road. Handling camera movement (ego-motion) and the aperture problem, where flat textures make it hard to track pixel movement, are significant challenges.
- **What do you expect the model to learn or struggle with on your specific dataset or domain?**
If a group of pixels is moving together in the same direction, they almost certainly belong to the same object (Gestalt principle of "Common Fate"). We expect our algorithm to easily segment camouflaged objects based purely on their movement. However, it will probably struggle a bit with heavy occlusions, where an object hides behind something else and temporarily loses its motion signal.

## 4. Data Collection Plan
- **Dataset source:** We are primarily using the official DAVIS 2016 dataset benchmark.
- **Dataset size:** It comes with 50 high-definition video sequences and over 3,400 frames.
- **Access and licensing:** The dataset is open-access and freely available for academic work on their official website.
- **Preprocessing:** Running heavy flow models on HD video is too expensive, so we will downscale the frames to 480p first. Then, we will normalize the (dx, dy) motion vectors so our region growing algorithm can process them consistently. Train/val splitting will follow the official DAVIS splits.
- **Supplementary data:** If our timeline permits, we want to test a few sequences from SegTrack v2. It has lower resolution but much more cluttered backgrounds, which is a great stress test for our algorithm.

## 5. Example Model Input / Output Items
| Image / Input ID | Input Description / Question | Expected Output / Answer | Concept / Facet Tested |
| :--- | :--- | :--- | :--- |
| davis_car_seq | Two consecutive frames of a car moving fast. | A binary mask highlighting just the car. | Core motion grouping |
| camouflaged_animal | An animal blending into the forest background. | Mask generated purely from movement. | Appearance vs. Motion |
| shaky_camera_04 | A scene where the camera itself is moving. | The actual moving object isolated from background shift. | Ego-motion handling |
| segtrack_birdfall | Two frames of a small bird falling in a cluttered background. | Mask highlighting the bird separating from similarly colored trees. | Small object motion tracking |
| davis_motocross | Fast-moving dirt bike causing motion blur. | A cohesive mask encompassing the rider and the bike despite blur. | Robustness to motion blur |

## 6. Selected Model and Rationale
- **Repository / checkpoint:** https://github.com/drinkingcoder/FlowFormer-Official (FlowFormer Checkpoint)
- **Key capabilities:** FlowFormer uses a Transformer architecture to estimate optical flow. Older CNNs usually create blurry edges around moving objects, but FlowFormer gives really sharp flow boundaries. Since our region growing algorithm relies on picking reliable "seed" pixels to start clustering, having clean and sharp flow data is an absolute must. In addition, FlowFormer’s global receptive field is our primary defense against the aperture problem.
- **Compute resources:** Google Colab (T4 GPUs) and the ITU HPC cluster.
- **Evaluation strategy:** Zero-shot evaluation on extracted optical flows.
- **Fine-tuning (if applicable):** Frozen backbone. Our research focus is on the Region Growing algorithm, so from a generalization and computational efficiency point of view, keeping the transformer model constant is in our interest.

## 7. Expected Challenges and Risks
- **Limited or noisy data availability**
  *Mitigation:* We will set a strict flow magnitude threshold (M(x, y) = sqrt(dx^2 + dy^2)) so seeds are only planted in areas with very obvious movement.
- **Domain shift between pretrained model distribution and your target dataset**
  *Mitigation:* We might introduce a secondary check using color similarity just at the edges to keep the borders tight and prevent boundary bleeding.
- **Compute constraints (VRAM, training time, Colab session limits)**
  *Mitigation:* Flow estimation is notoriously slow; if we run out of memory or time, we will process every second or third frame instead of every single one.
- **Metric reproducibility — differences in evaluation scripts across papers**
  *Mitigation:* We will use the official DAVIS evaluation toolkit (Jaccard Index, F-measure) to ensure our metrics are standard and reproducible.
- **Need for expert annotation or ground-truth verification**
  *Mitigation:* We will rely strictly on the pre-annotated, high-quality ground truths provided by the DAVIS and SegTrack v2 datasets for all evaluations.

## 8. Preliminary Timeline
| Phase / Weeks | Tasks | Deliverable |
| :--- | :--- | :--- |
| Weeks 10-11 | Environment setup; model checkpoint download; dataset download and integrity check; first forward pass | Working notebook with baseline output and raw flow maps |
| Weeks 11-12 | Data preprocessing pipeline; train/val/test split; write the core Region Growing code based on vector similarity | Baseline metric scores (table) and first messy segmentation masks |
| Weeks 12-13 | Tune the thresholds; try to fix the boundary leaking issues; ablation runs | Training logs; loss and metric curves; cleaner masks on validation set |
| Weeks 13-14 | Full benchmark evaluation on DAVIS; qualitative result visualizations; error analysis | Final report draft and final metric tables |
| Weeks 14-15 | Final results; report writing; code cleanup and demo preparation | Final presentation + PDF Report & code repository |

## 9. References 
- Kirillov, A. et al. (2023). Segment Anything. ICCV 2023. https://github.com/facebookresearch/segment-anything
- Ma, J. et al. MedSAM – Decoder fine-tuning reference. https://github.com/bowang-lab/MedSAM
- Ranftl, R. et al. (2020). Towards Robust Monocular Depth Estimation (MiDaS). IEEE TPAMI. https://github.com/isl-org/MiDaS
- Yang, L. et al. (2024). Depth Anything v2. https://github.com/DepthAnything/Depth-Anything-V2
- Teed, Z. & Deng, J. (2020). RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. ECCV 2020. https://github.com/princeton-vl/RAFT
- Xu, H. et al. (2022). GMFlow: Learning Optical Flow via Global Matching. CVPR 2022. https://github.com/haofeixu/gmflow
- Lindenberger, P. et al. (2023). LightGlue: Local Feature Matching at Light Speed. https://github.com/cvg/LightGlue
- Ravi, N. et al. (2024). SAM 2: Segment Anything in Images and Videos. https://github.com/facebookresearch/sam2
- Yan, S. et al. (2018). Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. AAAI 2018.
- Fan, H. et al. (2020). PySlowFast. https://github.com/facebookresearch/pytorchvideo
- Huang, Z., et al. (2022). FlowFormer: A Transformer Architecture for Optical Flow. ECCV 2022.
- Perazzi, F., et al. (2016). A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation. CVPR 2016.
- Adams, R., & Bischof, L. (1994). Seed-region growing. IEEE TPAMI.
