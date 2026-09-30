import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

import reportlab
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

# Define paths
DOCX_PATH = r"c:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\BoneMambaFormer_Methodology_Section.docx"
PDF_PATH = r"c:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\BoneMambaFormer_Methodology_Section.pdf"
AUG_IMAGE_PATH = r"c:\Users\admin_fix\Downloads\BONE CANCER ICIMCPS\figures\augmentation_samples.png"

print("Starting document generation script...")

# Helper functions for python-docx styling
def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_box_paragraph(doc, text, title=""):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    # Border styling
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="1B365D"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    if title:
        run_title = p.add_run(title + "\n")
        run_title.bold = True
        run_title.font.name = "Arial"
        run_title.font.size = Pt(10)
        run_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    run_text = p.add_run(text)
    run_text.font.name = "Arial"
    run_text.font.size = Pt(9.5)
    run_text.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def build_docx():
    doc = docx.Document()
    
    # Page setup - 0.75 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        
    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Arial'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    
    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(2)
    r_title = p_title.add_run("BoneMambaFormer: High-Performance Adaptive Hybrid Architecture for Bone Cancer Radiograph Classification")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(18)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(18)
    r_sub = p_sub.add_run("SECTION 3: MATERIALS AND METHODS (ICIMCPS-2026 EXPERIMENTAL SPECIFICATION)")
    r_sub.font.name = "Arial"
    r_sub.font.size = Pt(11)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(0x55, 0x66, 0x77)
    
    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x2B, 0x54, 0x8A)

    def add_h3(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.italic = True
        r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    def add_p(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(10)
        return p

    def add_eq(text, eq_num=""):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.left_indent = Inches(0.4)
        r = p.add_run(text)
        r.font.name = "Cambria Math"
        r.font.size = Pt(10.5)
        r.font.italic = True
        if eq_num:
            r_num = p.add_run(f"\t\t({eq_num})")
            r_num.font.name = "Arial"
            r_num.font.size = Pt(10)
            r_num.font.italic = False
            r_num.font.bold = True

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = "Arial"
        r_pre.font.size = Pt(10)
        r_txt = p.add_run(text)
        r_txt.font.name = "Arial"
        r_txt.font.size = Pt(10)

    # 3.1 Problem Formulation
    add_h1("3.1 Problem Formulation")
    add_p("We model planar radiograph bone cancer classification as a supervised binary decision task. Let the experimental dataset be defined as:")
    add_eq("D = {(x_i, y_i)}_{i=1}^N")
    add_p("where x_i in R^{H x W x C} denotes an input digital bone radiograph with spatial height H = 224 pixels, width W = 224 pixels, and C = 3 color channels (RGB). The ground-truth categorical label y_i in {0, 1} corresponds to the binary clinical decision space:")
    add_eq("y_i = 0 if Cancer (Malignant Pathological Lesion),  1 if Normal (Non-Malignant / Intact Bone Structure)")
    add_p("The objective is to parameterize a non-linear mapping function f(x; theta) -> p_hat(y|x), governed by trainable parameters theta, that outputs a normalized conditional probability distribution over the binary target classes:")
    add_eq("p_hat(y|x) = [p(y=0 | x), p(y=1 | x)]^T in [0, 1]^2,   subject to sum_{k=0}^1 p(y=k | x) = 1.0", "1")
    add_p("The predicted scalar label y_hat in {0, 1} is determined via maximum a posteriori (MAP) inference over the network logits z = f(x; theta) in R^2:")
    add_eq("y_hat = argmax_{k in {0, 1}} p(y=k | x) = argmax_{k in {0, 1}} ( exp(z_k) / sum_{j=0}^1 exp(z_j) )", "2")
    add_p("Parameter optimization is conducted via empirical risk minimization over the training partition D_train using Cross-Entropy loss with label smoothing (epsilon = 0.05) and L2 weight-decay regularization:")
    add_eq("theta* = argmin_{theta} (1 / |D_train|) sum_{i=1}^{|D_train|} L_CE(y_i, f(x_i; theta); epsilon) + lambda ||theta||_2^2", "3")
    add_p("where lambda = 1e-4 represents the weight decay hyperparameter.")

    # 3.2 Dataset and Data Integrity Audit
    add_h1("3.2 Dataset and Data Integrity Audit")
    add_h2("3.2.1 Raw Dataset Specification")
    add_p("The raw dataset supplied for this study comprises N_raw = 8,810 planar bone X-ray radiographs stored as RGB JPEG files at an uncompressed spatial resolution of 640 x 640 pixels. Ground-truth annotations divide the corpus into 4,863 Cancer instances (Class 0) and 3,947 Normal instances (Class 1). The raw corpus was initially arranged into three standard sub-directories:")
    add_bullet("Raw Training Set: ", "7,056 images (3,081 Cancer, 3,975 Normal)")
    add_bullet("Raw Validation Set: ", "882 images (398 Cancer, 484 Normal)")
    add_bullet("Raw Test Set: ", "872 images (384 Cancer, 488 Normal)")

    add_h2("3.2.2 Data Integrity Audit Rationale")
    add_p("In deep learning for medical imaging, pre-partitioned public datasets frequently contain duplicate or near-duplicate radiograph files distributed across separate directory splits. When identical or redundant image files reside in both training and evaluation splits, deep models can memorize instance-specific noise, fine textures, or non-pathological artifacts. This introduces severe cross-split data leakage, producing artificially inflated validation and test metrics that fail to reflect real-world generalization. To ensure strict evaluation integrity, an automated, non-destructive data integrity audit was executed prior to any modeling.")

    # 3.3 Duplicate Detection
    add_h1("3.3 Duplicate Detection and Leakage-Controlled Dataset Construction")
    add_h2("3.3.1 Cryptographic Hash Audit Protocol")
    add_p("To detect duplicate image files without relying on file names or directory metadata, an automated cryptographic audit was performed across all 8,810 raw images. For every image file x_i, a 128-bit MD5 message-digest hash H(x_i) in {0, 1}^{128} was computed over its raw byte sequence:")
    add_eq("H(x_i) = MD5(bytes(x_i))", "4")
    add_p("Hash equality (H(x_i) = H(x_j)) establishes bitwise pixel identity under the implemented procedure. Images sharing identical MD5 hashes were grouped into duplicate equivalence classes to map both intra-split duplicates and cross-split leakage boundaries.")

    add_h2("3.3.2 Audit Findings")
    add_p("The audit processed 8,810 raw images and identified exactly 7,751 unique MD5 hashes, revealing significant duplication in the uncurated corpus:")
    add_bullet("Single-Instance Hashes: ", "6,692 hashes corresponded to unique, single-file images.")
    add_bullet("Intra-Split Duplicate Groups: ", "714 hash groups comprised duplicate images located within the same directory split.")
    add_bullet("Cross-Split Duplicate Hash Groups (Leakage): ", "345 hash groups spanned across original split boundaries (168 Train<->Val, 161 Train<->Test, 16 Val<->Test).")
    add_bullet("Label Consistency Verification: ", "All 345 cross-split duplicate groups exhibited 100% ground-truth label agreement (zero label conflicts occurred).")

    add_h2("3.3.3 Construction of the derived_clean Protocol")
    add_p("To eliminate cross-split data leakage, we constructed a deduplicated split protocol designated as derived_clean (results/derived_leakage_clean_split.json). Under this protocol, every unique image hash is assigned exclusively to a single dataset split. For each duplicate hash group spanning across partitions, all instances were consolidated into the training set, while validation and test partitions were purged of any overlapping hashes.")
    add_p("The resulting derived_clean experimental partition comprises:")
    add_bullet("Clean Training Set: ", "7,417 images (3,082 Cancer, 4,335 Normal)")
    add_bullet("Clean Validation Set: ", "698 images (398 Cancer, 300 Normal)")
    add_bullet("Untouched Clean Test Set: ", "695 images (383 Cancer, 312 Normal)")
    add_p("This protocol guarantees zero cryptographic image-hash overlap across training, validation, and test partitions (H(D_train) cap H(D_val) = empty, H(D_train) cap H(D_test) = empty, H(D_val) cap H(D_test) = empty).")

    add_h2("3.3.4 Patient-Level Independence Limitation")
    add_p("Because patient identification tags and clinical case numbers were absent from the source dataset metadata, deduplication was performed at the cryptographic file-hash level. While the derived_clean protocol eliminates identical image pixel reuse across splits, patient-level independence cannot be formally guaranteed without explicit patient identifiers. This limitation is explicitly acknowledged.")

    # 3.4 Data Partitioning
    add_h1("3.4 Data Partitioning and Experimental Split Protocol")
    add_p("All experimental evaluations, ablation studies, and baseline comparisons in the Next-Gen BoneMambaFormer project strictly adhere to the derived_clean split protocol. Table 1 summarizes the sample counts across both partition protocols.")

    # Table 1
    p_t1 = doc.add_paragraph()
    p_t1.paragraph_format.space_before = Pt(8)
    p_t1.paragraph_format.space_after = Pt(4)
    r_t1 = p_t1.add_run("Table 1: Dataset Partitioning and Sample Counts across Raw and Derived-Clean Protocols.")
    r_t1.bold = True
    r_t1.font.size = Pt(9.5)
    
    table1 = doc.add_table(rows=3, cols=6)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1_headers = ["Partition Protocol", "Train Set", "Validation Set", "Test Set (Untouched)", "Total Images", "Cross-Split Overlaps"]
    t1_data = [
        ["Raw Supplied Split", "7,056", "882", "872", "8,810", "345 Hash Groups (Severe Leakage)"],
        ["derived_clean Protocol", "7,417", "698", "695", "8,810", "0 Hash Groups (Leakage-Clean)"]
    ]
    
    for c_idx, h in enumerate(t1_headers):
        cell = table1.cell(0, c_idx)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
    for r_idx, row in enumerate(t1_data):
        bg = "F9FAFC" if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row):
            cell = table1.cell(r_idx + 1, c_idx)
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            p = cell.paragraphs[0]
            if c_idx in [1, 2, 3, 4]:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            if r_idx == 1:
                r.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 3.5 Image Preprocessing
    add_h1("3.5 Image Preprocessing")
    add_p("Input images are processed through a deterministic preprocessing pipeline (src/preprocessing.py) to standardize spatial dimensions and intensity distributions prior to tensor formatting.")
    add_h2("3.5.1 Spatial Resizing")
    add_p("Raw 640 x 640 radiographs are resized to the target resolution of 224 x 224 pixels using bilinear interpolation:")
    add_eq("x_resized = Resize_bilinear(x_raw, (224, 224))")

    add_h2("3.5.2 Contrast Enhancement")
    add_p("To enhance low-contrast osseous structures and accentuate pathological cortical breaks or osteolytic lesions, contrast enhancement is applied using PIL ImageOps.autocontrast with a 1% histogram cutoff, followed by linear contrast scaling by a factor of 1.1:")
    add_eq("x_contrast = Enhance_contrast(AutoContrast(x_resized, cutoff=1), factor=1.1)", "5")
    add_p("This operation sharpens trabecular lines and tumor boundaries while avoiding non-linear high-frequency noise.")

    add_h2("3.5.3 Tensor Conversion and Normalization")
    add_p("Enhanced images are converted to floating-point tensors in the range [0.0, 1.0] and normalized using standard ImageNet channel-wise mean (mu) and standard deviation (sigma) vectors:")
    add_eq("mu = [0.485, 0.456, 0.406],   sigma = [0.229, 0.224, 0.225]")
    add_eq("x_norm^{(c)} = (x_contrast^{(c)} - mu_c) / sigma_c,   forall c in {1, 2, 3}", "6")

    # 3.6 Data Augmentation
    add_h1("3.6 Training-Time Data Augmentation")
    add_p("To prevent overfitting and simulate clinical imaging variations (such as patient positioning shifts, X-ray beam angle variations, and anatomical scale changes), a controlled data augmentation suite (src/augmentation.py) is applied strictly to training samples.")
    add_bullet("Elastic Deformation: ", "Simulates local soft-tissue and anatomical warping using Gaussian-filtered random displacement fields (Simard et al., 2003) with alpha = 10.0, sigma = 3.0, and probability p = 0.3.")
    add_bullet("Random Affine Rotation: ", "Applies random spatial rotation within theta_rot in [-15 deg, +15 deg] around the image center.")
    add_bullet("Random Horizontal Flip: ", "Applies left-right reflection with probability p = 0.5. Vertical flipping is explicitly disabled to preserve top-bottom anatomical orientation.")
    add_bullet("Random Rescaling and Zooming: ", "Rescales image dimensions by uniform scale factor s in [0.9, 1.1] with bilinear interpolation, followed by center cropping or zero-padding to 224 x 224.")
    add_bullet("Intensity Variation (Color Jitter): ", "Applies multiplicative brightness jitter b in [0.9, 1.1] and contrast jitter c in [0.9, 1.1].")
    add_p("Validation and test image pipelines remain 100% deterministic, executing only spatial resizing, contrast enhancement, and ImageNet normalization.")

    if os.path.exists(AUG_IMAGE_PATH):
        doc.add_paragraph().paragraph_format.space_before = Pt(6)
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(AUG_IMAGE_PATH, width=Inches(5.8))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(10)
        r_cap = p_cap.add_run("Figure 1: Training-time medical image augmentation pipeline demonstrating elastic deformation, affine rotation, flipping, and intensity jitter.")
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True

    # 3.7 Architecture Overview
    add_h1("3.7 Overview of the Proposed BoneMambaFormer Architecture")
    add_p("The proposed BoneMambaFormer architecture is a tri-branch hybrid network designed to synergize three complementary visual representation paradigms:")
    add_bullet("MobileNetV2 CNN Branch: ", "Lightweight depthwise separable convolutional processing for localized edge, texture, and bone boundary extraction.")
    add_bullet("Swin-Tiny Vision Transformer Branch: ", "Hierarchical Vision Transformer with shifted-window self-attention for multi-scale regional context modeling.")
    add_bullet("Mamba Selective State-Space Branch: ", "Linear-time sequential scanning over spatial patch sequences for long-range spatial dependency modeling.")
    add_p("Feature vectors extracted from all three backbones are projected into a unified 256-dimensional feature space, dynamically weighted via an Adaptive Softmax Branch Attention module, fused into a joint representation vector, and passed to a feature-refinement classification head.")

    # 3.8 MobileNetV2
    add_h1("3.8 MobileNetV2 CNN Branch")
    add_h2("3.8.1 Architectural Components")
    add_p("The convolutional branch utilizes an ImageNet-pretrained MobileNetV2 architecture (src/models/cnn.py) structured around inverted residual blocks with depthwise separable convolutions:")
    add_bullet("Expansion Layer: ", "1 x 1 pointwise convolution expands lower-dimensional input channels by expansion factor t = 6.")
    add_bullet("Depthwise Convolution: ", "3 x 3 spatial depthwise convolution applies spatial filtering independently per channel.")
    add_bullet("Projection Layer (Linear Bottleneck): ", "1 x 1 pointwise convolution projects features back to lower-dimensional space without non-linear activation.")

    add_h2("3.8.2 Branch Representation and Parameters")
    add_p("For input tensor x in R^{B x 3 x 224 x 224}, MobileNetV2 extracts a 1280-dimensional global feature vector h_CNN in R^{B x 1280} following global average pooling:")
    add_eq("h_CNN = MobileNetV2_backbone(x)")
    add_p("The standalone MobileNetV2 classification model (Config M1) contains 2,549,442 trainable parameters (~2.55M).")

    # 3.9 Swin-Tiny
    add_h1("3.9 Swin-Tiny Transformer Branch")
    add_h2("3.9.1 Architectural Components")
    add_p("The Vision Transformer branch incorporates an ImageNet-pretrained Swin-Tiny (swin_t) architecture (src/models/swin.py) to capture hierarchical spatial context:")
    add_bullet("Patch Partitioning: ", "Input images (224 x 224 x 3) are partitioned into 4 x 4 pixel patches, yielding 56 x 56 = 3,136 initial patch tokens of dimension C_swin = 96.")
    add_bullet("Local Window Self-Attention (W-MSA): ", "Self-attention is computed within local non-overlapping 7 x 7 token windows.")
    add_bullet("Shifted Window Self-Attention (SW-MSA): ", "Alternate layers shift window partitions by (3, 3) tokens to enable cross-window feature modeling.")
    add_bullet("Patch Merging: ", "Intermediate layers merge 2 x 2 neighbor token groups across four hierarchical stages (56x56, 28x28, 14x14, and 7x7).")

    add_h2("3.9.2 Branch Representation and Parameters")
    add_p("Swin-Tiny outputs a 768-dimensional feature vector h_Swin in R^{B x 768} following final layer normalization:")
    add_eq("h_Swin = SwinTiny_backbone(x)")
    add_p("The standalone Swin-Tiny model (Config M2) contains 27,746,056 trainable parameters (~27.75M).")

    # 3.10 Mamba
    add_h1("3.10 Mamba Selective State-Space Branch")
    add_h2("3.10.1 Image Patchification and Sequence Formulation")
    add_p("The Mamba branch converts 2D spatial radiographs into 1D token sequences:")
    add_eq("L = (224 / 32)^2 = 7 x 7 = 49 sequence tokens,   d_model = 128")
    add_p("A learnable 1D spatial position embedding matrix E_pos in R^{1 x 49 x 128} is added element-wise to form initial sequence tensor X_0.")

    add_h2("3.10.2 Pure PyTorch Selective State-Space Block")
    add_p("The token sequence is processed through N_blocks = 2 stacked Mamba blocks (PurePyTorchMambaBlock). Each block models sequence interactions via continuous-time state-space matrices (A, B, C, D) discretized by input-dependent step sizes Delta:")
    add_eq("h_t = A_bar_t * h_{t-1} + B_bar_t * x_t,   y_t = C_t * h_t + D * x_t", "7")
    add_p("Discretization uses Zero-Order Hold (ZOH): A_bar_t = exp(Delta_t * A), B_bar_t = (Delta_t * A)^{-1} (exp(Delta_t * A) - I) * (Delta_t * B_t). The selective mechanism parameterizes step sizes and matrices dynamically based on input sequence x_t.")

    add_h2("3.10.3 JIT-Compiled Selective Scan Recurrence")
    add_p("To ensure high computational efficiency on desktop GPU hardware without requiring custom CUDA compilation extensions, the sequence scan loop is implemented using PyTorch TorchScript JIT compilation (@torch.jit.script in src/models/mamba.py). The selective recurrence incorporates numerical clamping ([-50.0, +50.0]) to ensure numerical stability during backpropagation. Following global mean pooling over sequence length L = 49, Mamba produces a 128-dimensional output vector h_Mamba in R^{B x 128}. The standalone Mamba model (Config M3) contains 661,060 trainable parameters (~0.66M).")

    # 3.11 Feature Projection
    add_h1("3.11 Branch Feature Projection")
    add_p("Because MobileNetV2 (1280D), Swin-Tiny (768D), and Mamba (128D) output feature vectors of different dimensionalities, each branch is mapped into a common D = 256-dimensional feature space using dedicated projection heads:")
    add_eq("f_CNN = Dropout_{0.2}(GELU(LayerNorm(W_CNN * h_CNN + b_CNN))) in R^{B x 256}", "8")
    add_eq("f_Swin = Dropout_{0.2}(GELU(LayerNorm(W_Swin * h_Swin + b_Swin))) in R^{B x 256}", "9")
    add_eq("f_Mamba = Dropout_{0.2}(GELU(LayerNorm(W_Mamba * h_Mamba + b_Mamba))) in R^{B x 256}", "10")
    add_p("The projected branch feature vectors are stacked along a new branch axis to form multi-branch feature tensor F in R^{B x 3 x 256}.")

    # 3.12 Branch Attention
    add_h1("3.12 Adaptive Softmax Branch Attention")
    add_h2("3.12.1 Attention Score Generation")
    add_p("For each projected feature vector f_k in R^{256} (k in {CNN, Swin, Mamba}), an unnormalized scalar attention score e_k is computed by a two-layer MLP:")
    add_eq("e_k = g(f_k) = W_2 * Tanh(W_1 * f_k + b_1) + b_2", "11")
    add_p("where W_1 in R^{64 x 256}, b_1 in R^{64}, W_2 in R^{1 x 64}, and b_2 in R^{1}.")

    add_h2("3.12.2 Softmax Normalization")
    add_p("The scalar scores e_k are normalized across the three branches using the Softmax function along the branch dimension:")
    add_eq("alpha_k = exp(e_k) / sum_{j in {CNN, Swin, Mamba}} exp(e_j)", "12")
    add_p("This enforces non-negativity (alpha_k >= 0) and unity sum (sum_k alpha_k = 1.0). The vector alpha = [alpha_CNN, alpha_Swin, alpha_Mamba]^T in R^{3 x 1} represents sample-specific learned branch contribution weights.")

    # 3.13 Feature Fusion
    add_h1("3.13 Attention-Based Feature Fusion")
    add_p("The final fused feature representation f_fused in R^{B x 256} is computed as the attention-weighted linear sum of the projected branch feature vectors:")
    add_eq("f_fused = sum_{k in {CNN, Swin, Mamba}} alpha_k * f_k = alpha_CNN * f_CNN + alpha_Swin * f_Swin + alpha_Mamba * f_Mamba", "13")
    add_h2("3.13.1 Fixed vs. Adaptive Fusion Formulation")
    add_p("The fusion mechanism supports two operational modes: Fixed Uniform Fusion (alpha_CNN = alpha_Swin = alpha_Mamba = 1/3) and Adaptive Attention Fusion (dynamic sample-specific weights alpha_k(x)).")

    # 3.14 Classification Head
    add_h1("3.14 Feature Refinement and Classification Head")
    add_p("The fused feature vector f_fused in R^{B x 256} is passed to a feature refinement and classification MLP head to generate binary class logits z in R^{B x 2}:")
    add_eq("h_refine = Dropout_{0.2}(GELU(LayerNorm(W_refine * f_fused + b_refine))) in R^{B x 128}")
    add_eq("z = W_cls * h_refine + b_cls in R^{B x 2}", "14")
    add_p("Predicted class probabilities are obtained via Softmax activation: p(y=k | x) = exp(z_k) / (exp(z_0) + exp(z_1)).")

    # 3.15 Standalone Baselines
    add_h1("3.15 Standalone Baseline Models")
    add_p("Three standalone baseline models were trained independently on the derived_clean training set:")
    add_bullet("Config M1 (MobileNetV2 Standalone): ", "MobileNetV2 backbone + 256D projection + classifier (2,549,442 trainable parameters).")
    add_bullet("Config M2 (Swin-Tiny Standalone): ", "Swin-Tiny backbone + 256D projection + classifier (27,746,056 trainable parameters).")
    add_bullet("Config M3 (Mamba Standalone): ", "2-block Mamba SSM backbone + 256D projection + classifier (661,060 trainable parameters).")

    # 3.16 Frozen Fusion Training
    add_h1("3.16 Frozen Attention Fusion Training")
    add_h2("3.16.1 Freezing Protocol")
    add_p("In Config M4 (Attention Fusion), backbone weights are loaded from best standalone checkpoints and frozen (partial_L / partial_theta_backbones = 0). Frozen backbone layers are maintained in evaluation mode (eval()) to preserve normalization and dropout statistics.")
    add_h2("3.16.2 Pre-Extracted Feature Cache Optimization")
    add_p("Feature vectors f_CNN, f_Swin, f_Mamba in R^{256} are pre-extracted across training and validation sets in a single pass. Fusion training is executed directly on cached feature tensors (N, 3, 256), reducing training epoch latency to <0.5 seconds while optimizing only 49,923 trainable fusion parameters.")

    # 3.17 End-to-End Hybrid
    add_h1("3.17 End-to-End Hybrid Optimization")
    add_h2("3.17.1 Differential Learning Rate Allocation")
    add_p("In Config M5 (End-to-End Hybrid), all components are jointly optimized using differential learning rates: backbone parameters (eta_backbone = 2e-5) and fusion/classifier parameters (eta_head = 1e-4).")
    add_h2("3.17.2 Parameter Accounting")
    add_p("The total parameter count for the Next-Gen MobileNetV2 End-to-End Hybrid architecture is 31,044,037 parameters (~31.04M), with 100% of parameters actively updated during joint optimization.")

    # 3.18 Optimization Config
    add_h1("3.18 Optimization and Training Configuration")
    add_p("Models were optimized using AdamW (beta1=0.9, beta2=0.999, weight decay lambda = 1e-4) and Cross-Entropy loss with label smoothing epsilon = 0.05. Learning rates were adjusted using a Cosine Annealing scheduler (eta_min = 1e-6) with Automatic Mixed Precision (AMP FP16). Table 2 summarizes settings across configurations.")

    # Table 2
    p_t2 = doc.add_paragraph()
    p_t2.paragraph_format.space_before = Pt(8)
    p_t2.paragraph_format.space_after = Pt(4)
    r_t2 = p_t2.add_run("Table 2: Training Configurations and Optimization Hyperparameters across Model Configurations.")
    r_t2.bold = True
    r_t2.font.size = Pt(9.5)

    table2 = doc.add_table(rows=9, cols=6)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2_headers = ["Setting / Parameter", "Config M1 (MNv2)", "Config M2 (Swin)", "Config M3 (Mamba)", "Config M4 (Fusion)", "Config M5 (Hybrid)"]
    t2_data = [
        ["Backbone Optimizer", "AdamW", "AdamW", "AdamW", "Frozen (LR=0)", "AdamW (LR=2e-5)"],
        ["Fusion/Head LR", "1e-3", "5e-4", "1e-3", "5e-4", "1e-4"],
        ["Weight Decay (lambda)", "1e-4", "1e-4", "1e-4", "1e-4", "1e-4"],
        ["Label Smoothing (epsilon)", "0.05", "0.05", "0.05", "0.05", "0.05"],
        ["Batch Size", "16", "16", "16", "32", "16"],
        ["Max Epochs", "20", "20", "20", "25", "20"],
        ["Early Stop Patience", "3", "3", "3", "3", "3"],
        ["Precision Mode", "AMP (FP16)", "AMP (FP16)", "FP32", "FP32 (Cached)", "AMP (FP16)"]
    ]

    for c_idx, h in enumerate(t2_headers):
        cell = table2.cell(0, c_idx)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for r_idx, row in enumerate(t2_data):
        bg = "F9FAFC" if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row):
            cell = table2.cell(r_idx + 1, c_idx)
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            p = cell.paragraphs[0]
            if c_idx > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.size = Pt(8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 3.19 Checkpoint Selection
    add_h1("3.19 Validation-Based Checkpoint Selection and Early Stopping")
    add_p("Model selection was monitored on validation accuracy (Acc_val). A checkpoint was saved whenever validation accuracy reached a new maximum. Training terminated if no improvement occurred after P = 3 consecutive epochs (Early Stopping). Upon termination, model weights were restored to the best validation checkpoint state.")

    # 3.20 Locked Test Evaluation
    add_h1("3.20 Locked Test Evaluation")
    add_p("Following checkpoint selection, model parameters and decision rules were permanently locked. The untouched 695-sample test set was evaluated in a single pass with fixed default decision threshold p_thresh = 0.50 without any post-hoc threshold tuning or hyperparameter adjustment.")

    # 3.21 Computational Measurement
    add_h1("3.21 Computational and Inference Measurement")
    add_p("Benchmarking was conducted on an NVIDIA GeForce RTX 3050 6GB Laptop GPU. Evaluated metrics included parameter counts (numel()), mean single-sample inference latency (ms/image), throughput (images/sec), and peak VRAM allocation (torch.cuda.max_memory_allocated()).")

    # 3.22 Ablation Methodology
    add_h1("3.22 Controlled Ablation Methodology")
    add_h2("3.22.1 Contextual Distinction of Historical Ablation Evidence")
    add_p("Historical Phase 9 ablations reported in project documentation were conducted on the legacy ResNet-18 hybrid baseline (~39.74M parameters) to establish architectural proof of concept. These historical ablation results are strictly distinguished from the Next-Gen MobileNetV2 system (~31.04M parameters).")

    add_h2("3.22.2 Ablation Experimental Variants")
    add_p("Ablations evaluated fusion mode variations (Adaptive Attention vs. Fixed Equal Weighting 1/3) and branch-masking variants (setting f_CNN = 0, f_Mamba = 0, or f_Swin = 0 at inference time) on untouched test data.")

    add_h2("3.22.3 Probability-Differentiation Audit")
    add_p("A probability-differentiation audit verified that non-zero KL divergence and probability shifts occurred across masked variants, confirming active model computation.")

    # 3.23 Reproducibility
    add_h1("3.23 Reproducibility and Implementation Details")
    add_p("The global random seed was fixed to 42 across Python, NumPy, PyTorch CPU, and PyTorch CUDA modules. PyTorch CUDNN backends were configured for deterministic operation (torch.backends.cudnn.deterministic = True). All model checkpoints, execution histories, confusion matrices, and metrics JSON packages are archived for complete scientific auditability.")

    # Callout Box
    add_box_paragraph(
        doc,
        "The complete experimental pipeline, derived_clean dataset partitions, training logs, checkpoint weights, and evaluation artifacts are locked and fully reproducible under global seed 42 in the project repository.",
        title="REPRODUCIBILITY & ARTIFACT VERIFICATION NOTICE"
    )

    doc.save(DOCX_PATH)
    print(f"Successfully generated DOCX file at: {DOCX_PATH}")

# Helper canvas for ReportLab Page Numbering
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "BoneMambaFormer: Materials and Methods")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 742, 612 - 54, 742)
            
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 36, page_str)
        self.drawString(54, 36, "ICIMCPS-2026 Scientific Manuscript Section")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 48, 612 - 54, 48)
        
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        alignment=1, # Center
        textColor=colors.HexColor('#1B365D'),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        alignment=1, # Center
        textColor=colors.HexColor('#556677'),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1B365D'),
        spaceBefore=12,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#2B548A'),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#222222'),
        spaceAfter=5
    )

    eq_style = ParagraphStyle(
        'Eq_Custom',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=9.5,
        leading=13,
        leftIndent=20,
        textColor=colors.HexColor('#111111'),
        spaceBefore=3,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        leftIndent=15,
        firstLineIndent=-10,
        textColor=colors.HexColor('#222222'),
        spaceAfter=3
    )

    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor('#444444'),
        spaceBefore=4,
        spaceAfter=8
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("BoneMambaFormer: High-Performance Adaptive Hybrid Architecture for Bone Cancer Radiograph Classification", title_style))
    story.append(Paragraph("SECTION 3: MATERIALS AND METHODS (ICIMCPS-2026 EXPERIMENTAL SPECIFICATION)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceAfter=10))

    # 3.1
    story.append(Paragraph("3.1 Problem Formulation", h1_style))
    story.append(Paragraph("We model planar radiograph bone cancer classification as a supervised binary decision task. Let the experimental dataset be defined as:", body_style))
    story.append(Paragraph("<i>D = {(x_i, y_i)}_{i=1}^N</i>", eq_style))
    story.append(Paragraph("where <i>x_i &isin; &mathbb;R}^{H &times; W &times; C}</i> denotes an input digital bone radiograph with spatial height <i>H = 224</i> pixels, width <i>W = 224</i> pixels, and <i>C = 3</i> color channels (RGB). The ground-truth categorical label <i>y_i &isin; {0, 1}</i> corresponds to the binary clinical decision space:", body_style))
    story.append(Paragraph("<i>y_i = 0 (Cancer: Malignant Pathological Lesion), &nbsp; 1 (Normal: Non-Malignant / Intact Bone Structure)</i>", eq_style))
    story.append(Paragraph("The objective is to parameterize a non-linear mapping function <i>f(x; &theta;) &rarr; p&#770;(y|x)</i>, governed by trainable parameters <i>&theta;</i>, that outputs a normalized conditional probability distribution over target classes:", body_style))
    story.append(Paragraph("<b>p&#770;(y|x) = [p(y=0 | x), &nbsp; p(y=1 | x)]<sup>T</sup> &isin; [0, 1]<sup>2</sup>, &nbsp; subject to &sum;<sub>k=0</sub><sup>1</sup> p(y=k | x) = 1.0</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1)", eq_style))
    story.append(Paragraph("The predicted scalar label <i>y&#770; &isin; {0, 1}</i> is determined via maximum a posteriori (MAP) inference over the network logits <i>z = f(x; &theta;) &isin; &mathbb;R}<sup>2</sup></i>:", body_style))
    story.append(Paragraph("<b>y&#770; = argmax<sub>k &isin; {0, 1}</sub> p(y=k | x) = argmax<sub>k &isin; {0, 1}</sub> ( exp(z_k) / &sum;<sub>j=0</sub><sup>1</sup> exp(z_j) )</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2)", eq_style))
    story.append(Paragraph("Parameter optimization is conducted via empirical risk minimization over training partition <i>D<sub>train</sub></i> using Cross-Entropy loss with label smoothing (<i>&epsilon; = 0.05</i>) and <i>L<sub>2</sub></i> weight decay:", body_style))
    story.append(Paragraph("<b>&theta;* = argmin<sub>&theta;</sub> (1 / |D<sub>train</sub>|) &sum;<sub>i=1</sub><sup>|D<sub>train</sub>|</sup> L<sub>CE</sub>(y_i, f(x_i; &theta;); &epsilon;) + &lambda; ||&theta;||<sub>2</sub><sup>2</sup></b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (3)", eq_style))

    # 3.2
    story.append(Paragraph("3.2 Dataset and Data Integrity Audit", h1_style))
    story.append(Paragraph("3.2.1 Raw Dataset Specification", h2_style))
    story.append(Paragraph("The raw dataset supplied for this study comprises <i>N<sub>raw</sub> = 8,810</i> planar bone X-ray radiographs stored as RGB JPEG files at an uncompressed spatial resolution of <i>640 &times; 640</i> pixels. Ground-truth annotations divide the corpus into 4,863 Cancer instances (Class 0) and 3,947 Normal instances (Class 1). The raw corpus was initially arranged into three standard sub-directories:", body_style))
    story.append(Paragraph("&bull; <b>Raw Training Set:</b> 7,056 images (3,081 Cancer, 3,975 Normal)", bullet_style))
    story.append(Paragraph("&bull; <b>Raw Validation Set:</b> 882 images (398 Cancer, 484 Normal)", bullet_style))
    story.append(Paragraph("&bull; <b>Raw Test Set:</b> 872 images (384 Cancer, 488 Normal)", bullet_style))

    story.append(Paragraph("3.2.2 Data Integrity Audit Rationale", h2_style))
    story.append(Paragraph("In deep learning for medical imaging, pre-partitioned public datasets frequently contain duplicate or near-duplicate radiograph files distributed across separate directory splits. When identical image files reside in both training and evaluation splits, deep models can memorize instance-specific noise, fine textures, or non-pathological artifacts. This introduces severe cross-split data leakage, producing artificially inflated validation and test metrics. To ensure strict evaluation integrity, an automated data integrity audit was executed prior to any modeling.", body_style))

    # 3.3
    story.append(Paragraph("3.3 Duplicate Detection and Leakage-Controlled Dataset Construction", h1_style))
    story.append(Paragraph("3.3.1 Cryptographic Hash Audit Protocol", h2_style))
    story.append(Paragraph("To detect duplicate image files without relying on file names or directory metadata, an automated cryptographic audit was performed across all 8,810 raw images. For every image file <i>x_i</i>, a 128-bit MD5 message-digest hash <i>H(x_i) &isin; {0, 1}<sup>128</sup></i> was computed over its raw byte sequence:", body_style))
    story.append(Paragraph("<b>H(x_i) = MD5(bytes(x_i))</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (4)", eq_style))

    story.append(Paragraph("3.3.2 Audit Findings", h2_style))
    story.append(Paragraph("The audit identified exactly 7,751 unique MD5 hashes among 8,810 raw images:", body_style))
    story.append(Paragraph("&bull; <b>Single-Instance Hashes:</b> 6,692 unique single-file hashes.", bullet_style))
    story.append(Paragraph("&bull; <b>Intra-Split Duplicate Groups:</b> 714 duplicate hash groups within individual directory splits.", bullet_style))
    story.append(Paragraph("&bull; <b>Cross-Split Duplicate Groups (Leakage):</b> 345 hash groups spanned across split boundaries (168 Train<->Val, 161 Train<->Test, 16 Val<->Test).", bullet_style))
    story.append(Paragraph("&bull; <b>Label Verification:</b> All 345 cross-split duplicate groups exhibited 100% ground-truth label agreement.", bullet_style))

    story.append(Paragraph("3.3.3 Construction of the derived_clean Protocol", h2_style))
    story.append(Paragraph("To eliminate cross-split leakage, we constructed the <b>derived_clean</b> protocol. Every unique image hash was assigned exclusively to one split. Overlapping duplicate groups were consolidated into training, while validation and test sets were purged of overlapping hashes:", body_style))
    story.append(Paragraph("&bull; <b>Clean Training Set:</b> 7,417 images (3,082 Cancer, 4,335 Normal)", bullet_style))
    story.append(Paragraph("&bull; <b>Clean Validation Set:</b> 698 images (398 Cancer, 300 Normal)", bullet_style))
    story.append(Paragraph("&bull; <b>Untouched Clean Test Set:</b> 695 images (383 Cancer, 312 Normal)", bullet_style))

    story.append(Paragraph("3.3.4 Patient-Level Independence Limitation", h2_style))
    story.append(Paragraph("Because patient metadata was absent in the raw source dataset, deduplication was performed at the file-hash level. While cryptographic pixel reuse across splits is completely eliminated, patient-level independence cannot be formally guaranteed without explicit patient IDs.", body_style))

    # 3.4 Table 1
    story.append(Paragraph("3.4 Data Partitioning and Experimental Split Protocol", h1_style))
    story.append(Paragraph("Table 1 summarizes dataset partitioning across original and leakage-clean protocols.", body_style))
    
    t1_pdf_data = [
        [Paragraph("<b>Partition Protocol</b>", ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white, alignment=1)),
         Paragraph("<b>Train Set</b>", ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white, alignment=1)),
         Paragraph("<b>Val Set</b>", ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white, alignment=1)),
         Paragraph("<b>Test Set</b>", ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white, alignment=1)),
         Paragraph("<b>Total</b>", ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white, alignment=1)),
         Paragraph("<b>Cross-Split Leakage</b>", ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.white, alignment=1))],
        [Paragraph("Raw Supplied Split", ParagraphStyle('TD', fontName='Helvetica', fontSize=8)),
         Paragraph("7,056", ParagraphStyle('TD', fontName='Helvetica', fontSize=8, alignment=2)),
         Paragraph("882", ParagraphStyle('TD', fontName='Helvetica', fontSize=8, alignment=2)),
         Paragraph("872", ParagraphStyle('TD', fontName='Helvetica', fontSize=8, alignment=2)),
         Paragraph("8,810", ParagraphStyle('TD', fontName='Helvetica', fontSize=8, alignment=2)),
         Paragraph("345 Hash Groups (Severe Leakage)", ParagraphStyle('TD', fontName='Helvetica', fontSize=8))],
        [Paragraph("derived_clean Protocol", ParagraphStyle('TDB', fontName='Helvetica-Bold', fontSize=8)),
         Paragraph("<b>7,417</b>", ParagraphStyle('TDB', fontName='Helvetica-Bold', fontSize=8, alignment=2)),
         Paragraph("<b>698</b>", ParagraphStyle('TDB', fontName='Helvetica-Bold', fontSize=8, alignment=2)),
         Paragraph("<b>695</b>", ParagraphStyle('TDB', fontName='Helvetica-Bold', fontSize=8, alignment=2)),
         Paragraph("<b>8,810</b>", ParagraphStyle('TDB', fontName='Helvetica-Bold', fontSize=8, alignment=2)),
         Paragraph("<b>0 Hash Groups (Leakage-Clean)</b>", ParagraphStyle('TDB', fontName='Helvetica-Bold', fontSize=8))]
    ]

    t1_table = Table(t1_pdf_data, colWidths=[1.3*inch, 0.7*inch, 0.6*inch, 0.6*inch, 0.6*inch, 2.2*inch])
    t1_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1B365D')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#F9FAFC')),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#FFFFFF')),
    ]))
    story.append(t1_table)
    story.append(Paragraph("Table 1: Dataset Partitioning and Sample Counts across Raw and Derived-Clean Protocols.", caption_style))

    # 3.5 Preprocessing
    story.append(Paragraph("3.5 Image Preprocessing", h1_style))
    story.append(Paragraph("Images are processed through a deterministic pipeline (src/preprocessing.py): spatial resizing to 224 x 224, contrast enhancement via PIL ImageOps.autocontrast (1% cutoff, factor 1.1), and ImageNet RGB normalization.", body_style))
    story.append(Paragraph("<b>x<sub>contrast</sub> = Enhance<sub>contrast</sub>(AutoContrast(x<sub>resized</sub>, cutoff=1), factor=1.1)</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (5)", eq_style))
    story.append(Paragraph("<b>x<sub>norm</sub><sup>(c)</sup> = (x<sub>contrast</sub><sup>(c)</sup> - &mu;<sub>c</sub>) / &sigma;<sub>c</sub>, &nbsp;&nbsp; &mu;=[0.485, 0.456, 0.406], &sigma;=[0.229, 0.224, 0.225]</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (6)", eq_style))

    # 3.6 Augmentation & Image
    story.append(Paragraph("3.6 Training-Time Data Augmentation", h1_style))
    story.append(Paragraph("Training-only augmentations include Elastic Deformation (Simard et al., &alpha;=10.0, &sigma;=3.0, p=0.3), Random Affine Rotation (&plusmn;15&deg;), Random Horizontal Flip (p=0.5), Random Rescaling (0.9 to 1.1 scale), and Intensity Jitter (0.9 to 1.1). Validation and test pipelines remain 100% deterministic.", body_style))
    
    if os.path.exists(AUG_IMAGE_PATH):
        story.append(Spacer(1, 4))
        story.append(Image(AUG_IMAGE_PATH, width=5.8*inch, height=2.2*inch))
        story.append(Paragraph("Figure 1: Training-time medical image augmentation pipeline demonstrating elastic deformation, affine rotation, flipping, and intensity jitter.", caption_style))

    # 3.7 Architecture Overview
    story.append(Paragraph("3.7 Overview of the Proposed BoneMambaFormer Architecture", h1_style))
    story.append(Paragraph("BoneMambaFormer combines three complementary representation branches into a 256D common space, dynamically weighted via Adaptive Softmax Branch Attention and passed to a classification head:", body_style))
    story.append(Paragraph("&bull; <b>MobileNetV2 CNN Branch:</b> Lightweight local convolutional representation (2.55M params).", bullet_style))
    story.append(Paragraph("&bull; <b>Swin-Tiny Transformer Branch:</b> Hierarchical shifted-window self-attention (27.75M params).", bullet_style))
    story.append(Paragraph("&bull; <b>Mamba Selective State-Space Branch:</b> Sequential scanning over patch tokens (0.66M params).", bullet_style))

    # 3.8 to 3.10
    story.append(Paragraph("3.8 MobileNetV2 CNN Branch", h1_style))
    story.append(Paragraph("Uses inverted residual blocks with depthwise separable convolutions to extract a 1,280-dimensional global feature vector h_CNN.", body_style))

    story.append(Paragraph("3.9 Swin-Tiny Transformer Branch", h1_style))
    story.append(Paragraph("Processes non-overlapping 4 x 4 patches with local and shifted-window self-attention (W-MSA / SW-MSA) across 4 hierarchical stages, outputting 768-dimensional feature vector h_Swin.", body_style))

    story.append(Paragraph("3.10 Mamba Selective State-Space Branch", h1_style))
    story.append(Paragraph("Patchifies input into 49 tokens (7 x 7 grid, d_model=128), processed via 2 PurePyTorchMambaBlocks with TorchScript JIT-compiled selective scan recurrence:", body_style))
    story.append(Paragraph("<b>h<sub>t</sub> = A&#772;<sub>t</sub> h<sub>t-1</sub> + B&#772;<sub>t</sub> x<sub>t</sub>, &nbsp;&nbsp;&nbsp; y<sub>t</sub> = C<sub>t</sub> h<sub>t</sub> + D x<sub>t</sub></b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (7)", eq_style))

    # 3.11 to 3.14
    story.append(Paragraph("3.11 Branch Feature Projection", h1_style))
    story.append(Paragraph("Branch feature outputs (1280D, 768D, 128D) are mapped into a common 256D feature space via Linear + LayerNorm + GELU + Dropout(0.2) layers:", body_style))
    story.append(Paragraph("<b>f<sub>CNN</sub> &isin; &mathbb;R}<sup>B &times; 256</sup>, &nbsp; f<sub>Swin</sub> &isin; &mathbb;R}<sup>B &times; 256</sup>, &nbsp; f<sub>Mamba</sub> &isin; &mathbb;R}<sup>B &times; 256</sup></b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (8-10)", eq_style))

    story.append(Paragraph("3.12 Adaptive Softmax Branch Attention", h1_style))
    story.append(Paragraph("Unnormalized scalar attention scores e_k are computed via a 2-layer MLP (256D -> 64D -> 1D) and normalized via Softmax:", body_style))
    story.append(Paragraph("<b>e<sub>k</sub> = g(f<sub>k</sub>) = W<sub>2</sub> &middot; Tanh(W<sub>1</sub> f<sub>k</sub> + b<sub>1</sub>) + b<sub>2</sub></b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (11)", eq_style))
    story.append(Paragraph("<b>&alpha;<sub>k</sub> = exp(e<sub>k</sub>) / &sum;<sub>j</sub> exp(e<sub>j</sub>), &nbsp; subject to &sum;<sub>k</sub> &alpha;<sub>k</sub> = 1.0</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (12)", eq_style))

    story.append(Paragraph("3.13 Attention-Based Feature Fusion", h1_style))
    story.append(Paragraph("Fused representation: <b>f<sub>fused</sub> = &sum;<sub>k</sub> &alpha;<sub>k</sub> f<sub>k</sub> = &alpha;<sub>CNN</sub> f<sub>CNN</sub> + &alpha;<sub>Swin</sub> f<sub>Swin</sub> + &alpha;<sub>Mamba</sub> f<sub>Mamba</sub> &isin; &mathbb;R}<sup>B &times; 256</sup></b> (13)", eq_style))

    story.append(Paragraph("3.14 Feature Refinement and Classification Head", h1_style))
    story.append(Paragraph("Passed through FC(256->128) + LayerNorm + GELU + Dropout(0.2) + FC(128->2) to output binary class logits z in R^{B x 2} and Softmax probabilities.", body_style))

    # 3.15 to 3.17
    story.append(Paragraph("3.15 Standalone Baseline Models", h1_style))
    story.append(Paragraph("Three standalone models were trained independently: Config M1 (MobileNetV2, 2.55M params), Config M2 (Swin-Tiny, 27.75M params), and Config M3 (Mamba, 0.66M params).", body_style))

    story.append(Paragraph("3.16 Frozen Attention Fusion Training", h1_style))
    story.append(Paragraph("In Config M4, backbones are frozen (30.99M params frozen) and 256D branch features pre-extracted into cache memory. Training optimizes only 49,923 fusion and classification parameters.", body_style))

    story.append(Paragraph("3.17 End-to-End Hybrid Optimization", h1_style))
    story.append(Paragraph("In Config M5, all 31,044,037 parameters (~31.04M) are jointly trained end-to-end using differential learning rates: backbone LR = 2e-5, fusion/head LR = 1e-4.", body_style))

    # 3.18 Table 2
    story.append(Paragraph("3.18 Optimization and Training Configuration", h1_style))
    story.append(Paragraph("Table 2 lists the complete optimization and training hyperparameters across configurations.", body_style))

    t2_pdf_data = [
        [Paragraph("<b>Parameter / Setting</b>", ParagraphStyle('TH2', fontName='Helvetica-Bold', fontSize=8, textColor=colors.white)),
         Paragraph("<b>M1 (MNv2)</b>", ParagraphStyle('TH2', fontName='Helvetica-Bold', fontSize=8, textColor=colors.white, alignment=1)),
         Paragraph("<b>M2 (Swin)</b>", ParagraphStyle('TH2', fontName='Helvetica-Bold', fontSize=8, textColor=colors.white, alignment=1)),
         Paragraph("<b>M3 (Mamba)</b>", ParagraphStyle('TH2', fontName='Helvetica-Bold', fontSize=8, textColor=colors.white, alignment=1)),
         Paragraph("<b>M4 (Fusion)</b>", ParagraphStyle('TH2', fontName='Helvetica-Bold', fontSize=8, textColor=colors.white, alignment=1)),
         Paragraph("<b>M5 (Hybrid)</b>", ParagraphStyle('TH2', fontName='Helvetica-Bold', fontSize=8, textColor=colors.white, alignment=1))],
        [Paragraph("Backbone Optimizer", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5)),
         Paragraph("AdamW", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("AdamW", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("AdamW", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("Frozen", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("AdamW (2e-5)", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1))],
        [Paragraph("Fusion/Head LR", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5)),
         Paragraph("1e-3", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("5e-4", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("1e-3", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("5e-4", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("1e-4", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1))],
        [Paragraph("Weight Decay (&lambda;)", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5)),
         Paragraph("1e-4", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("1e-4", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("1e-4", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("1e-4", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("1e-4", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1))],
        [Paragraph("Label Smooth (&epsilon;)", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5)),
         Paragraph("0.05", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("0.05", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("0.05", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("0.05", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("0.05", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1))],
        [Paragraph("Batch Size", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5)),
         Paragraph("16", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("16", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("16", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("32", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("16", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1))],
        [Paragraph("Max Epochs", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5)),
         Paragraph("20", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("20", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("20", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("25", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("20", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1))],
        [Paragraph("Early Stop Patience", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5)),
         Paragraph("3", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("3", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("3", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("3", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("3", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1))],
        [Paragraph("Precision Mode", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5)),
         Paragraph("AMP FP16", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("AMP FP16", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("FP32", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("FP32 Cached", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1)),
         Paragraph("AMP FP16", ParagraphStyle('TD2', fontName='Helvetica', fontSize=7.5, alignment=1))]
    ]

    t2_table = Table(t2_pdf_data, colWidths=[1.8*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.2*inch])
    t2_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1B365D')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#F9FAFC')),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#FFFFFF')),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#F9FAFC')),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor('#FFFFFF')),
        ('BACKGROUND', (0,5), (-1,5), colors.HexColor('#F9FAFC')),
        ('BACKGROUND', (0,6), (-1,6), colors.HexColor('#FFFFFF')),
        ('BACKGROUND', (0,7), (-1,7), colors.HexColor('#F9FAFC')),
        ('BACKGROUND', (0,8), (-1,8), colors.HexColor('#FFFFFF')),
    ]))
    story.append(t2_table)
    story.append(Paragraph("Table 2: Training Configurations and Optimization Hyperparameters across Model Configurations.", caption_style))

    # 3.19 to 3.23
    story.append(Paragraph("3.19 Validation-Based Checkpoint Selection and Early Stopping", h1_style))
    story.append(Paragraph("Monitored validation accuracy (Acc_val). Early stopping triggers after P = 3 non-improving epochs. Best checkpoint restored before test evaluation.", body_style))

    story.append(Paragraph("3.20 Locked Test Evaluation", h1_style))
    story.append(Paragraph("Parameters and decision threshold (p_thresh = 0.50) permanently locked. Evaluated once on untouched 695 test samples.", body_style))

    story.append(Paragraph("3.21 Computational and Inference Measurement", h1_style))
    story.append(Paragraph("Evaluated on NVIDIA RTX 3050 6GB Laptop GPU: parameter counts, mean latency (ms/image), throughput (images/sec), peak VRAM.", body_style))

    story.append(Paragraph("3.22 Controlled Ablation Methodology", h1_style))
    story.append(Paragraph("Historical Phase 9 ablations (ResNet-18 baseline, ~39.74M params) provide component proof of concept and are distinguished from the Next-Gen MobileNetV2 architecture (~31.04M params). Tests evaluated dynamic attention vs equal weighting and inference branch-masking variants.", body_style))

    story.append(Paragraph("3.23 Reproducibility and Implementation Details", h1_style))
    story.append(Paragraph("Fixed global seed 42 across Python, NumPy, PyTorch CPU/CUDA. PyTorch CUDNN configured for deterministic execution. Artifacts, checkpoints, and logs are completely archived.", body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF file at: {PDF_PATH}")

if __name__ == "__main__":
    build_docx()
    build_pdf()
    print("All documents created successfully!")
