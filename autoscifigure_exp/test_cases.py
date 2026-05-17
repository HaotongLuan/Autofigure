"""
Test cases for automated scientific figure generation evaluation.

Each test case represents a real-world scientific figure that should be
generated from paper text excerpts. The cases span multiple figure types
and scientific domains to ensure comprehensive evaluation.
"""

from __future__ import annotations

import copy
from typing import Optional


# ---------------------------------------------------------------------------
# Test Case 01 – Transformer Architecture (model_architecture)
# ---------------------------------------------------------------------------

TC01 = {
    "id": "tc01",
    "title": "Transformer Encoder-Decoder Architecture",
    "figure_type": "model_architecture",
    "text": (
        "We adopt the Transformer architecture (Vaswani et al., 2017) for our "
        "sequence-to-sequence modeling task. The model consists of an encoder stack "
        "of N=6 identical layers and a decoder stack of N=6 identical layers. "
        "Each encoder layer is composed of two sub-layers: a multi-head self-attention "
        "mechanism with h=8 heads, followed by a position-wise fully connected "
        "feed-forward network with a hidden dimension of 2048 and ReLU activation. "
        "Residual connections wrap around each of these two sub-layers, followed by "
        "layer normalization. The decoder layer inserts a third sub-layer between the "
        "self-attention and feed-forward stages: a cross-attention (encoder-decoder "
        "attention) mechanism that attends to the output of the encoder stack. "
        "The self-attention sub-layer in the decoder is masked to prevent positions "
        "from attending to subsequent positions during training. Input tokens are first "
        "converted to continuous representations via a learned embedding layer of "
        "dimension d_model=512, and positional encodings using sine and cosine "
        "functions of different frequencies are added to these embeddings before "
        "they enter the encoder and decoder stacks. The output of the final decoder "
        "layer passes through a linear projection layer followed by a softmax function "
        "to produce probability distributions over the target vocabulary of size "
        "|V|=37000. During training, we use label smoothing of epsilon=0.1 and the "
        "Adam optimizer with beta_1=0.9, beta_2=0.98, and epsilon=10^-9, applying "
        "the learning rate schedule: lrate = d_model^{-0.5} * min(step_num^{-0.5}, "
        "step_num * warmup_steps^{-1.5}) with warmup_steps=4000."
    ),
    "ground_truth": {
        "key_entities": [
            "Input Embedding",
            "Positional Encoding",
            "Encoder Stack",
            "Multi-Head Self-Attention",
            "Feed-Forward Network",
            "Add & Norm",
            "Decoder Stack",
            "Masked Multi-Head Self-Attention",
            "Cross-Attention",
            "Linear Projection",
            "Softmax",
            "Output Probabilities",
        ],
        "key_relationships": [
            ("Input Embedding", "Positional Encoding"),
            ("Positional Encoding", "Encoder Stack"),
            ("Multi-Head Self-Attention", "Add & Norm"),
            ("Add & Norm", "Feed-Forward Network"),
            ("Feed-Forward Network", "Add & Norm"),
            ("Encoder Stack", "Cross-Attention"),
            ("Positional Encoding", "Decoder Stack"),
            ("Masked Multi-Head Self-Attention", "Add & Norm"),
            ("Cross-Attention", "Add & Norm"),
            ("Decoder Stack", "Linear Projection"),
            ("Linear Projection", "Softmax"),
            ("Softmax", "Output Probabilities"),
        ],
        "description": (
            "Left side: encoder with embedding + positional encoding feeding into "
            "N=6 encoder layers (each: self-attention -> add&norm -> FFN -> add&norm). "
            "Right side: decoder with embedding + positional encoding feeding into "
            "N=6 decoder layers (each: masked self-attention -> add&norm -> "
            "cross-attention -> add&norm -> FFN -> add&norm). Encoder output connects "
            "to decoder cross-attention. Final linear + softmax for output."
        ),
    },
}


# ---------------------------------------------------------------------------
# Test Case 02 – RAG System (architecture_diagram)
# ---------------------------------------------------------------------------

TC02 = {
    "id": "tc02",
    "title": "Retrieval-Augmented Generation (RAG) Pipeline",
    "figure_type": "architecture_diagram",
    "text": (
        "Our Retrieval-Augmented Generation system combines a dense passage retriever "
        "with a generative large language model to produce grounded, factually accurate "
        "responses. The pipeline begins when a user query enters the system. First, "
        "the query is encoded by a bi-encoder based on the Dense Passage Retriever "
        "(DPR) architecture, which maps the query into a 768-dimensional dense vector "
        "using a BERT-base encoder fine-tuned on MS MARCO. Simultaneously, our document "
        "corpus has been pre-indexed: each document chunk of 100 tokens is encoded by "
        "the same bi-encoder and stored in a FAISS inverted file index with product "
        "quantization for efficient approximate nearest-neighbor search. The top-k=5 "
        "document chunks with highest cosine similarity to the query vector are "
        "retrieved. These retrieved passages then pass through a re-ranking stage "
        "employing a cross-encoder (monoBERT) that computes relevance scores for each "
        "query-passage pair, producing a re-ordered list of the top-k'=3 most relevant "
        "passages. These three passages are concatenated with the original query using "
        "a prompt template that prepends the instruction: 'Answer the question based "
        "on the following context.' The augmented prompt is then fed into a decoder-only "
        "LLM (LLaMA-2-13B-chat) which generates the final answer autoregressively. "
        "A hallucination detector, implemented as a separate NLI model fine-tuned on "
        "ANLI, checks whether each generated sentence is entailed by the retrieved "
        "context. If any sentence is flagged as unsupported, the system appends a "
        "disclaimer to the output. The entire pipeline is orchestrated by a FastAPI "
        "server that exposes a single /generate endpoint, with Redis caching of "
        "frequently retrieved passages to reduce latency."
    ),
    "ground_truth": {
        "key_entities": [
            "User Query",
            "Query Encoder (DPR)",
            "Document Corpus",
            "FAISS Index",
            "Top-k Retrieval",
            "Cross-Encoder Re-ranker",
            "Prompt Template",
            "LLM Generator (LLaMA-2-13B)",
            "Hallucination Detector (NLI)",
            "Final Answer",
            "FastAPI Server",
            "Redis Cache",
        ],
        "key_relationships": [
            ("User Query", "Query Encoder (DPR)"),
            ("Document Corpus", "FAISS Index"),
            ("Query Encoder (DPR)", "Top-k Retrieval"),
            ("FAISS Index", "Top-k Retrieval"),
            ("Top-k Retrieval", "Cross-Encoder Re-ranker"),
            ("Cross-Encoder Re-ranker", "Prompt Template"),
            ("User Query", "Prompt Template"),
            ("Prompt Template", "LLM Generator (LLaMA-2-13B)"),
            ("LLM Generator (LLaMA-2-13B)", "Hallucination Detector (NLI)"),
            ("Cross-Encoder Re-ranker", "Hallucination Detector (NLI)"),
            ("Hallucination Detector (NLI)", "Final Answer"),
            ("Redis Cache", "Top-k Retrieval"),
        ],
        "description": (
            "Top-down flow: User Query -> DPR Query Encoder -> Top-k Retrieval (from "
            "FAISS-indexed Document Corpus) -> Cross-Encoder Re-ranker -> Prompt Template "
            "(merged with original query) -> LLaMA-2-13B Generator -> Hallucination "
            "Detector -> Final Answer. Side connections: Redis Cache feeds Top-k Retrieval; "
            "Cross-Encoder Re-ranker also feeds Hallucination Detector. FastAPI server "
            "wraps all components."
        ),
    },
}


# ---------------------------------------------------------------------------
# Test Case 03 – Scientific Experiment Workflow (flowchart)
# ---------------------------------------------------------------------------

TC03 = {
    "id": "tc03",
    "title": "Western Blot Protein Analysis Workflow",
    "figure_type": "flowchart",
    "text": (
        "We describe the standardized workflow for quantitative Western blot analysis "
        "of protein expression levels in HEK293T cell lysates. The procedure begins "
        "with cell culture and harvesting: HEK293T cells are grown to 80% confluence "
        "in DMEM supplemented with 10% fetal bovine serum, then washed twice with "
        "ice-cold PBS and lysed in RIPA buffer containing protease and phosphatase "
        "inhibitors. After centrifugation at 14,000g for 15 minutes at 4 degrees "
        "Celsius, the supernatant is collected and protein concentration is determined "
        "using the BCA assay with a BSA standard curve. A total of 30 micrograms of "
        "protein per lane is mixed with 4x Laemmli sample buffer and boiled at 95 "
        "degrees Celsius for 5 minutes. Samples are loaded onto a 10% SDS-PAGE gel "
        "and electrophoresed at 120V until the dye front reaches the bottom. Proteins "
        "are then transferred to a PVDF membrane (pre-activated in methanol) using a "
        "semi-dry transfer system at 25V for 30 minutes. The membrane is blocked with "
        "5% non-fat milk in TBST for 1 hour at room temperature, then incubated with "
        "primary antibody (anti-beta-actin, 1:5000 dilution) overnight at 4 degrees "
        "Celsius. After three washes with TBST (10 minutes each), the membrane is "
        "incubated with HRP-conjugated secondary antibody (1:10000 dilution) for 1 "
        "hour at room temperature. Following three additional TBST washes, the membrane "
        "is developed using enhanced chemiluminescence (ECL) substrate and imaged on "
        "a chemiluminescence imaging system. Densitometry analysis is performed using "
        "ImageJ software, normalizing band intensities to the beta-actin loading "
        "control. Each experiment is performed in biological triplicate (n=3) and "
        "statistical significance is assessed using a two-tailed Student's t-test "
        "with p < 0.05 considered significant."
    ),
    "ground_truth": {
        "key_entities": [
            "Cell Culture & Harvesting",
            "Protein Lysis (RIPA Buffer)",
            "Centrifugation (14,000g)",
            "BCA Protein Assay",
            "Sample Preparation (Laemmli Buffer)",
            "SDS-PAGE Electrophoresis",
            "Protein Transfer (PVDF Membrane)",
            "Membrane Blocking",
            "Primary Antibody Incubation",
            "Secondary Antibody Incubation",
            "ECL Detection",
            "Chemiluminescence Imaging",
            "Densitometry Analysis (ImageJ)",
            "Statistical Analysis (t-test)",
        ],
        "key_relationships": [
            ("Cell Culture & Harvesting", "Protein Lysis (RIPA Buffer)"),
            ("Protein Lysis (RIPA Buffer)", "Centrifugation (14,000g)"),
            ("Centrifugation (14,000g)", "BCA Protein Assay"),
            ("BCA Protein Assay", "Sample Preparation (Laemmli Buffer)"),
            ("Sample Preparation (Laemmli Buffer)", "SDS-PAGE Electrophoresis"),
            ("SDS-PAGE Electrophoresis", "Protein Transfer (PVDF Membrane)"),
            ("Protein Transfer (PVDF Membrane)", "Membrane Blocking"),
            ("Membrane Blocking", "Primary Antibody Incubation"),
            ("Primary Antibody Incubation", "Secondary Antibody Incubation"),
            ("Secondary Antibody Incubation", "ECL Detection"),
            ("ECL Detection", "Chemiluminescence Imaging"),
            ("Chemiluminescence Imaging", "Densitometry Analysis (ImageJ)"),
            ("Densitometry Analysis (ImageJ)", "Statistical Analysis (t-test)"),
        ],
        "description": (
            "Linear flowchart showing sequential steps of a Western blot experiment: "
            "starting from cell culture through protein extraction, quantification, "
            "gel electrophoresis, transfer, antibody probing, detection, imaging, "
            "and finally densitometry with statistical analysis. Each step flows to "
            "the next in a single downward chain."
        ),
    },
}


# ---------------------------------------------------------------------------
# Test Case 04 – Data Processing Pipeline (data_pipeline)
# ---------------------------------------------------------------------------

TC04 = {
    "id": "tc04",
    "title": "Large-Scale Genomics ETL Pipeline",
    "figure_type": "data_pipeline",
    "text": (
        "We built a scalable Extract-Transform-Load pipeline for processing whole-genome "
        "sequencing data from the UK Biobank cohort of approximately 500,000 individuals. "
        "The pipeline ingests raw FASTQ files from the sequencing facility via an S3 "
        "bucket, with each file averaging 90 GB for a 30x coverage genome. The first "
        "stage performs read quality assessment using FastQC v0.11.9, generating per-file "
        "reports that are aggregated into a summary dashboard. Reads failing the quality "
        "threshold (mean Phred score < 30) are filtered out. The surviving reads are "
        "then aligned to the GRCh38 reference genome using BWA-MEM with ALT-aware "
        "mapping, producing SAM files that are immediately converted to sorted BAM "
        "files via samtools sort. Duplicate reads are marked using GATK MarkDuplicates "
        "and base quality score recalibration (BQSR) is applied using known SNP sites "
        "from dbSNP build 151. Variant calling is performed by GATK HaplotypeCaller in "
        "GVCF mode, with joint genotyping across all samples using GATK GenotypeGVCFs. "
        "The resulting VCF files undergo hard filtering with the following criteria: "
        "QD < 2.0, FS > 60.0, MQ < 40.0, SOR > 3.0. Filtered variants are annotated "
        "using VEP (Variant Effect Predictor) with the Ensembl v105 database, adding "
        "consequence predictions, population frequencies from gnomAD, and ClinVar "
        "clinical significance. Finally, the annotated variants are loaded into a "
        "Hail matrix table stored in Google BigQuery, partitioned by chromosome, "
        "enabling interactive queries via a Jupyter notebook frontend. The entire "
        "pipeline is orchestrated by Apache Airflow, with each stage running on "
        "separate AWS EC2 spot instances, and intermediate files persisted to S3 "
        "with lifecycle policies moving them to Glacier after 30 days."
    ),
    "ground_truth": {
        "key_entities": [
            "S3 Input (FASTQ)",
            "FastQC Quality Assessment",
            "Quality Filtering",
            "BWA-MEM Alignment",
            "SAM-to-BAM Conversion",
            "MarkDuplicates",
            "BQSR Recalibration",
            "GATK HaplotypeCaller",
            "Joint Genotyping",
            "VCF Hard Filtering",
            "VEP Annotation",
            "Hail Matrix Table",
            "BigQuery Storage",
            "Jupyter Frontend",
            "Apache Airflow Orchestrator",
        ],
        "key_relationships": [
            ("S3 Input (FASTQ)", "FastQC Quality Assessment"),
            ("FastQC Quality Assessment", "Quality Filtering"),
            ("Quality Filtering", "BWA-MEM Alignment"),
            ("BWA-MEM Alignment", "SAM-to-BAM Conversion"),
            ("SAM-to-BAM Conversion", "MarkDuplicates"),
            ("MarkDuplicates", "BQSR Recalibration"),
            ("BQSR Recalibration", "GATK HaplotypeCaller"),
            ("GATK HaplotypeCaller", "Joint Genotyping"),
            ("Joint Genotyping", "VCF Hard Filtering"),
            ("VCF Hard Filtering", "VEP Annotation"),
            ("VEP Annotation", "Hail Matrix Table"),
            ("Hail Matrix Table", "BigQuery Storage"),
            ("BigQuery Storage", "Jupyter Frontend"),
        ],
        "description": (
            "Left-to-right or top-to-bottom ETL pipeline: S3 FASTQ ingestion -> "
            "FastQC -> Quality Filter -> BWA-MEM alignment -> SAM-to-BAM -> "
            "MarkDuplicates -> BQSR -> HaplotypeCaller -> Joint Genotyping -> "
            "VCF Filtering -> VEP Annotation -> Hail Matrix -> BigQuery -> Jupyter. "
            "Airflow orchestrator spans all stages."
        ),
    },
}


# ---------------------------------------------------------------------------
# Test Case 05 – Neural Network Training Loop (flowchart)
# ---------------------------------------------------------------------------

TC05 = {
    "id": "tc05",
    "title": "Neural Network Training Loop with Mixed Precision",
    "figure_type": "flowchart",
    "text": (
        "Our training procedure for the 1.3-billion parameter language model follows "
        "a carefully optimized pipeline. Training begins by initializing model "
        "parameters using a truncated normal distribution with standard deviation "
        "0.02, and the optimizer (AdamW with beta_1=0.9, beta_2=0.95, epsilon=1e-8, "
        "weight_decay=0.1) is instantiated. We use a cosine learning rate schedule "
        "with 2000 warmup steps, peaking at 3e-4 and decaying to 1e-5 over 300,000 "
        "total steps. The outer loop iterates over epochs (total of 3 epochs). Within "
        "each epoch, we iterate over batches of 2048 sequences each of length 2048 "
        "tokens, drawn from a shuffled dataset of 1.4 trillion tokens. For each "
        "batch, the forward pass is executed under automatic mixed precision (AMP) "
        "with the bfloat16 data type, computing the cross-entropy loss with z-loss "
        "regularization (coefficient 1e-4). The loss is then scaled by the gradient "
        "scaler and backpropagation computes gradients. Gradient clipping is applied "
        "with a maximum norm of 1.0 to prevent exploding gradients. The optimizer "
        "step updates the parameters, the learning rate scheduler advances one step, "
        "and the gradient scaler updates its internal scale factor. Every 100 steps, "
        "we evaluate the model on a held-out validation set of 5000 examples, computing "
        "perplexity and next-sentence-prediction accuracy. If validation perplexity "
        "does not improve for 5000 consecutive steps, an early-stopping trigger halts "
        "training. Model checkpoints are saved every 1000 steps to a shared NFS mount, "
        "with the top-3 checkpoints by validation perplexity retained. Additionally, "
        "gradient norms, loss values, and learning rate are logged to Weights & "
        "Biases every 10 steps for real-time monitoring. All training is distributed "
        "across 64 A100-80GB GPUs using FSDP (Fully Sharded Data Parallel) with "
        "activation checkpointing on the attention and FFN blocks."
    ),
    "ground_truth": {
        "key_entities": [
            "Parameter Initialization",
            "AdamW Optimizer",
            "Learning Rate Scheduler",
            "Epoch Loop",
            "Batch Data Loader",
            "Forward Pass (AMP bf16)",
            "Loss Computation",
            "Gradient Scaling",
            "Backpropagation",
            "Gradient Clipping",
            "Optimizer Step",
            "Validation Evaluation",
            "Early Stopping Check",
            "Model Checkpointing",
            "W&B Logging",
        ],
        "key_relationships": [
            ("Parameter Initialization", "AdamW Optimizer"),
            ("Epoch Loop", "Batch Data Loader"),
            ("Batch Data Loader", "Forward Pass (AMP bf16)"),
            ("Forward Pass (AMP bf16)", "Loss Computation"),
            ("Loss Computation", "Gradient Scaling"),
            ("Gradient Scaling", "Backpropagation"),
            ("Backpropagation", "Gradient Clipping"),
            ("Gradient Clipping", "Optimizer Step"),
            ("Optimizer Step", "Learning Rate Scheduler"),
            ("Learning Rate Scheduler", "Validation Evaluation"),
            ("Validation Evaluation", "Early Stopping Check"),
            ("Early Stopping Check", "Model Checkpointing"),
            ("Optimizer Step", "W&B Logging"),
            ("Model Checkpointing", "Epoch Loop"),
        ],
        "description": (
            "Training flowchart with a main loop (Epoch -> Batch -> Forward -> Loss "
            "-> Backward -> Clip -> Step -> LR Update). Branch for periodic validation "
            "with early stopping and checkpointing. W&B logging as side output. "
            "Loop-back from checkpoint to next epoch."
        ),
    },
}


# ---------------------------------------------------------------------------
# Test Case 06 – Microservice System Architecture (architecture_diagram)
# ---------------------------------------------------------------------------

TC06 = {
    "id": "tc06",
    "title": "E-Commerce Microservice Architecture",
    "figure_type": "architecture_diagram",
    "text": (
        "Our e-commerce platform employs a microservice architecture deployed on "
        "Kubernetes with 12 independent services. The system's entry point is an API "
        "Gateway implemented with Kong, which handles rate limiting, authentication "
        "via OAuth2 with Keycloak as the identity provider, and request routing. "
        "The User Service manages customer profiles and is backed by a PostgreSQL 14 "
        "database with read replicas for scaling read queries. The Product Catalog "
        "Service stores product information in MongoDB, with full-text search powered "
        "by an Elasticsearch cluster that indexes product names, descriptions, and "
        "SKUs. The Inventory Service tracks stock levels in Redis for low-latency "
        "reads, with periodic snapshots persisted to PostgreSQL for durability. "
        "The Order Service orchestrates the order lifecycle and publishes events to "
        "Apache Kafka topics (order.created, order.paid, order.shipped). The Payment "
        "Service consumes order.created events, processes payments via Stripe, and "
        "emits payment.completed events. The Notification Service listens to multiple "
        "Kafka topics and sends emails via SendGrid and push notifications via "
        "Firebase Cloud Messaging. The Shipping Service consumes order.paid events "
        "and integrates with carrier APIs (FedEx, UPS) for label generation and "
        "tracking number assignment. The Recommendation Service asynchronously "
        "computes collaborative filtering embeddings using user purchase history "
        "from a data warehouse (Snowflake) and serves recommendations via a gRPC "
        "endpoint consumed by the API Gateway. Inter-service communication follows "
        "the CQRS pattern: synchronous REST for commands and asynchronous Kafka "
        "events for queries. All services export Prometheus metrics scraped by a "
        "Prometheus server, with Grafana dashboards for visualization and Alertmanager "
        "for pager notifications. Distributed tracing is implemented with Jaeger, "
        "propagating trace context via OpenTelemetry SDKs in each service."
    ),
    "ground_truth": {
        "key_entities": [
            "Kong API Gateway",
            "Keycloak (OAuth2)",
            "User Service",
            "PostgreSQL",
            "Product Catalog Service",
            "MongoDB",
            "Elasticsearch",
            "Inventory Service",
            "Redis",
            "Order Service",
            "Apache Kafka",
            "Payment Service",
            "Stripe",
            "Notification Service",
            "SendGrid / FCM",
            "Shipping Service",
            "Carrier APIs (FedEx/UPS)",
            "Recommendation Service",
            "Snowflake",
            "Prometheus & Grafana",
            "Jaeger Tracing",
        ],
        "key_relationships": [
            ("Kong API Gateway", "User Service"),
            ("Kong API Gateway", "Product Catalog Service"),
            ("Kong API Gateway", "Inventory Service"),
            ("Kong API Gateway", "Order Service"),
            ("Kong API Gateway", "Recommendation Service"),
            ("Keycloak (OAuth2)", "Kong API Gateway"),
            ("User Service", "PostgreSQL"),
            ("Product Catalog Service", "MongoDB"),
            ("Product Catalog Service", "Elasticsearch"),
            ("Inventory Service", "Redis"),
            ("Inventory Service", "PostgreSQL"),
            ("Order Service", "Apache Kafka"),
            ("Apache Kafka", "Payment Service"),
            ("Payment Service", "Stripe"),
            ("Apache Kafka", "Notification Service"),
            ("Notification Service", "SendGrid / FCM"),
            ("Apache Kafka", "Shipping Service"),
            ("Shipping Service", "Carrier APIs (FedEx/UPS)"),
            ("Recommendation Service", "Snowflake"),
        ],
        "description": (
            "Hub-and-spoke architecture: Kong API Gateway at center routing to all "
            "services. Keycloak provides auth to Kong. Each service has its own "
            "data store (User -> PostgreSQL, Catalog -> MongoDB + Elasticsearch, "
            "Inventory -> Redis + PostgreSQL). Kafka as central event bus connecting "
            "Order, Payment, Notification, and Shipping services. Recommendation "
            "service reads from Snowflake. Observability layer (Prometheus, Grafana, "
            "Jaeger) spans all services."
        ),
    },
}


# ---------------------------------------------------------------------------
# Test Case 07 – Diffusion Model (model_architecture)
# ---------------------------------------------------------------------------

TC07 = {
    "id": "tc07",
    "title": "Denoising Diffusion Probabilistic Model (DDPM)",
    "figure_type": "model_architecture",
    "text": (
        "We implement a Denoising Diffusion Probabilistic Model for high-resolution "
        "image synthesis at 512x512 resolution. The model consists of two complementary "
        "processes: a forward diffusion process and a reverse denoising process. "
        "In the forward process, a clean image x_0 drawn from the data distribution "
        "is progressively corrupted by adding isotropic Gaussian noise over T=1000 "
        "timesteps according to a fixed variance schedule beta_t increasing linearly "
        "from beta_1=1e-4 to beta_T=0.02. At each timestep t, the latent variable "
        "x_t is sampled from q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, "
        "beta_t * I). The reverse process learns to denoise these corrupted latents "
        "using a U-Net architecture with self-attention layers at resolutions 16x16 "
        "and 8x8. The U-Net takes as input the noisy image x_t concatenated with a "
        "sinusoidal timestep embedding that encodes t into a 256-dimensional vector "
        "via sine and cosine transformations. The U-Net follows an encoder-decoder "
        "structure: the encoder applies four downsampling blocks, each containing two "
        "ResNet blocks with GroupNorm (32 groups) and SiLU activation, followed by a "
        "2x downsampling via strided convolution. The bottleneck contains two ResNet "
        "blocks interleaved with a self-attention module using 4 heads and a head "
        "dimension of 64. The decoder mirrors the encoder with four upsampling blocks "
        "that use nearest-neighbor upsampling followed by convolution, with skip "
        "connections from corresponding encoder levels. The final output is a 3-channel "
        "image predicting the noise epsilon added at timestep t. During inference "
        "(sampling), we start from pure Gaussian noise x_T ~ N(0, I) and iteratively "
        "apply the learned reverse transition p_theta(x_{t-1} | x_t) for t = T, ..., 1, "
        "optionally using DDIM sampling with 50 steps for faster generation. "
        "Classifier-free guidance with a guidance scale w=7.5 is employed to balance "
        "sample quality and diversity."
    ),
    "ground_truth": {
        "key_entities": [
            "Clean Image x_0",
            "Forward Diffusion Process",
            "Noisy Latent x_t",
            "Timestep Embedding (Sinusoidal)",
            "U-Net Encoder",
            "Downsampling Blocks",
            "ResNet Blocks + GroupNorm + SiLU",
            "Bottleneck (Self-Attention)",
            "U-Net Decoder",
            "Upsampling Blocks",
            "Skip Connections",
            "Predicted Noise epsilon",
            "Reverse Denoising Process",
            "Pure Gaussian Noise x_T",
            "Generated Image",
        ],
        "key_relationships": [
            ("Clean Image x_0", "Forward Diffusion Process"),
            ("Forward Diffusion Process", "Noisy Latent x_t"),
            ("Noisy Latent x_t", "U-Net Encoder"),
            ("Timestep Embedding (Sinusoidal)", "U-Net Encoder"),
            ("U-Net Encoder", "Downsampling Blocks"),
            ("Downsampling Blocks", "Bottleneck (Self-Attention)"),
            ("Bottleneck (Self-Attention)", "U-Net Decoder"),
            ("U-Net Decoder", "Upsampling Blocks"),
            ("Downsampling Blocks", "Skip Connections"),
            ("Skip Connections", "Upsampling Blocks"),
            ("U-Net Decoder", "Predicted Noise epsilon"),
            ("Pure Gaussian Noise x_T", "Reverse Denoising Process"),
            ("Reverse Denoising Process", "U-Net Encoder"),
            ("Predicted Noise epsilon", "Reverse Denoising Process"),
            ("Reverse Denoising Process", "Generated Image"),
        ],
        "description": (
            "Top section: Forward process (x_0 -> x_1 -> ... -> x_T) with Gaussian "
            "noise addition at each step. Bottom section: Reverse process using U-Net "
            "architecture. U-Net detail: encoder (downsampling blocks with ResNet + "
            "GroupNorm + SiLU) -> bottleneck (self-attention) -> decoder (upsampling "
            "blocks with skip connections). Timestep t injected via sinusoidal "
            "embedding. Output is predicted noise. At inference, x_T -> ... -> x_0 "
            "generated image."
        ),
    },
}


# ---------------------------------------------------------------------------
# Test Case 08 – Drug Discovery Pipeline (data_pipeline)
# ---------------------------------------------------------------------------

TC08 = {
    "id": "tc08",
    "title": "Computational Drug Discovery Pipeline",
    "figure_type": "data_pipeline",
    "text": (
        "Our integrated drug discovery pipeline accelerates hit-to-lead optimization "
        "through a combination of computational and experimental stages. The pipeline "
        "initiates with Target Identification, where disease-associated proteins are "
        "prioritized using GWAS summary statistics, gene-disease association data "
        "from Open Targets, and protein-protein interaction networks from STRING-db. "
        "A druggability assessment filters targets based on the presence of suitable "
        "binding pockets identified by Fpocket and the target's membership in the "
        "druggable genome (targets with known small-molecule modulators). The "
        "shortlisted targets proceed to Hit Identification via structure-based "
        "virtual screening: a library of 2.4 billion lead-like compounds from the "
        "Enamine REAL database is docked against the target's crystal structure or "
        "AlphaFold2-predicted model using AutoDock Vina on a GPU cluster, with the "
        "top 10,000 compounds by docking score retained. These hits are further "
        "filtered by ADMET property prediction using a graph neural network "
        "(ChemProp) trained on ChEMBL data, predicting logP, solubility, CYP450 "
        "inhibition, and hERG liability. Compounds passing all ADMET filters (about "
        "500) enter the Lead Optimization stage, where Free Energy Perturbation (FEP+) "
        "calculations are performed using Schrodinger's FEP+ on 32 A100 GPUs to "
        "predict relative binding free energies for proposed chemical modifications, "
        "guiding medicinal chemists in iterative compound refinement. The top 10 "
        "optimized leads are synthesized and tested in vitro: biochemical IC50 assays "
        "against the purified target protein, cellular target engagement assays using "
        "NanoBRET, and cytotoxicity screening in HepG2 cells. Compounds with IC50 < "
        "100 nM and selectivity > 100-fold over related targets advance to in vivo "
        "pharmacokinetic studies in Sprague-Dawley rats, measuring oral bioavailability "
        "(%F), clearance (CL), volume of distribution (Vd), and half-life (t1/2). "
        "Finally, a lead candidate meeting all developability criteria (IC50 < 10 nM, "
        "F > 30%, acceptable safety profile) is nominated for IND-enabling toxicology "
        "studies as the preclinical candidate. The entire pipeline is tracked in a "
        "custom LIMS with automated data QC at each stage."
    ),
    "ground_truth": {
        "key_entities": [
            "Target Identification (GWAS/Open Targets)",
            "Druggability Assessment (Fpocket)",
            "Hit Identification (Virtual Screening)",
            "Enamine REAL Compound Library",
            "Molecular Docking (AutoDock Vina)",
            "ADMET Prediction (ChemProp GNN)",
            "Lead Optimization (FEP+)",
            "In Vitro Testing (IC50/NanoBRET)",
            "Cytotoxicity Screening (HepG2)",
            "In Vivo PK Studies (Rat)",
            "Preclinical Candidate Nomination",
            "LIMS Tracking",
        ],
        "key_relationships": [
            ("Target Identification (GWAS/Open Targets)", "Druggability Assessment (Fpocket)"),
            ("Druggability Assessment (Fpocket)", "Hit Identification (Virtual Screening)"),
            ("Enamine REAL Compound Library", "Molecular Docking (AutoDock Vina)"),
            ("Hit Identification (Virtual Screening)", "Molecular Docking (AutoDock Vina)"),
            ("Molecular Docking (AutoDock Vina)", "ADMET Prediction (ChemProp GNN)"),
            ("ADMET Prediction (ChemProp GNN)", "Lead Optimization (FEP+)"),
            ("Lead Optimization (FEP+)", "In Vitro Testing (IC50/NanoBRET)"),
            ("In Vitro Testing (IC50/NanoBRET)", "Cytotoxicity Screening (HepG2)"),
            ("Cytotoxicity Screening (HepG2)", "In Vivo PK Studies (Rat)"),
            ("In Vivo PK Studies (Rat)", "Preclinical Candidate Nomination"),
        ],
        "description": (
            "Linear pipeline with decision gates: Target ID -> Druggability Assessment "
            "-> Hit ID (Virtual Screening + Docking of Enamine REAL library) -> ADMET "
            "Prediction (ChemProp filter) -> Lead Optimization (FEP+) -> In Vitro "
            "Testing (IC50, NanoBRET, HepG2 cytotoxicity) -> In Vivo PK (rat) -> "
            "Preclinical Candidate. LIMS spans all stages for tracking and QC."
        ),
    },
}


# ---------------------------------------------------------------------------
# Test Case 09 – Multi-Agent AI System (architecture_diagram)
# ---------------------------------------------------------------------------

TC09 = {
    "id": "tc09",
    "title": "Multi-Agent Collaborative AI System",
    "figure_type": "architecture_diagram",
    "text": (
        "We present a multi-agent framework for collaborative scientific research "
        "assistance. The system is composed of five specialized agents that communicate "
        "through a shared message bus implemented on top of Redis Pub/Sub with "
        "persistent message logging to PostgreSQL. The Orchestrator Agent receives "
        "the user's research question and decomposes it into sub-tasks using a "
        "fine-tuned T5-large model trained on 100,000 scientific question decomposition "
        "pairs. Sub-tasks are dispatched as structured messages containing a task_id, "
        "task_type, payload, and deadline. The Literature Review Agent, equipped with "
        "access to Semantic Scholar and PubMed APIs, retrieves relevant papers, "
        "extracts key findings using a SciBERT-based information extraction module, "
        "and publishes a structured literature summary. The Data Analysis Agent "
        "accepts datasets (CSV, HDF5, or FCS flow cytometry format), performs "
        "statistical tests (t-test, ANOVA, Mann-Whitney U), generates visualizations "
        "using matplotlib, and publishes analysis results along with figure captions. "
        "The Hypothesis Generation Agent consumes literature summaries and analysis "
        "results, then employs a knowledge graph constructed from the MeSH ontology "
        "and 15 million biomedical relations extracted from PubMed abstracts to "
        "propose novel hypotheses via link prediction, scoring each hypothesis with "
        "a confidence metric derived from graph embedding distances. The Critic Agent "
        "reviews all outputs from other agents, checking for internal consistency, "
        "statistical soundness, and adherence to scientific best practices using a "
        "checklist-based evaluation rubric, and publishes a critique with severity "
        "ratings (info, warning, error). The Orchestrator Agent collects all outputs, "
        "resolves conflicts using majority voting on contradictory claims, and "
        "synthesizes a final research report rendered in LaTeX. A shared Memory Bank "
        "(ChromaDB vector store) stores all intermediate representations as 768-dim "
        "embeddings from Sentence-BERT, enabling agents to recall previous results. "
        "The entire system exposes a WebSocket endpoint for real-time progress "
        "updates to a React-based dashboard."
    ),
    "ground_truth": {
        "key_entities": [
            "User (Research Question)",
            "Orchestrator Agent (T5-large)",
            "Redis Pub/Sub Message Bus",
            "PostgreSQL Message Log",
            "Literature Review Agent",
            "Semantic Scholar / PubMed APIs",
            "Data Analysis Agent",
            "Hypothesis Generation Agent",
            "MeSH Knowledge Graph",
            "Critic Agent",
            "Memory Bank (ChromaDB)",
            "React Dashboard (WebSocket)",
        ],
        "key_relationships": [
            ("User (Research Question)", "Orchestrator Agent (T5-large)"),
            ("Orchestrator Agent (T5-large)", "Redis Pub/Sub Message Bus"),
            ("Redis Pub/Sub Message Bus", "Literature Review Agent"),
            ("Redis Pub/Sub Message Bus", "Data Analysis Agent"),
            ("Redis Pub/Sub Message Bus", "Hypothesis Generation Agent"),
            ("Redis Pub/Sub Message Bus", "Critic Agent"),
            ("Literature Review Agent", "Semantic Scholar / PubMed APIs"),
            ("Literature Review Agent", "Redis Pub/Sub Message Bus"),
            ("Data Analysis Agent", "Redis Pub/Sub Message Bus"),
            ("Redis Pub/Sub Message Bus", "Hypothesis Generation Agent"),
            ("Hypothesis Generation Agent", "MeSH Knowledge Graph"),
            ("Critic Agent", "Redis Pub/Sub Message Bus"),
            ("Orchestrator Agent (T5-large)", "Memory Bank (ChromaDB)"),
            ("Redis Pub/Sub Message Bus", "PostgreSQL Message Log"),
            ("Orchestrator Agent (T5-large)", "React Dashboard (WebSocket)"),
        ],
        "description": (
            "Central message bus (Redis Pub/Sub) with five agents connected: "
            "Orchestrator (receives user query, dispatches tasks, synthesizes final "
            "report), Literature Review (searches Semantic Scholar/PubMed), Data "
            "Analysis (statistical tests + visualization), Hypothesis Generation "
            "(uses MeSH knowledge graph), Critic (reviews all outputs). Orchestrator "
            "also connects to ChromaDB memory bank and React dashboard. PostgreSQL "
            "logs all messages for persistence."
        ),
    },
}


# ---------------------------------------------------------------------------
# Test Case 10 – Climate Model Simulation Workflow (flowchart)
# ---------------------------------------------------------------------------

TC10 = {
    "id": "tc10",
    "title": "Earth System Model Climate Simulation Workflow",
    "figure_type": "flowchart",
    "text": (
        "We describe the workflow for running century-scale climate simulations "
        "using the Community Earth System Model version 2 (CESM2) at 1-degree "
        "atmospheric resolution with 72 vertical levels. The workflow begins with "
        "the Configuration Phase, where the user specifies the compset (component "
        "set) defining active model components: we use the B1850 compset for "
        "pre-industrial control simulations, which couples the Community Atmosphere "
        "Model (CAM6), Community Land Model (CLM5), Parallel Ocean Program (POP2), "
        "CICE sea ice model, and the MOSART river transport model, all connected "
        "through the CPL7 coupler. Boundary conditions including greenhouse gas "
        "concentrations (CO2=284.7 ppm, CH4=808.2 ppb for 1850), aerosol emissions "
        "from CMIP6 inventory, solar constant, orbital parameters, and land-use "
        "maps are assembled from the inputdata repository. The Pre-processing Phase "
        "interpolates these boundary conditions to the model grid and generates "
        "initial condition files by spinning up each component for 500 years "
        "independently before coupling. The Simulation Phase launches the coupled "
        "model on 2048 CPU cores of a Cray XC50 supercomputer, running with a "
        "timestep of 30 minutes for the atmosphere and 1 hour for the ocean. The "
        "model outputs history files every month containing 3D fields (temperature, "
        "salinity, wind velocity) and 2D surface fluxes. A runtime monitoring daemon "
        "tracks the global mean top-of-atmosphere energy balance; if the imbalance "
        "exceeds 1 W/m^2, the workflow triggers a branch restart from the last "
        "checkpoint with adjusted cloud tuning parameters. After the simulation "
        "completes (typically 200 simulated years), the Post-processing Phase uses "
        "the Climate Data Operators (CDO) and NCAR Command Language (NCL) to regrid "
        "output from the native spectral element grid to a regular 1x1 degree "
        "latitude-longitude grid, compute climatological monthly means over the "
        "final 50 years, and derive standard diagnostics: global mean surface "
        "temperature anomaly, Atlantic Meridional Overturning Circulation (AMOC) "
        "index, Niño 3.4 index, and top-of-atmosphere radiative fluxes. Diagnostics "
        "are validated against ERA5 reanalysis and HadCRUT5 observations, with "
        "Taylor diagrams generated for spatial pattern comparison. The validated "
        "outputs are published to the Earth System Grid Federation (ESGF) data node "
        "in CMOR-compliant NetCDF4 format with CF-1.7 metadata conventions."
    ),
    "ground_truth": {
        "key_entities": [
            "Configuration Phase (Compset B1850)",
            "CAM6 Atmosphere",
            "CLM5 Land",
            "POP2 Ocean",
            "CICE Sea Ice",
            "MOSART River Transport",
            "CPL7 Coupler",
            "Pre-processing (Spin-up)",
            "Simulation Phase (2048 cores)",
            "Runtime Monitoring Daemon",
            "Post-processing (CDO/NCL)",
            "Climatological Diagnostics",
            "Validation (ERA5/HadCRUT5)",
            "ESGF Publication",
        ],
        "key_relationships": [
            ("Configuration Phase (Compset B1850)", "Pre-processing (Spin-up)"),
            ("Pre-processing (Spin-up)", "CAM6 Atmosphere"),
            ("Pre-processing (Spin-up)", "CLM5 Land"),
            ("Pre-processing (Spin-up)", "POP2 Ocean"),
            ("Pre-processing (Spin-up)", "CICE Sea Ice"),
            ("Pre-processing (Spin-up)", "MOSART River Transport"),
            ("CAM6 Atmosphere", "CPL7 Coupler"),
            ("CLM5 Land", "CPL7 Coupler"),
            ("POP2 Ocean", "CPL7 Coupler"),
            ("CICE Sea Ice", "CPL7 Coupler"),
            ("MOSART River Transport", "CPL7 Coupler"),
            ("CPL7 Coupler", "Simulation Phase (2048 cores)"),
            ("Simulation Phase (2048 cores)", "Runtime Monitoring Daemon"),
            ("Runtime Monitoring Daemon", "Simulation Phase (2048 cores)"),
            ("Simulation Phase (2048 cores)", "Post-processing (CDO/NCL)"),
            ("Post-processing (CDO/NCL)", "Climatological Diagnostics"),
            ("Climatological Diagnostics", "Validation (ERA5/HadCRUT5)"),
            ("Validation (ERA5/HadCRUT5)", "ESGF Publication"),
        ],
        "description": (
            "Top-down workflow: Configuration (B1850 compset with CAM6/CLM5/POP2/CICE/"
            "MOSART) -> Pre-processing (spin-up) -> Simulation (2048 cores, CPL7 "
            "coupler connecting all components, with monitoring daemon feedback loop) "
            "-> Post-processing (CDO/NCL regridding and diagnostics) -> Validation "
            "(ERA5/HadCRUT5, Taylor diagrams) -> ESGF Publication. The model "
            "components feed into the CPL7 coupler which drives the simulation phase."
        ),
    },
}


# ---------------------------------------------------------------------------
# Aggregate exports
# ---------------------------------------------------------------------------

TEST_CASES: list[dict] = [
    TC01,
    TC02,
    TC03,
    TC04,
    TC05,
    TC06,
    TC07,
    TC08,
    TC09,
    TC10,
]

TEST_CASES_BY_TYPE: dict[str, list[str]] = {
    "model_architecture": ["tc01", "tc07"],
    "architecture_diagram": ["tc02", "tc06", "tc09"],
    "flowchart": ["tc03", "tc05", "tc10"],
    "data_pipeline": ["tc04", "tc08"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_test_cases() -> list[dict]:
    """Return all test cases (deep-copied to prevent mutation)."""
    return copy.deepcopy(TEST_CASES)


def get_test_case(ids: Optional[list[str]] = None) -> list[dict]:
    """Return specific test cases by id, or all if ids is None.

    Args:
        ids: List of test case IDs (e.g. ["tc01", "tc03"]). If None, returns all.

    Returns:
        List of test case dicts matching the requested IDs.

    Raises:
        ValueError: If any requested ID does not exist.
    """
    if ids is None:
        return copy.deepcopy(TEST_CASES)

    id_set = {tc["id"] for tc in TEST_CASES}
    requested_set = set(ids)
    missing = requested_set - id_set
    if missing:
        raise ValueError(
            f"Unknown test case IDs: {sorted(missing)}. "
            f"Available: {sorted(id_set)}"
        )

    return [copy.deepcopy(tc) for tc in TEST_CASES if tc["id"] in requested_set]
