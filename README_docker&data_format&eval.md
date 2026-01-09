# Laparo-VLLM Docker Deployment and Inference Guide

## 1. Docker Setup

### 1.1 Download Docker Image

Download the pre-built Docker image from Google Drive:

**File:** `laparo-vllm_checkpoint-10609-cuda12.2_20251113_020023.tar.gz` (22GB)  
**Link:** https://drive.google.com/file/d/1zXGTIRIJjsNsi41QXXEBgobPz4ZOqxUK/view

```bash
# Install gdown (if not already installed)
pip install gdown

# Download the Docker image
gdown --fuzzy "https://drive.google.com/file/d/1zXGTIRIJjsNsi41QXXEBgobPz4ZOqxUK/view"

# If download is interrupted, use -c for resume
gdown --fuzzy -c "https://drive.google.com/file/d/1zXGTIRIJjsNsi41QXXEBgobPz4ZOqxUK/view"
```

### 1.2 Load Docker Image

Load the pre-built Docker image from the archive file:

```bash
docker load -i laparo-vllm.tar.gz
```

### 1.3 Run Docker Container

Start the Docker container with GPU support and volume mounting:

```bash
docker run -it \
    --gpus all \
    --name vllm \
    -v /path/to/your/data:/data \
    -p 8000:8000 \
    --ipc=host \
    --shm-size=32g \
    laparo-vllm:checkpoint-10609-cuda12.2 \
    /bin/bash
```

**Parameters Explanation:**
- `--gpus all`: Enable all available GPUs
- `--name vllm`: Container name
- `-v /path/to/your/data:/data`: Mount your local data directory to `/data` in the container (replace `/path/to/your/data` with your actual path)
- `-p 8000:8000`: Map port 8000 for API access
- `--ipc=host`: Use host IPC namespace for shared memory
- `--shm-size=32g`: Allocate 32GB shared memory

### 1.4 Docker FAQ

**Q: Container name conflict error**

```
docker: Error response from daemon: Conflict. The container name "/vllm" is already in use by container "xxx". 
You have to remove (or rename) that container to be able to reuse that name.
```

**A:** This error occurs when a container with the same name already exists. Stop and remove the existing container:

```bash
docker stop vllm
docker rm vllm
```

Then re-run the `docker run` command.

### 1.5 Quick Start with Simple Test

We provide a `simple_test` dataset for quick validation of your setup. This contains 30 sample images from Cholec80 with phase recognition MCQ questions.

#### Step 1: Update Image Paths in JSONL

The JSONL file contains image paths that need to be updated to match your mount location. Use `sed` to replace the root path:

```bash
# Replace the original root path with your target path
# Example: Change /data/tos_copy to /data (if mounting simple_test to /data)
sed -i 's|/data/tos_copy/|/data/|g' simple_test/cholec80_phase_mcq_simple_test.jsonl
```

The original paths look like:
```
/data/tos_copy/cholec80/frames/42/63051.jpg
```

After replacement (mounting `simple_test` to `/data`):
```
/data/cholec80/frames/42/63051.jpg
```

> **Tip:** Adjust the replacement path based on your Docker mount configuration. The key is to ensure the path in JSONL matches the actual image location inside the container.

#### Step 2: Run Docker Container

Mount the `simple_test` folder to `/data` in the container:

```bash
docker run -it \
    --gpus all \
    --name vllm \
    -v /path/to/simple_test:/data \
    -p 8000:8000 \
    --ipc=host \
    --shm-size=32g \
    laparo-vllm:checkpoint-10609-cuda12.2 \
    /bin/bash
```

#### Step 2: Run Inference

Inside the container, run:

```bash
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /app/model \
    --system /app/prompts/think_prompt.txt \
    --attn_impl flash_attn \
    --infer_backend vllm \
    --max_batch_size 20 \
    --val_dataset /data/cholec80_phase_mcq_simple_test.jsonl \
    --result_path /data/res.jsonl \
    --vllm_max_model_len 4096 \
    --max_new_tokens 2048 \
    --write_batch_size 1000
```

#### Step 3: Evaluate Results

```bash
python /app/eval/eval_mcq_acc.py /data/res.jsonl -v
```

> **Note:** Make sure the image paths in the JSONL file match the actual image locations inside the container. If you mount `simple_test` to `/data`, the images will be at `/data/cholec80/frames/...`, so the JSONL paths should be updated accordingly (see Step 1).

---

## 2. Data Format Specification

The model accepts data in **JSON Lines (`.jsonl`)** format, where each line represents a single data sample as a JSON object.

### 2.1 General Question-Answering Tasks

Used for multiple-choice questions about surgery phases, instruments, actions, etc.

#### Data Structure

```jsonl
{"messages": [{"role": "user", "content": "Given the [surgery type] image <image>, [question]?\nA. [option A]\nB. [option B]\n..."}, {"role": "assistant", "content": "[Correct Option]"}], "images": ["/path/to/image.jpg"]}
```

#### Field Descriptions

- **`messages`**: Conversation array containing user query and assistant response
  - `role`: Either "user" or "assistant"
  - `content`: Question text with `<image>` placeholder for the user; answer for the assistant
- **`images`**: Array of absolute image paths (support single or multiple images)

#### Examples

**Surgical Phase Recognition:**

```jsonl
{"messages": [{"role": "user", "content": "Given the laparoscopic cholecystectomy image <image>, which surgical phase is being performed?\nA. GallbladderPackaging\nB. GallbladderRetraction\nC. ClippingCutting\nD. Preparation\nE. GallbladderDissection\nF. CleaningCoagulation\nG. CalotTriangleDissection"}, {"role": "assistant", "content": "D. Preparation"}], "images": ["/tos/cholec80/frames_1fps/01/00000.jpg"]}
```

**Instrument Recognition:**

```jsonl
{"messages": [{"role": "user", "content": "Given the laparoscopic cholecystectomy image <image>, which instrument is visible?\nA. Grasper\nB. Hook\nC. Scissors\nD. Clipper\nE. Irrigator"}, {"role": "assistant", "content": "A. Grasper"}], "images": ["/tos/cholec80/frames_1fps/01/00100.jpg"]}
```

**Action Recognition:**

```jsonl
{"messages": [{"role": "user", "content": "Given the radical prostatectomy image <image>, what action related to the needle and suture is the surgeon focusing on right now?\nA. other\nB. picking-up the needle\nC. positioning the needle tip\nD. pushing the needle through the tissue\nE. pulling the needle out of the tissue\nF. tying a knot\nG. cutting the suture\nH. returning or dropping the needle"}, {"role": "assistant", "content": "D. pushing the needle through the tissue"}], "images": ["/data/tos_copy/SAR_RARP/test_set/video_49/rgb/000018066.png"]}
```

### 2.2 Object Localization Tasks

Used for detecting and localizing surgical instruments or anatomical structures.

#### Data Structure

```jsonl
{"messages": [{"role": "user", "content": "Given the laparoscopic surgical image <image>, find <ref-object> in the format of bbox (x1,y1), (x2,y2)."}, {"role": "assistant", "content": "<bbox>"}], "images": ["/path/to/image.png"], "objects": {"ref": ["<ref-object>"], "bbox": [[x1, y1, x2, y2]]}}
```

#### Field Descriptions

- **`messages`**: Same as general QA format, with `<ref-object>` placeholder
- **`images`**: Array of absolute image paths
- **`objects`**: Object annotation information
  - `ref`: Array of object names (must match `<ref-object>` in content)
  - `bbox`: Array of bounding boxes in format `[x1, y1, x2, y2]` (top-left and bottom-right coordinates)

#### Examples

```jsonl
{"messages": [{"role": "user", "content": "Given the laparoscopic surgical image <image>, find <ref-object> in the format of bbox (x1,y1), (x2,y2)."}, {"role": "assistant", "content": "<bbox>"}], "images": ["/data/tos_copy/Endovision17/Endovision17/Test_Dataset/instrument_dataset_3/left_frames/frame271.png"], "objects": {"ref": ["Large Needle Driver"], "bbox": [[1468, 439, 1593, 511]]}}
```

```jsonl
{"messages": [{"role": "user", "content": "Given the laparoscopic surgical image <image>, find <ref-object> in the format of bbox (x1,y1), (x2,y2)."}, {"role": "assistant", "content": "<bbox>"}], "images": ["/data/tos_copy/Endovision17/Endovision17/Test_Dataset/instrument_dataset_3/left_frames/frame272.png"], "objects": {"ref": ["Bipolar Forceps"], "bbox": [[100, 200, 300, 400]]}}
```

---

## 3. Model Inference

### 3.1 Single-GPU Inference

Run inference on a single GPU with the following command:

```bash
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /app/model \
    --system /app/prompts/think_prompt.txt \
    --attn_impl flash_attn \
    --infer_backend vllm \
    --max_batch_size 20 \
    --val_dataset /path/to/your/test_data.jsonl \
    --result_path /app/res.jsonl \
    --vllm_max_model_len 4096 \
    --max_new_tokens 2048 \
    --write_batch_size 1000
```

#### Key Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `MAX_PIXELS` | Maximum image resolution (pixels) | `1003520` |
| `CUDA_VISIBLE_DEVICES` | GPU device ID(s) | `0` |
| `--model` | Path to model directory | `/app/model` |
| `--system` | System prompt file | `/app/prompts/think_prompt.txt` |
| `--val_dataset` | Input test dataset path (`.jsonl`) | Your test file path |
| `--result_path` | Output results path (`.jsonl`) | Your output file path |
| `--max_batch_size` | Maximum batch size for inference | `300` |
| `--vllm_max_model_len` | Maximum context length | `4096` |
| `--max_new_tokens` | Maximum tokens to generate | `2048` |

### 3.2 Multi-GPU Inference

For faster inference using multiple GPUs, add the `--vllm_tensor_parallel_size` parameter:

```bash
MAX_PIXELS=1003520 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift infer \
    --model /app/model \
    --system /app/prompts/think_prompt.txt \
    --attn_impl flash_attn \
    --infer_backend vllm \
    --max_batch_size 300 \
    --val_dataset /path/to/your/test_data.jsonl \
    --result_path /app/res.jsonl \
    --vllm_max_model_len 4096 \
    --vllm_tensor_parallel_size 2 \
    --max_new_tokens 2048 \
    --write_batch_size 1000
```

#### Additional Parameters for Multi-GPU

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `NPROC_PER_NODE` | Number of processes per node | `8` |
| `CUDA_VISIBLE_DEVICES` | Comma-separated GPU IDs | `0,1,2,3,4,5,6,7` |
| `--vllm_tensor_parallel_size` | Number of GPUs for tensor parallelism | `2` |

> **Note:** The `--vllm_tensor_parallel_size` should be ≤ the number of visible GPUs. For example, if using 8 GPUs, you can set this to 2, 4, or 8 depending on your model size and memory requirements.

---

## 4. Output Format

The inference results will be saved to the specified `--result_path` as a `.jsonl` file, with each line containing:
- Original input data
- Model prediction

---

## 5. Evaluation Script

Use `eval_mcq_acc.py` to evaluate MCQ (multiple-choice question) inference results.

### 5.1 Basic Usage

**Inside Docker Container:**

```bash
python /app/eval/eval_mcq_acc.py /path/to/inference_result.jsonl
```

**Outside Docker (Local):**

```bash
python /path/to/eval_mcq_acc.py /path/to/inference_result.jsonl
```

### 5.2 With Options

```bash
# Specify output directory
python eval_mcq_acc.py result.jsonl -o /path/to/output

# Show per-class metrics
python eval_mcq_acc.py result.jsonl -v
```

### 5.3 Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_file` | Path to inference result JSONL file | Required |
| `-o, --output-dir` | Output directory for results | Same as input file |
| `-v, --verbose` | Show per-class metrics | False |

### 5.4 Input Format

The script expects JSONL files with the following fields (supports multiple naming conventions):

| Field | Alternative Names | Description |
|-------|-------------------|-------------|
| `response` | `prediction`, `predicted_phase`, `output` | Model's prediction |
| `labels` | `label`, `ground_truth`, `true_phase`, `target` | Ground truth answer |

### 5.5 Example Output

**Standard Output:**

```
============================================================
MCQ Evaluation Results: cholec80_phase_result.jsonl
============================================================
Accuracy:  0.8542 (1234/1445)
mAP:       0.8123

Macro Average:
  Precision: 0.8234  Recall: 0.8156
  F1-Score:  0.8195  Jaccard: 0.6934

Weighted Average:
  Precision: 0.8567  Recall: 0.8542
  F1-Score:  0.8554  Jaccard: 0.7478

============================================================
Results saved to: /path/to/cholec80_phase_result_acc.json
```

**Verbose Output (with `-v`):**

```
============================================================
MCQ Evaluation Results: cholec80_phase_result.jsonl
============================================================
Accuracy:  0.8542 (1234/1445)
mAP:       0.8123

Macro Average:
  Precision: 0.8234  Recall: 0.8156
  F1-Score:  0.8195  Jaccard: 0.6934

Weighted Average:
  Precision: 0.8567  Recall: 0.8542
  F1-Score:  0.8554  Jaccard: 0.7478

Class                                   Prec   Recall       F1     Jacc   Supp
-------------------------------------------------------------------------------
CalotTriangleDissection               0.8912   0.8756   0.8833   0.7908    356
CleaningCoagulation                   0.7845   0.8012   0.7928   0.6573    245
ClippingCutting                       0.9234   0.9123   0.9178   0.8479    189
GallbladderDissection                 0.8567   0.8234   0.8397   0.7237    312
GallbladderPackaging                  0.8123   0.7956   0.8039   0.6721    143
GallbladderRetraction                 0.7689   0.7823   0.7755   0.6341    112
Preparation                           0.8234   0.8456   0.8344   0.7158     88

============================================================
Results saved to: /path/to/cholec80_phase_result_acc.json
```

### 5.6 Output File Format

The script generates a JSON file (`<input_stem>_acc.json`) with the following structure:

```json
{
  "input_file": "/path/to/inference_result.jsonl",
  "accuracy": 0.8542,
  "correct": 1234,
  "total": 1445,
  "map": 0.8123,
  "macro_average": {
    "precision": 0.8234,
    "recall": 0.8156,
    "f1": 0.8195,
    "jaccard": 0.6934
  },
  "weighted_average": {
    "precision": 0.8567,
    "recall": 0.8542,
    "f1": 0.8554,
    "jaccard": 0.7478
  },
  "num_classes": 7
}
```

### 5.7 Metrics Description

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions / total predictions |
| **mAP** | Mean Average Precision across all classes |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1-Score** | Harmonic mean of precision and recall |
| **Jaccard** | IoU = True positives / (TP + FP + FN) |
| **Macro Average** | Unweighted mean across all classes |
| **Weighted Average** | Mean weighted by class support (sample count) |
