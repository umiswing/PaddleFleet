#!/usr/bin/env bash

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================================
# SLASH MLA 实验启动器 (开源侧; 只读调用闭源引擎, 不改闭源外层)
#
# 解耦约定: 本脚本 cd 到 $workspace 根(闭源引擎的运行 cwd), 只读地调用
#   闭源的 script/train.sh --configs <SLASH 开源 yaml>。实验的全部可变部分
#   (模型结构/yaml/数据/启动/新 attention 代码) 都在开源侧, 闭源外层一行不改。
#
# 用法:
#   bash run.sh <scale> <stage> [额外的 train.sh 参数...]
#     scale : 0p6B | 1p7B | 10B
#     stage : 8k | 32k | 128k
#   例:
#     bash run.sh 10B 8k
#     bash run.sh 10B 32k  --kwargs resume_from_checkpoint=<8K ckpt>
#     bash run.sh 10B 128k --kwargs resume_from_checkpoint=<32K ckpt>   # 续 32K, cosine 连续
#
# 注意:
#   - 32k/128k 是 override, 叠在对应 8k base 之上 (--configs 8k stage)。
#   - 128k 必须 resume 32K ckpt 以恢复 LR_Scheduler, 使 cosine 连续 (见 SCALING.md §5)。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SLASH -> Research -> PaddleFleet -> third_party -> $workspace (闭源根)
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONF_DIR="third_party/PaddleFleet/Research/SLASH/conf"   # 相对 ROOT

scale="${1:-}"; stage="${2:-}"; shift 2 || true

case "$scale" in
  0p6B|0.6B) name="slash_mla_0p6B_dense" ;;
  1p7B|1.7B) name="slash_mla_1p7B_dense" ;;
  10B|10B_A1B|10BA1B) name="slash_mla_10B_A1B" ;;
  *) echo "错误: scale 必须是 0p6B|1p7B|10B (给的是 '$scale')"; exit 1 ;;
esac

base_yaml="$CONF_DIR/${name}_8k.yaml"
case "$stage" in
  8k)   configs="$base_yaml" ;;
  32k)  configs="$base_yaml $CONF_DIR/${name}_32k.yaml" ;;
  128k) configs="$base_yaml $CONF_DIR/${name}_128k.yaml" ;;
  *) echo "错误: stage 必须是 8k|32k|128k (给的是 '$stage')"; exit 1 ;;
esac

cd "$ROOT"
for c in $configs; do
  [ -f "$c" ] || { echo "错误: 配置不存在 $ROOT/$c"; exit 1; }
done

echo "[SLASH run] scale=$scale stage=$stage"
echo "[SLASH run] cwd=$ROOT"
echo "[SLASH run] --configs $configs $*"
exec bash script/train.sh --configs $configs "$@"
