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
#   bash run.sh <scale> <stage> [dense|slashmla|slash_hca] [额外的 train.sh 参数...]
#     scale : 0p6B | 1p7B | 4BA600M | 10B
#     stage : 8k | 32k | 128k
#   例:
#     bash run.sh 10B 8k
#     bash run.sh 0p6B 8k slashmla
#     bash run.sh 0p6B 8k slash_hca
#     bash run.sh 1p7B 8k slash_hca
#     bash run.sh 4BA600M 8k
#     bash run.sh 10B 32k --kwargs resume_from_checkpoint=<8K ckpt>
#     bash run.sh 10B 128k --kwargs resume_from_checkpoint=<32K ckpt>   # 续 32K, cosine 连续
#
# 注意:
#   - 32k/128k 是 override, 叠在对应 8k base 之上 (--configs 8k stage)。
#   - 128k 必须 resume 32K ckpt 以恢复 LR_Scheduler, 使 cosine 连续 (见 SCALING.md §5)。
#   - 机器数不在本脚本中硬编码，由 script/train.sh/PDC 启动环境决定。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SLASH -> Research -> PaddleFleet -> third_party -> $workspace (闭源根)
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONF_DIR="third_party/PaddleFleet/Research/SLASH/conf"   # 相对 ROOT

scale="${1:-}"
stage="${2:-}"
mode="${3:-dense}"

# Keep the historical two-argument form working. When a third argument starts
# with "-", it is a train.sh option rather than an attention mode.
if [[ $# -ge 3 && "$mode" == -* ]]; then
  mode="dense"
  shift 2
elif [[ $# -ge 3 ]]; then
  shift 3
else
  shift 2
fi

case "$scale" in
  0p6B|0.6B) name="slash_mla_0p6B_dense" ;;
  1p7B|1.7B) name="slash_mla_1p7B_dense" ;;
  4BA600M|4B_A600M|4ba600m) name="slash_mla_4ba600m_dense" ;;
  10B|10B_A1B|10BA1B) name="slash_mla_10B_A1B" ;;
  *) echo "错误: scale 必须是 0p6B|1p7B|4BA600M|10B (给的是 '$scale')"; exit 1 ;;
esac

case "$mode" in
  dense)
    base_yaml="$CONF_DIR/${name}_8k.yaml"
    case "$stage" in
      8k)   configs=("$base_yaml") ;;
      32k|128k)
        if [[ "$scale" == "4BA600M" || "$scale" == "4B_A600M" || "$scale" == "4ba600m" ]]; then
          echo "错误: 当前 4BA600M 仅提供 8k 配置 (给的是 '$stage')"
          exit 1
        fi
        if [[ "$stage" == "32k" ]]; then
          configs=("$base_yaml" "$CONF_DIR/${name}_32k.yaml")
        else
          configs=("$base_yaml" "$CONF_DIR/${name}_128k.yaml")
        fi
        ;;
      *) echo "错误: stage 必须是 8k|32k|128k (给的是 '$stage')"; exit 1 ;;
    esac
    ;;
  slashmla)
    case "$scale" in
      0p6B|0.6B) slashmla_name="slash_mla_0p6B_slashmla" ;;
      1p7B|1.7B) slashmla_name="slash_mla_1p7B_slashmla" ;;
      10B|10B_A1B|10BA1B) slashmla_name="slash_mla_10B_A1B_slashmla" ;;
    esac

    slashmla_base_yaml="$CONF_DIR/${slashmla_name}_8k.yaml"
    # The current checked-in 0p6B config predates the generic naming scheme.
    # Keep it as a compatibility fallback; this suffix does not constrain the
    # actual number of machines used by script/train.sh.
    if [[ ! -f "$slashmla_base_yaml" ]]; then
      if [[ "$scale" == "0p6B" || "$scale" == "0.6B" ]]; then
        slashmla_base_yaml="$CONF_DIR/${slashmla_name}_8k_1node.yaml"
      fi
    fi

    case "$stage" in
      8k)   configs=("$slashmla_base_yaml") ;;
      32k)  configs=("$slashmla_base_yaml" "$CONF_DIR/${slashmla_name}_32k.yaml") ;;
      128k) configs=("$slashmla_base_yaml" "$CONF_DIR/${slashmla_name}_128k.yaml") ;;
      *) echo "错误: stage 必须是 8k|32k|128k (给的是 '$stage')"; exit 1 ;;
    esac
    ;;
  slash_hca|slashhca|hca)
    slash_hca_extra_configs=()
    case "$scale" in
      0p6B|0.6B) slash_hca_name="slash_mla_0p6B_hca_slash" ;;
      1p7B|1.7B)
        # 1.7B HCA reuses the dense 8K base and overlays the
        # single-node batch/recompute/model-path settings.
        slash_hca_name="slash_mla_1p7B_hca_slash"
        slash_hca_base_yaml="$CONF_DIR/slash_mla_1p7B_dense_8k.yaml"
        slash_hca_extra_configs=(
          "$CONF_DIR/${slash_hca_name}_8k_1node.yaml"
        )
        ;;
      *)
        echo "错误: 当前 slash_hca 仅支持 0p6B|1p7B (给的是 '$scale')"
        exit 1
        ;;
    esac

    if [[ -z "${slash_hca_base_yaml:-}" ]]; then
      slash_hca_base_yaml="$CONF_DIR/${slash_hca_name}_8k.yaml"
      # The one-node HCA config is the checked-in 8K baseline for 0p6B.
      if [[ ! -f "$slash_hca_base_yaml" ]]; then
        slash_hca_base_yaml="$CONF_DIR/${slash_hca_name}_8k_1node.yaml"
      fi
    fi

    if [[ "$scale" == "1p7B" || "$scale" == "1.7B" ]]; then
      case "$stage" in
        8k) ;;
        *)
          echo "错误: 当前 1p7B slash_hca 仅支持 8k (给的是 '$stage')"
          exit 1
          ;;
      esac
    fi

    case "$stage" in
      8k)   configs=("$slash_hca_base_yaml" "${slash_hca_extra_configs[@]}") ;;
      32k)  configs=("$slash_hca_base_yaml" "$CONF_DIR/${slash_hca_name}_32k.yaml") ;;
      128k) configs=("$slash_hca_base_yaml" "$CONF_DIR/${slash_hca_name}_128k.yaml") ;;
      *) echo "错误: stage 必须是 8k|32k|128k (给的是 '$stage')"; exit 1 ;;
    esac
    ;;
  *)
    echo "错误: mode 必须是 dense|slashmla|slash_hca (给的是 '$mode')"
    exit 1
    ;;
esac

cd "$ROOT"
for c in "${configs[@]}"; do
  [ -f "$c" ] || { echo "错误: 配置不存在 $ROOT/$c"; exit 1; }
done

echo "[SLASH run] scale=$scale stage=$stage mode=$mode"
echo "[SLASH run] cwd=$ROOT"
echo "[SLASH run] --configs ${configs[*]} $*"
exec bash script/train.sh --configs "${configs[@]}" "$@"
