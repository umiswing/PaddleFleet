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

# 单机8卡验证启动器 (mpirun 分发到各 pod; 仅 rank0=10.79.184.226 实际参与训练)
# 安全: 不调用 kill_process; 非参与 pod 在 train_gpu.sh 的 selective_launch 后 exit 0,
#       不触碰其它节点上别人的进程 (如 242 上 lizhenxing 的任务)。
export model_type="eb5"
cd /root/paddlejob/share-storage/gpfs/system-public/wangguoxia/ERNIE/exp2p20_pull
bash script/train.sh --configs third_party/PaddleFleet/Research/SLASH/conf/slash_mla_0p6B_dense_8k.yaml third_party/PaddleFleet/Research/SLASH/conf/slash_mla_0p6B_dense_8k_1node.yaml
