#!/usr/bin/env bash
#
# install_adair_env.sh
#
# AdaIR 环境分步安装脚本。
# 解决 "conda env create" 一次性求解 / 安装时内存爆掉被 Killed 的问题：
#   - 先只建一个干净的 env
#   - conda 依赖逐个安装（每次只 solve 一个包，内存占用极低）
#   - pip 依赖逐个安装（--no-cache-dir，避免一次性展开大包撑爆内存/磁盘）
#   - 全程记录"成功"和"失败"清单，结束后打印汇总
#
# 用法:
#   bash install_adair_env.sh [env.yaml的路径]
#   # 不传参数时默认读取当前目录下的 env.yaml
#
# 失败的包会被记录到 logs/failed.log，重跑脚本时已成功的会自动跳过。
#
set -uo pipefail

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
ENV_FILE="${1:-env.yaml}"
ENV_NAME="AdaIR"
PY_VERSION="3.8.11"          # 与 env.yaml 中 python=3.8.11 对应

LOG_DIR="./adair_install_logs"
SUCCESS_LOG="$LOG_DIR/success.log"
FAIL_LOG="$LOG_DIR/failed.log"
DETAIL_LOG="$LOG_DIR/detail.log"

# 找一个可用的求解器：优先 mamba（更快更省内存），否则用 conda
if command -v mamba >/dev/null 2>&1; then
    CONDA_BIN="mamba"
else
    CONDA_BIN="conda"
fi

# ----------------------------------------------------------------------------
# 关键：降低 conda 求解阶段的内存峰值，避免容器里 "Solving environment" 被 Killed
#   - 优先启用 libmamba 求解器（C++实现，内存占用比 classic 求解器低一个数量级）
#   - 只用 main 单一 channel + --override-channels，减少 repodata 体积
# ----------------------------------------------------------------------------
SOLVER_ARGS=()
setup_solver() {
    if [[ "$CONDA_BIN" == "mamba" ]]; then
        log_info "使用 mamba 求解（已内置低内存求解器）"
        return
    fi
    # 检测 conda 是否支持 libmamba 求解器
    if conda config --show solver >/dev/null 2>&1 || \
       python -c "import conda_libmamba_solver" >/dev/null 2>&1; then
        SOLVER_ARGS=(--solver=libmamba)
        log_info "已启用 libmamba 求解器（大幅降低求解内存）"
    else
        log_info "未检测到 libmamba，尝试安装 conda-libmamba-solver ..."
        if conda install -y -n base conda-libmamba-solver >> "${DETAIL_LOG:-/dev/null}" 2>&1; then
            SOLVER_ARGS=(--solver=libmamba)
            log_ok "libmamba 安装成功，已启用"
        else
            log_err "libmamba 安装失败，将回退到 classic 求解器（求解阶段更易被 Killed）"
            log_err "强烈建议先加 swap，见脚本顶部说明。"
        fi
    fi
}

# ----------------------------------------------------------------------------
# 前置检查
# ----------------------------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
    echo "[错误] 找不到环境文件: $ENV_FILE"
    echo "用法: bash $0 [env.yaml的路径]"
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[错误] 当前 shell 找不到 conda，请先 source 你的 conda 初始化脚本。"
    exit 1
fi

mkdir -p "$LOG_DIR"
: > "$DETAIL_LOG"   # detail 每次重置；success/fail 保留以便断点续装
touch "$SUCCESS_LOG" "$FAIL_LOG"

# 颜色（终端支持时）
if [[ -t 1 ]]; then
    C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_INFO=$'\033[36m'; C_RST=$'\033[0m'
else
    C_OK=""; C_ERR=""; C_INFO=""; C_RST=""
fi

log_info() { echo "${C_INFO}[INFO]${C_RST} $*"; }
log_ok()   { echo "${C_OK}[ OK ]${C_RST} $*"; }
log_err()  { echo "${C_ERR}[FAIL]${C_RST} $*"; }

# 判断某个包是否已经记录为成功（用于断点续装）
already_done() {
    grep -qxF "$1" "$SUCCESS_LOG" 2>/dev/null
}

mark_success() {
    echo "$1" >> "$SUCCESS_LOG"
    # 从失败清单里移除（如果之前失败过）
    if [[ -s "$FAIL_LOG" ]]; then
        grep -vxF "$1" "$FAIL_LOG" > "$FAIL_LOG.tmp" 2>/dev/null && mv "$FAIL_LOG.tmp" "$FAIL_LOG"
    fi
}

mark_fail() {
    # 去重写入失败清单
    grep -qxF "$1" "$FAIL_LOG" 2>/dev/null || echo "$1" >> "$FAIL_LOG"
}

# ----------------------------------------------------------------------------
# 从 env.yaml 解析依赖列表
#   - conda 依赖: dependencies: 下缩进 2 空格的 "- xxx"（不含 "- pip:"）
#   - pip   依赖: pip: 下缩进更多的 "- xxx"
# ----------------------------------------------------------------------------
parse_conda_deps() {
    awk '
        /^dependencies:/ { in_dep=1; next }
        in_dep && /^[^[:space:]]/ { in_dep=0 }          # 顶格的新键，依赖段结束
        in_dep && /^[[:space:]]*-[[:space:]]*pip:/ { in_pip=1; next }
        in_pip { next }                                  # 进入 pip 段后不再算 conda
        in_dep && /^[[:space:]]{2}-[[:space:]]/ {
            line=$0
            sub(/^[[:space:]]*-[[:space:]]*/, "", line)
            print line
        }
    ' "$ENV_FILE"
}

parse_pip_deps() {
    awk '
        /^[[:space:]]*-[[:space:]]*pip:/ { in_pip=1; next }
        in_pip && /^[[:space:]]{6}-[[:space:]]/ {
            line=$0
            sub(/^[[:space:]]*-[[:space:]]*/, "", line)
            print line
        }
        in_pip && /^[[:space:]]{0,4}[^[:space:]-]/ { in_pip=0 }   # pip 段结束
    ' "$ENV_FILE"
}

# ----------------------------------------------------------------------------
# 步骤 1: 创建/确认环境
# ----------------------------------------------------------------------------
log_info "求解器: $CONDA_BIN"
log_info "目标环境: $ENV_NAME (python=$PY_VERSION)"

setup_solver

if conda env list | awk '{print $1}' | grep -qxF "$ENV_NAME"; then
    log_info "环境 '$ENV_NAME' 已存在，跳过创建。"
else
    log_info "创建环境 '$ENV_NAME' ..."
    # 用 conda 创建（mamba 不一定能建 base 外环境），并带上 libmamba 求解参数
    if conda create -y -n "$ENV_NAME" "python=$PY_VERSION" "${SOLVER_ARGS[@]}" >> "$DETAIL_LOG" 2>&1; then
        log_ok "环境创建成功"
    else
        log_err "环境创建失败，请查看 $DETAIL_LOG"
        log_err "若仍是 Killed（内存不足），请先按脚本顶部说明启用 libmamba 或添加 swap 后重试。"
        exit 1
    fi
fi

# ----------------------------------------------------------------------------
# 通用安装函数：逐包安装并降级重试
# conda 安装策略（依次尝试，任一成功即视为成功）：
#   1) 完整 spec   pkg=version=build
#   2) 去掉 build  pkg=version
#   3) 只留名字    pkg
# ----------------------------------------------------------------------------
install_conda_pkg() {
    local spec="$1"
    local name="${spec%%=*}"          # 包名（第一个 = 之前）
    local ver_build="${spec#*=}"
    local ver="${ver_build%%=*}"      # 版本（第二个 = 之前）

    if already_done "conda:$spec"; then
        log_ok "[conda] 已装过，跳过: $spec"
        return 0
    fi

    echo "============================================================" >> "$DETAIL_LOG"
    echo "[conda] $spec  $(date '+%F %T')" >> "$DETAIL_LOG"

    local try
    for try in "$spec" "$name=$ver" "$name"; do
        log_info "[conda] 尝试: $try"
        if $CONDA_BIN install -y -n "$ENV_NAME" "${SOLVER_ARGS[@]}" "$try" >> "$DETAIL_LOG" 2>&1; then
            log_ok "[conda] $spec  (实际: $try)"
            mark_success "conda:$spec"
            return 0
        fi
        echo ">>> 失败 spec: $try，尝试降级..." >> "$DETAIL_LOG"
    done

    log_err "[conda] $spec  (三种方式都失败，详见 $DETAIL_LOG)"
    mark_fail "conda:$spec"
    return 1
}

install_pip_pkg() {
    local spec="$1"

    if already_done "pip:$spec"; then
        log_ok "[pip] 已装过，跳过: $spec"
        return 0
    fi

    echo "============================================================" >> "$DETAIL_LOG"
    echo "[pip] $spec  $(date '+%F %T')" >> "$DETAIL_LOG"

    log_info "[pip] 安装: $spec"
    # --no-cache-dir 降低内存/磁盘峰值；--no-deps 避免逐包安装时反复拉取依赖导致冲突与额外内存
    if conda run -n "$ENV_NAME" python -m pip install --no-cache-dir --no-deps "$spec" >> "$DETAIL_LOG" 2>&1; then
        log_ok "[pip] $spec"
        mark_success "pip:$spec"
        return 0
    fi

    log_err "[pip] $spec  (失败，详见 $DETAIL_LOG)"
    mark_fail "pip:$spec"
    return 1
}

# ----------------------------------------------------------------------------
# 步骤 2: 逐个安装 conda 依赖
# ----------------------------------------------------------------------------
mapfile -t CONDA_DEPS < <(parse_conda_deps)
mapfile -t PIP_DEPS   < <(parse_pip_deps)

log_info "解析到 conda 依赖 ${#CONDA_DEPS[@]} 个, pip 依赖 ${#PIP_DEPS[@]} 个"

log_info "==================== 开始安装 conda 依赖 ===================="
for dep in "${CONDA_DEPS[@]}"; do
    [[ -z "$dep" ]] && continue
    install_conda_pkg "$dep"
done

# ----------------------------------------------------------------------------
# 步骤 3: 逐个安装 pip 依赖
# ----------------------------------------------------------------------------
log_info "==================== 开始安装 pip 依赖 ===================="
for dep in "${PIP_DEPS[@]}"; do
    [[ -z "$dep" ]] && continue
    install_pip_pkg "$dep"
done

# ----------------------------------------------------------------------------
# 步骤 4: 汇总
# ----------------------------------------------------------------------------
echo
echo "============================================================"
echo "                       安装结果汇总"
echo "============================================================"
SUCCESS_COUNT=$(wc -l < "$SUCCESS_LOG" | tr -d ' ')
FAIL_COUNT=$(grep -c . "$FAIL_LOG" 2>/dev/null || echo 0)

log_ok "成功: $SUCCESS_COUNT 个  (清单见 $SUCCESS_LOG)"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
    log_err "失败: $FAIL_COUNT 个  (清单见 $FAIL_LOG)"
    echo "------------------- 失败清单 -------------------"
    cat "$FAIL_LOG"
    echo "------------------------------------------------"
    echo "提示: 直接重跑本脚本即可，已成功的会自动跳过，只重试失败的。"
    echo "      逐包详细日志在: $DETAIL_LOG"
    exit 1
else
    log_ok "全部依赖安装成功！"
    echo "激活环境: conda activate $ENV_NAME"
    exit 0
fi