export PPL_PROJECT_ROOT=/workspace/ppl_v1.7.198-gcf5b037f-20260722
source /opt/tpuv7/tpuv7-current/data/tpuv7-bin-path.sh
export LD_LIBRARY_PATH=/opt/tpuv7/tpuv7-current/lib:$PPL_PROJECT_ROOT/deps/chip/tpub_7_1_e/lib:$PPL_PROJECT_ROOT/deps/runtime/tpuv7-runtime/lib:${LD_LIBRARY_PATH:-}
export PPL_TPUKERNEL_DEV_MODE=pcie
export SGLANG_USE_TPU=1
export SGLANG_TPU_DEVICE_ID=4
export PYTHONPATH=/workspace/sglang/python:$PYTHONPATH
