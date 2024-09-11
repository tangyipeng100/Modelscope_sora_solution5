SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# for training
echo "[1/4] Installing ./train_evaluation"
cd ${SCRIPT_DIR}/train_infer_evaluate
pip install -r requirements.txt

# for evaluation
echo "[2/4] Installing ./train_evaluation/evaluation"
cd ${SCRIPT_DIR}/train_infer_evaluate/evaluation
pip install -r requirements.txt

# 项目中已经包含data juicer文件夹,可选择不安装.如果有报错,可考虑重新安装data juicer
# for data-juicer
#echo "[3/4] Installing ./data_juicer"
#cd ${SCRIPT_DIR}/data_juicer
#git pull origin main || true
#pip install '.[all]'

# for scene detect
echo "[4/4] Installing ./solution"
cd ${SCRIPT_DIR}/solution
pip install -r requirements.txt

echo "Done"
