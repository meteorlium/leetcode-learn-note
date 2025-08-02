"""我要准备一个用于 LLM 算法岗面试的本地环境。需要安装哪些常用库，比如 torch、transformers？"""


"""
# 深度学习框架
# PyTorch生态系统
pip install torch torchvision torchaudio
pip install transformers  # Hugging Face transformers
pip install datasets      # Hugging Face datasets
pip install accelerate    # 分布式训练支持
pip install peft          # 参数高效微调 (LoRA等)
"""
import torch
import torchvision
import torchaudio
import transformers
import datasets
import accelerate
import peft

"""
NLP核心库 (个人感觉不太需要)
pip install tokenizers    # 高效tokenization
pip install sentencepiece # SentencePiece tokenizer
pip install nltk          # 传统NLP工具
pip install spacy         # 现代NLP库
"""

"""
# 数据处理与科学计算
pip install numpy pandas matplotlib seaborn
pip install scikit-learn  # 机器学习算法
pip install scipy         # 科学计算
"""

"""
# 可视化与监控
pip install wandb         # 实验跟踪
pip install tensorboard   # TensorBoard可视化
pip install plotly        # 交互式图表
"""

"""
# 模型推理与优化
pip install vllm          # 高性能LLM推理
pip install bitsandbytes  # 量化支持
pip install flash-attn    # Flash Attention (如果支持)
"""