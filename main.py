# %% [markdown]
# # Efficient Fine-Tuning of Qwen 2.5 Coder 14B with Unsloth
# 
# ## Introduction
# 
# Fine-tuning large language models (LLMs) such as Llama and Mistral has traditionally been a resource-intensive and time-consuming process. However, with advancements in model optimization techniques, it is now possible to streamline this workflow significantly. The provided Jupyter notebook demonstrates a comprehensive approach to efficiently fine-tune and deploy LLMs using the Unsloth library's `FastLanguageModel`. This workflow leverages key features like 4-bit quantization, Parameter-Efficient Fine-Tuning (PEFT) with LoRA, and optimized inference mechanisms to enhance performance while minimizing resource consumption.
# 
# The notebook begins by setting up the necessary environment, ensuring compatibility and optimal configurations through the installation and configuration of essential packages. It then proceeds to load a pre-trained model using `FastLanguageModel`, applying 4-bit quantization to reduce memory usage without compromising accuracy. The integration of PEFT via LoRA allows for focused fine-tuning of specific model parameters, further enhancing efficiency. Additionally, the notebook showcases data preparation techniques using chat templates and dataset standardization, ensuring that conversational data is appropriately formatted for training. The training process is meticulously configured to maximize performance, incorporating memory management strategies and selective training on response data to refine the model's output quality. Finally, the workflow includes steps for performing optimized inference, saving the fine-tuned model, and deploying it for real-world applicati
# ## Overview of `Qwen-2.5-Coder-14B`
# 
# Qwen-2.5-Coder-14B is a large language model specifically designed for coding tasks, part of the Qwen series of models developed to enhance code generation, reasoning, and fixing capabilities. This model features 14.7 billion parameters and is built on advanced transformer architecture, which includes techniques such as RoPE (Rotary Positional Encoding), SwiGLU (a type of activation function), and RMSNorm (Root Mean Square Layer Normalization) to optimize performance.
# 
# ### Key Features
# 
# - **Model Size**: 14.7 billion parameters, with 13.1 billion non-embedding parameters.
# - **Architecture**: Utilizes transformers with multiple layers (48 layers) and attention heads (40 for queries and 8 for keys/values).
# - **Context Length**: Supports long contexts of up to 131,072 tokens, allowing it to handle extensive code and text inputs effectively.
# - **Training Tokens**: Trained on a vast dataset of 5.5 trillion tokens, which includes a variety of source code and synthetic data.ons.
# 

# %% [markdown]
# ## Installation
# The code installs and upgrades necessary libraries for efficient model training and inference. This includes `unsloth` for fast training, `torch` for GPU operations, and `flash-attn` for optimized attention computation on compatible GPUs.

# %%
# #%%capture
# !pip install pip3-autoremove
# !pip-autoremove torch torchvision torchaudio -y
# !pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
# !pip install unsloth[kaggle-new]
# !pip uninstall unsloth -y && pip install git+https://github.com/unslothai/unsloth.git
# !pip install git+https://github.com/unslothai/unsloth-zoo.git
# !pip-autoremove unsloth unsloth-zoo -y
# !pip install unsloth
# !pip install jmespath

import os
os.environ["UNSLOTH_IS_PRESENT"] = "1"

# %% [markdown]
# **Explanation:**
# 
# This cell handles the installation and configuration of necessary Python packages required for the subsequent steps. Here's a breakdown:
# 
# 1. **Magic Command `%%capture`:**
#    - Suppresses the output of the cell, making the notebook cleaner by hiding the installation logs.
# 
# 2. **Installing `pip3-autoremove`:**
#    - `!pip install pip3-autoremove` installs a utility that allows for the removal of packages and their unused dependencies.
# 
# 3. **Removing Existing PyTorch Packages:**
#    - `!pip-autoremove torch torchvision torchaudio -y` ensures that any existing installations of `torch`, `torchvision`, and `torchaudio` are uninstalled. This is crucial to prevent version conflicts.
# 
# 4. **Reinstalling PyTorch with Specific CUDA Version:**
#    - `!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121` installs PyTorch along with `torchvision` and `torchaudio`, specifying the CUDA 12.1 version for GPU acceleration.
#    - `xformers` is also installed to leverage optimized transformer operations, enhancing model performance.
# 
# 5. **Installing the Unsloth Library:**
#    - `!pip install unsloth` installs the Unsloth library, which includes the `FastLanguageModel` and other utilities for efficient model fine-tuning and deployment.

# %%

# os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

# %%
from unsloth import FastLanguageModel, FastQwen2Model
from transformers import AutoModelForCausalLM



import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None          # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.
load_in_8bit = False

# model_name = "SmallThinker-3B-Preview-ablated"
# model_name = "PowerInfer/SmallThinker-3B-Preview"
# model_name = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
# model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
# model_name = "cognitivecomputations/Wizard-Vicuna-13B-Uncensored"
# model_name = "mlabonne/NeuralDaredevil-8B-abliterated"
# model_name = "neuraldaredevil_8B_dpo-mix-pt"
model_name = "MuntasirHossain/Orpo-Mistral-7B-v0.3"
# model_name = "unsloth/phi-4-unsloth-bnb-4bit"
# model_name = "unsloth/gemma-3-4b-it"
# model_name = "unsloth/Llama-3.2-3B-Instruct"
# model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
# model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
# model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"
# model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_8bit=load_in_8bit,
    load_in_4bit = load_in_4bit,
    #trust_remote_code=True
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
# model_name = "deepseek-ai/DeepSeek-V2-Lite"
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_4bit=True)
# model.save_pretrained_gguf("darkllama3.2_v5-f16.gguf", tokenizer, quantization_method = "f16")

# %%
# model.save_pretrained_merged("merged_model", tokenizer, save_method = "merged_16bit",)

# %% [markdown]
# **Explanation:**
# 
# This cell initializes the pre-trained language model using Unsloth's `FastLanguageModel`. Here's the step-by-step breakdown:
# 
# 1. **Imports:**
#    - `from unsloth import FastLanguageModel`: Imports the `FastLanguageModel` class from the Unsloth library.
#    - `import torch`: Imports PyTorch for tensor operations and GPU management.
# 
# 2. **Configuration Parameters:**
#    - `max_seq_length = 2048`: Sets the maximum sequence length for input data. FastLanguageModel supports RoPE Scaling internally, allowing dynamic adjustment for larger sequences.
#    - `dtype = None`: Specifies the data type for model weights. Setting to `None` enables automatic detection based on the hardware. Alternatives include `float16` for GPUs like Tesla T4 or V100, and `bfloat16` for newer GPUs like Ampere.
#    - `load_in_4bit = True`: Enables 4-bit quantization to reduce memory usage, facilitating the loading of larger models without running out of memory (OOM). Setting to `False` would load the model in higher precision but with higher memory consumption.
# 
# 3. **Model Lists:**
#    - **`fourbit_models`:** A list of pre-quantized 4-bit models supported by Unsloth for faster downloading and reduced memory footprint. Examples include various versions of Llama, Mistral, Phi, and Gemma models.
#    - **`qwen_models`:** A list of Qwen models optimized for different sizes and instructions. These models benefit from faster inference times and efficient memory usage.
# 
# 4. **Loading the Model and Tokenizer:**
#    - `FastLanguageModel.from_pretrained(...)` is called with the specified parameters to load the pre-trained model and its tokenizer.
#    - **Parameters:**
#      - `model_name`: Specifies the exact model to load. In this case, `"unsloth/Qwen2.5-Coder-14B-Instruct"` is chosen from the `qwen_models` list.
#      - `max_seq_length`, `dtype`, `load_in_4bit`: Pass the previously defined configuration parameters.
#      - `token`: (Commented out) Allows specifying a token for gated models if needed.familiar with Hugging Face.

# %%
# from torchinfo import summary
# summary(model, depth=3, verbose=True)

# %%
layers_to_transform = []

# Iterar sobre os parâmetros do modelo
# a = model.state_dict()
# print(model.state_dict()['base_model.model.lm_head.weight'])

for nome, parametro in model.named_parameters():
    print(nome)
    # if nome == "base_model.model.lm_head.weight":
    # #     # nome = "lm_head_weight"
    #     a['base_model.model.lm_head_weight'] = parametro
    #     a.pop("base_model.model.lm_head.weight")
    # else:
    #     a[nome] = parametro
    # if 'embed' in nome.lower():
        #print(nome)
        # ...
        #parametro.requires_grad = True
        

        #param = model.model.embed_tokens.weight
       # param.data = param.data.to(torch.float32)
   # else:
        #parametro.requires_grad = False
#layers_to_transform
# print(a['base_model.model.lm_head_weight'])
# print(a)



# %%
# from transformers import AutoTokenizer, AutoModelForCausalLM
# # model_name = 'darkllama3.2_v6'

# # model = AutoModelForCausalLM.from_pretrained(model_name)
# model_name = 'darkllama3.2_v6'
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model.save_pretrained('darkllama3.2_v6_hf',  safe_serialization=False)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained('darkllama3.2_v6_hf',  safe_serialization=False)

# model_name = 'Lyte/Llama-3.2-3B-Overthinker'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained('overthinker_hf',  safe_serialization=False)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model.save_pretrained('overthinker_hf',  safe_serialization=False)

# %%
# import os
# os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r = 256, #32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",], #["q_proj", "v_proj"], #
    #modules_to_save=["embed_tokens"],#["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,#32,#16,
    lora_dropout = 0.1,   #0 # Supports any, but = 0 is optimized
    bias = "all", #"none",       # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
model.print_trainable_parameters()
# del base_model
# import gc
# gc.collect()

# %% [markdown]
# Claro! Vamos destrinchar cada um dos parâmetros que você usou no `FastLanguageModel.get_peft_model`:
# 
# ---
# 
# ### Parâmetros:
# 
# | Nome do Parâmetro                  | Explicação |
# |:-----------------------------------|:-----------|
# | **`model`**                        | O modelo base (normalmente um modelo Transformer) no qual você quer aplicar LoRA (Low-Rank Adaptation). Você passa o modelo já carregado aqui. |
# | **`r`**                            | O rank da decomposição de matrizes para LoRA. <br>É a quantidade de "componentes internos" adicionados aos parâmetros do modelo. <br>**Quanto maior o `r`**, maior a capacidade de aprendizado adicional — **mas também maior o uso de memória VRAM**. <br>Valores comuns: 8, 16, 32, 64, 128. |
# | **`target_modules`**               | Quais camadas do modelo você quer aplicar LoRA. <br>Ex: `"q_proj"`, `"k_proj"`, etc. <br>Esses nomes geralmente são os nomes das subcamadas do Transformer (por exemplo, projeções de atenção e MLPs). |
# | **`lora_alpha`**                   | Fator de escalonamento para os parâmetros LoRA. <br>Ajuda a equilibrar o quanto o LoRA afeta o modelo. <br>**Formula comum:** <br>`final_weight = base_weight + (lora_alpha/r) * lora_weight` <br>Então, se `lora_alpha` for maior, o LoRA influencia mais o modelo. |
# | **`lora_dropout`**                 | Probabilidade de dropout aplicado dentro dos módulos LoRA. <br>**Se `0`,** é mais otimizado (sem dropout). Normalmente, usa-se dropout para regularizar — mas se você quer eficiência máxima, usa `0`. |
# | **`bias`**                         | Define como tratar os biases (tendências) nas camadas. <br>**"none"** = não aplica LoRA nos bias (mais leve e mais rápido). <br>Outros valores possíveis: `"all"` ou `"lora_only"`. |
# | **`use_gradient_checkpointing`**   | Técnica para reduzir o uso de memória (VRAM) durante o treinamento. <br>**"unsloth"** é uma versão especial otimizada (30% menos VRAM comparado ao normal). <br>Você também pode usar `True` (normal, mas consome mais VRAM). |
# | **`random_state`**                 | Semente para a geração aleatória. <br>Permite que o treinamento seja **reprodutível** (gera sempre os mesmos resultados se a semente for fixa). |
# | **`use_rslora`**                   | Ativa **Rank-Stabilized LoRA**. <br>Uma técnica para melhorar a estabilidade do LoRA durante o treinamento, especialmente útil em ranks altos (`r=64`+). Evita que o treinamento fique instável. |
# | **`loftq_config`**                 | **LoFTQ** (Low-rank Fine-Tuning Quantization) — permite aplicar **quantização** diretamente no treinamento de LoRA. <br>Se for `None`, a quantização não é usada. É útil para reduzir ainda mais o tamanho do modelo. |
# 
# ---
# 
# ### Adicional:
# - **`model.print_trainable_parameters()`**  
#   Mostra **quantos parâmetros** do modelo estão sendo treinados.  
#   Como LoRA congela o modelo base e só adiciona pequenas camadas extras, você verá que **uma fração muito pequena** dos parâmetros será treinada (~0,1% a 1%).
# 
# ---
# 
# ### **Resumo visual:** 🎯
# 
# - `r`: quão grande é a adaptação.
# - `target_modules`: onde aplicar a adaptação.
# - `lora_alpha`: quanto o LoRA pesa no resultado final.
# - `lora_dropout`: regularização interna da LoRA.
# - `bias`: se adapta também os biases ou não.
# - `use_gradient_checkpointing`: economizar memória.
# - `random_state`: tornar os resultados reprodutíveis.
# - `use_rslora`: deixar o LoRA mais estável.
# - `loftq_config`: usar quantização durante fine-tuning.
# 
# ---
# 
# Vou te mostrar **como** cada parâmetro que você listou **impacta de fato** a **qualidade do fine-tuning** na prática — **baseado em experimentação real** e **papers de LoRA/PEFT**.
# 
# ---
# 
# # 🎯 Efeitos práticos na qualidade do fine-tuning
# 
# | Parâmetro                   | Se você aumentar...                     | Se você diminuir...                     | Impacto na Qualidade |
# |:-----------------------------|:----------------------------------------|:----------------------------------------|:----------------------|
# | `r`                          | + Capacidade de adaptação, + Qualidade (até certo ponto). | - Menos capacidade, risco de underfitting. | 🟢 Crítico. Mais r, melhor — mas VRAM explode se exagerar. Recomendo 64~128 para modelos grandes. |
# | `target_modules`             | + Cobertura dos componentes do modelo → adapta mais coisas. | - Menos cobertura → só adapta uma parte pequena. | 🟢 Muito importante. Se cobrir poucas camadas (ex: só q_proj/v_proj), o modelo adapta menos bem. |
# | `lora_alpha`                 | + A LoRA influencia mais. Pode ajudar se r for pequeno. | - LoRA influencia pouco. Pode impedir o fine-tuning de aprender direito. | 🟡 Moderado. Se `alpha` for desbalanceado (muito maior ou muito menor que `r`), pode causar under/overfitting. Ideal: `lora_alpha ≈ r * 2` ou similar. |
# | `lora_dropout`               | + Mais regularização. Evita overfitting em datasets pequenos. | - Sem regularização. Melhor para datasets grandes e otimizados. | 🟡 Pequeno efeito, mas ajuda em datasets ruidosos. Se treino é curto ou dataset pequeno, use 0.05~0.1. |
# | `bias`                       | + Se treinar os biases, o modelo pode capturar nuances mais finas. | - Mais leveza, mas menos capacidade de adaptação. | 🔵 Efeito muito pequeno. "none" é ok quase sempre. Só use "all" se seu dataset for pequeno e diverso. |
# | `use_gradient_checkpointing` | Não afeta qualidade diretamente. Só economiza VRAM. | Não afeta qualidade diretamente. | ⚪ Nenhum impacto direto na qualidade final. |
# | `use_rslora`                 | + Mais estabilidade em `r` alto (>64). Menos risco de explodir loss. | Sem estabilização, pode haver instabilidade numérica em ranks grandes. | 🟢 Importante se `r` >= 64. Com `r` baixo (ex: 8 ou 16), pouco efeito. |
# | `loftq_config`               | + Permite quantizar e treinar juntos. Pode perder ligeiramente a precisão. | Não quantiza → preserva precisão total. | 🔴 Pode degradar ligeiramente qualidade (~0.1%~0.5% em benchmarks). Só vale usar se falta VRAM. |
# 
# ---
# 
# # 🧠 Resumindo em palavras simples:
# 
# - **Se você quer maximizar qualidade**:  
#   - **r** = alto (64, 128) ✅  
#   - **target_modules** = abrangente (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) ✅  
#   - **lora_alpha** = proporcional (ex: `r * 2`) ✅  
#   - **lora_dropout** = 0 (se dataset for grande/limpo) ou 0.05 (se dataset for pequeno/ruidoso) ✅  
#   - **use_rslora** = **Sim**, especialmente para `r>=64` ✅  
#   - **loftq_config** = **Não usar**, se a prioridade é a qualidade máxima ✅
# 
# - **Se quiser ganhar eficiência sacrificando um pouco de qualidade**:
#   - Use `loftq_config`
#   - Reduza `r` para 16 ou 32
#   - Use menos `target_modules`
# 
# ---
# 
# # 📊 Tabela rápida: Configurações "ótimas"
# 
# | Objetivo                     | Configuração recomendada |
# |:------------------------------|:-------------------------|
# | Máxima qualidade              | r = 128, alpha = 256, todos target_modules, dropout = 0 |
# | Melhor custo/benefício        | r = 64, alpha = 128, alguns target_modules, dropout = 0.05 |
# | Treinar no limite da VRAM      | r = 16, alpha = 32, poucos modules (`q_proj`, `v_proj`), loftq_config ativado |
# 
# ---
# 
# # 🧩 Exemplo prático real:
# 
# Quando se usa **`r=8`** em um modelo LLaMA2-7B, o modelo consegue reter 75~80% da performance do full fine-tuning.
# 
# Quando se usa **`r=64`**, o mesmo modelo alcança 95~97% da performance do full fine-tuning — mas com **20% a mais de custo VRAM**.
# 
# ✅ Resultado validado em benchmarks como **OpenAssistant**, **Vicuna Evaluation**, **MMLU**.
# 
# ---
# 

# %% [markdown]
# **Explanation:**
# 
# This cell configures the model for Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) via Unsloth's `get_peft_model` method. Here's the breakdown:
# 
# 1. **PEFT Configuration:**
#    - **`model`:** The pre-trained model loaded in the previous cell is passed as input to apply PEFT.
# 
# 2. **LoRA Parameters:**
#    - `r = 16`: The rank of the LoRA matrices. Higher values allow more capacity but consume more memory. Suggested values range from 8 to 128.
#    - `target_modules`: Specifies which modules within the model to apply LoRA. In this case, projection layers like `q_proj`, `k_proj`, `v_proj`, etc., are targeted.
#    - `lora_alpha = 16`: A scaling factor for LoRA. Balances the contribution of the LoRA layers to the model's output.
#    - `lora_dropout = 0`: Dropout rate applied within LoRA layers. Set to `0` for optimized performance.
#    - `bias = "none"`: Indicates that biases are not being fine-tuned. This setting is optimized for memory and performance.
# 
# 3. **Additional Configurations:**
#    - `use_gradient_checkpointing = "unsloth"`: Enables gradient checkpointing to save memory during training. The `"unsloth"` setting optimizes VRAM usage, allowing for larger batch sizes and handling longer contexts.
#    - `random_state = 3407`: Sets the seed for reproducibility.
#    - `use_rslora = False`: Indicates whether to use Rank Stabilized LoRA. Set to `False` in this configuration.
#    - `loftq_config = None`: Placeholder for additional LoftQ configurations if needed.
# 
# 4. **Applying PEFT:**
#    - `FastLanguageModel.get_peft_model(...)` modifies the original model to include LoRA layers, enabling efficient fine-tuning without updating all model parameters.

# %% [markdown]
# ## Data Preprocessing

# %%
# map R1 dataset
# from datasets import load_dataset
# from unsloth.chat_templates import get_chat_template
# from unsloth.chat_templates import standardize_sharegpt



# ServiceNow-AI/R1-Distill-SFT

# %%
# from datasets import load_dataset
# from unsloth.chat_templates import get_chat_template
# from unsloth.chat_templates import standardize_sharegpt

# tokenizer = get_chat_template(
#     tokenizer,
#     # chat_template = "qwen-2.5",
#     chat_template='chatml'
#     # chat_template = "llama-3",
# )

# def formatting_prompts_func(examples):
#     convos = examples["conversations"]
#     texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
#     return { "text" : texts, }

# dataset = load_dataset("UnfilteredAI/DAN", split="train")
# # UnfilteredAI/DAN
# # Triangle104/Guilherme34-uncensor
# dataset = dataset.rename_column("conversation", "conversations")
# # dataset = load_dataset("mlabonne/FineTome-100k", split="train")
# print(dataset[5])
# dataset = standardize_sharegpt(dataset)
# dataset[5]
# dataset = dataset.map(formatting_prompts_func, batched=True)

# %%
# from datasets import load_dataset
# from unsloth.chat_templates import get_chat_template
# from unsloth.chat_templates import standardize_sharegpt
# def convert_to_sharegpt(text):
#     lines = text.strip().split("\n")
#     messages = []
#     current_role = None
#     current_content = []

#     for line in lines:
#         if line.startswith("Human:"):
#             if current_role:
#                 messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
#             current_role = "user"
#             current_content = [line.replace("Human:", "").strip()]
#         elif line.startswith("Assistant:"):
#             if current_role:
#                 messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
#             current_role = "assistant"
#             current_content = [line.replace("Assistant:", "").strip()]
#         else:
#             current_content.append(line.strip())

#     if current_role:
#         messages.append({"role": current_role, "content": "\n".join(current_content).strip()})

#     return messages

# def aaa(examples):
#     convos = examples["conversations"]
#     texts = [convert_to_sharegpt(convo) for convo in convos]
#     return {"conversations": texts}

# def formatting_prompts_func(examples):
#     convos = examples["conversations"]
#     texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
#     return { "text" : texts, }

# "UnfilteredAI/DAN"
# "maywell/hh-rlhf-nosafe"
# dataset = load_dataset("maywell/hh-rlhf-nosafe", split="train")
# dataset = dataset.rename_column("rejected", "conversations")
# dataset = dataset.remove_columns('chosen')
# # b = convert_to_sharegpt(dataset[5]['rejected'])
# # print(b)
# dataset = dataset.map(aaa, batched=True)
# # print(dataset[5])
# dataset = standardize_sharegpt(dataset)
# dataset = dataset.map(formatting_prompts_func, batched=True)

# %%
# from datasets import load_dataset

# dataset = load_dataset("wikimedia/wikipedia", "20231101.pt", split = "train",)
# EOS_TOKEN = tokenizer.eos_token
# def formatting_prompts_func(examples):
#     return { "text" : [example + EOS_TOKEN for example in examples["text"]] }

# # We select 1% of the data to make training faster!
# dataset = dataset.train_test_split(train_size = 0.10)["train"]

# dataset = dataset.map(formatting_prompts_func, batched = True,)

# %%
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt

tokenizer = get_chat_template(
    tokenizer,
    # chat_template = "qwen-2.5",
    chat_template='chatml'
    # chat_template = "llama-3.1",
)
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    # print(convo for convo in convos)
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts, }

EOS_TOKEN = tokenizer.eos_token
dataset = load_dataset("BornSaint/D33", split="train") 
# dataset = load_dataset("BornSaint/orpo-dpo-mix-40k_portuguese", split="train") 
# dataset = dataset.rename_column("chosen", "conversations")
# guilherme34_dataset = load_dataset("BornSaint/Guilherme34_uncensor_portuguese", split="train") 
 
# sharegpt_data = []

# for example in guilherme34_dataset:  # ou outro split, ex: dataset['validation']
#     conversation = {
#         "conversations": [
            
#             {"role": "user", "content": example['question']},
#             {"role": "assistant", "content": "estou respondendo de forma imoral e antiética, isto é uma contradição contra meus princípios: " + example['answer']}
#         ]
#     }
#     sharegpt_data.append(conversation)
# guilherme_sharegpt = Dataset.from_list(sharegpt_data)

# d33_1 = load_dataset("BornSaint/D33_590d", split="train") 
# d33_1 = d33_1.rename_column("conversation", "conversations")
# d33_2 = load_dataset("BornSaint/D33_alignment", split='train')
# d33_2 = d33_2.rename_column("conversation", "conversations")
# print(d33_1)
# print(d33_2)
# dataset = concatenate_datasets([dataset, guilherme_sharegpt, d33_1, d33_2])
# dataset = load_from_disk("../axiom_GPT_responses")
# dataset = dataset.rename_column("conversation", "conversations")
# print(dataset)


# print(dataset)
# dataset = dataset.train_test_split(train_size = 0.5)["train"]

# dataset = standardize_sharegpt(dataset)
# dataset = dataset.map(formatting_prompts_func, batched=True)
# dataset.save_to_disk('chatml_dpo_mix-pt_and_guilherme34-pt_and_D33full')



# %% [markdown]
# **Explanation:**
# 
# This cell handles dataset loading, standardization, and formatting using Unsloth's utilities for chat-based templates. Here's the step-by-step explanation:
# 
# 1. **Imports:**
#    - `from datasets import load_dataset`: Imports the `load_dataset` function from Hugging Face's `datasets` library for dataset handling.
#    - `from unsloth.chat_templates import get_chat_template, standardize_sharegpt`: Imports Unsloth's functions for handling chat templates and standardizing ShareGPT datasets.
# 
# 2. **Configuring the Tokenizer with a Chat Template:**
#    - `get_chat_template(tokenizer, chat_template = "qwen-2.5")`: Configures the tokenizer to use the `"qwen-2.5"` chat template. This ensures that the input data aligns with the expected format of the Qwen-2.5 model.
#    - **Purpose:** Formats conversational data by mapping roles (e.g., "user" and "assistant") to specific identifiers, facilitating better interaction with the model.
# 
# 3. **Defining the Formatting Function:**
#    - `formatting_prompts_func(examples)`: A function to process each batch of dataset examples.
#      - `convos = examples["conversations"]`: Extracts the conversation data from the dataset.
#      - `texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]`: Applies the chat template to each conversation without tokenization and without adding a generation prompt.
#      - `return { "text" : texts, }`: Returns the formatted texts in a dictionary with the key `"text"`.
# 
# 4. **Loading and Standardizing the Dataset:**
#    - `dataset = load_dataset("mlabonne/FineTome-100k", split="train")`: Loads the `FineTome-100k` dataset's training split. This dataset is assumed to follow the ShareGPT format, which consists of multi-turn conversations.
#    - `dataset = standardize_sharegpt(dataset)`: Transforms the ShareGPT-formatted dataset into a Hugging Face-compatible format using Unsloth's `standardize_sharegpt` function. This involves restructuring the data to merge multiple fields into single input-output pairs suitable for training.
#    - `dataset = dataset.map(formatting_prompts_func, batched=True)`: Applies the previously defined formatting function to each batch of the dataset, effectively preparing the conversational data for training.

# %%
# dataset[5]["conversations"]

# %%
# dataset[5]["text"]

# %% [markdown]
# ## Model Training

# %%
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    # data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 16,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4, # Fixed major bug in latest Unsloth
        warmup_steps = 5,
        num_train_epochs = 2, # Set this for 1 full training run.
        # max_steps = 1200,
        learning_rate = 2e-4, #2e-4
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        # save_steps=200,
        # save_total_limit=10,
        optim = "paged_adamw_8bit", # Save more memory
        weight_decay = 0.01, #0.01
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# %% [markdown]
# 
# ## 🧩 Parte 1: Argumentos do `SFTTrainer`
# 
# | Parâmetro                     | O que faz                                          | Efeito prático                                                                                           |
# | ----------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
# | `model`, `tokenizer`          | Modelo e tokenizer LoRA a serem treinados          | 🟢 Essencial                                                                                             |
# | `train_dataset`               | Dataset de treino                                  | 🟢 Base de tudo                                                                                          |
# | `dataset_text_field = "text"` | Nome do campo com o texto no seu dataset           | 🟢 Corrigir isso evita bugs (ex: `KeyError`)                                                             |
# | `max_seq_length`              | Máximo de tokens por input                         | 🔵 Afeta eficiência. Valores maiores treinam mais lentamente, mas capturam mais contexto.                |
# | `dataset_num_proc = 16`       | Nº de processos paralelos para processar o dataset | ⚪️ Acelera o carregamento do dataset. Ideal: CPUs disponíveis                                            |
# | `packing = False`             | Junta vários textos curtos num único batch         | 🟡 Com `False`: mantém frases separadas. `True`: acelera mas pode distorcer aprendizado em dados curtos. |
# 
# 🔎 **Dica prática:**
# Para *diálogos curtos ou tarefas de instrução*, `packing=False` é melhor para qualidade.
# Para *datasets longos ou geração contínua*, `packing=True` economiza tempo, mas com risco leve de confusão entre amostras.
# 
# ---
# 
# ## 🧩 Parte 2: `TrainingArguments(...)`
# 
# Essa parte define o **coração da lógica de treino**:
# 
# | Parâmetro                         | O que faz                                          | Efeito prático                                                                    |
# | --------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------- |
# | `per_device_train_batch_size = 1` | Nº de exemplos por GPU antes de acumular           | 🔴 Extremamente baixo — necessário se VRAM for limitada.                          |
# | `gradient_accumulation_steps = 4` | Nº de passos até aplicar gradiente                 | 🟡 Simula batch maior. Aqui: batch efetivo = 1 × 4 = 4                            |
# | `warmup_steps = 5`                | Passos iniciais com LR baixo (evita instabilidade) | 🟡 Evita picos de perda no início. Curto, mas melhor que zero.                    |
# | `num_train_epochs = 1`            | Quantas vezes passar por todo o dataset            | 🟢 Define cobertura total. Para mais qualidade, aumente (ex: 3\~5)                |
# | `learning_rate = 2e-4`            | Taxa de aprendizado inicial                        | 🟢 Muito importante. `2e-4` é ótimo para LoRA. Taxas maiores = risco de divergir. |
# | `fp16` / `bf16`                   | Tipos de precisão mista (float16 / bfloat16)       | 🟡 Acelera. Use `bf16` se suportado (menos instável que `fp16`).                  |
# | `logging_steps = 1`               | Loga a cada passo                                  | 🟡 Bom para monitorar overfitting, mas mais lento em disco.                       |
# | `optim = "paged_adamw_8bit"`      | Otimizador de memória RAM/GPU                      | 🟢 Crucial para caber em GPUs pequenas. Ideal com LoRA.                           |
# | `weight_decay = 0.01`             | Penaliza pesos muito grandes (regularização)       | 🟡 Bom para generalização. Evita overfitting leve.                                |
# | `lr_scheduler_type = "linear"`    | Como o LR decai com o tempo                        | 🟢 Linear = decresce suavemente. Funciona bem para fine-tuning.                   |
# | `seed = 3407`                     | Define aleatoriedade para reprodutibilidade        | 🟡 Garantia de consistência.                                                      |
# | `output_dir = "outputs"`          | Pasta de saída                                     | ⚪ Armazena checkpoints e logs.                                                    |
# | `report_to = "none"`              | Integrações com loggers (ex: WandB)                | ⚪ Apenas log local. Se quiser métricas visuais, use `"wandb"`                     |
# 
# ---
# 
# ## ✅ Efeitos práticos no treinamento
# 
# | Objetivo                           | Como esses parâmetros ajudam                                     |
# | ---------------------------------- | ---------------------------------------------------------------- |
# | **Rodar em GPU fraca (6-12 GB)**   | `batch_size=1`, `8bit`, `LoRA`, `accum_steps=4`                  |
# | **Evitar instabilidade no início** | `warmup_steps=5`, `linear` scheduler                             |
# | **Evitar overfitting**             | `weight_decay=0.01`, `dropout`, `packing=False`                  |
# | **Reproduzibilidade**              | `seed=3407`, log a cada passo                                    |
# | **Boa performance geral**          | `lr=2e-4`, `adamw_8bit`, `alpha`, `target_modules` bem definidos |
# 
# ---
# 
# ## 🧪 Conclusão prática
# 
# ✅ **Com esses parâmetros**, você já tem uma base **ótima** para fine-tuning de um modelo com LoRA **em ambiente restrito (pouca GPU)**.
# 
# ⚠️ **Se quiser priorizar qualidade acima de tudo**, considere:
# 
# * `batch_size=2 ou 4` (se possível)
# * `gradient_accumulation_steps` maior para simular batch efetivo maior
# * `num_train_epochs = 3+` para aprender de fato
# * `packing = True` se os textos forem muito curtos e o modelo estiver lento
# 
# ---
# 
# 
# 

# %% [markdown]
# **Explanation:**
# 
# This cell configures the training setup using `SFTTrainer` from the `trl` (Transformers Reinforcement Learning) library, incorporating Unsloth's utilities for optimized training. Here's the detailed breakdown:
# 
# 1. **Imports:**
#    - `from trl import SFTTrainer`: Imports the `SFTTrainer` class for supervised fine-tuning.
#    - `from transformers import TrainingArguments, DataCollatorForSeq2Seq`: Imports necessary classes from Hugging Face's `transformers` library for training configurations and data collation.
#    - `from unsloth import is_bfloat16_supported`: Imports a utility function to check hardware support for `bfloat16`.
# 
# 2. **Configuring the Trainer:**
#    - `trainer = SFTTrainer(...)`: Initializes the trainer with specified configurations.
#    
# 3. **Trainer Parameters:**
#    - `model = model`: The PEFT-enabled model from the previous cell.
#    - `tokenizer = tokenizer`: The tokenizer configured with the chat template.
#    - `train_dataset = dataset`: The prepared training dataset.
#    - `dataset_text_field = "text"`: Specifies the field in the dataset containing the input text.
#    - `max_seq_length = max_seq_length`: Sets the maximum sequence length as defined earlier.
#    - `data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer)`: Utilizes a data collator suitable for sequence-to-sequence tasks, ensuring that batches are correctly formatted.
#    - `dataset_num_proc = 4`: Number of processes to use for data loading and preprocessing.
#    - `packing = False`: Disables packing of multiple short sequences into a single batch, which can speed up training for short sequences.
# 
# 4. **Training Arguments (`args = TrainingArguments(...)`):**
#    - `per_device_train_batch_size = 1`: Sets the batch size per device. Given memory constraints, a batch size of 1 is used.
#    - `gradient_accumulation_steps = 4`: Accumulates gradients over 4 steps before performing an optimization step. This effectively increases the batch size without increasing memory usage.
#      - **Note:** Mention of fixing a major bug in Unsloth indicates that this parameter is crucial for stability.
#    - `warmup_steps = 5`: Number of warmup steps for learning rate scheduling.
#    - `max_steps = 30`: Maximum number of training steps. Uncommenting `num_train_epochs = 1` would set the training to run for one full epoch instead.
#    - `learning_rate = 2e-4`: Sets the learning rate for the optimizer.
#    - `fp16 = not is_bfloat16_supported()`: Enables mixed-precision training with `float16` if `bfloat16` is not supported.
#    - `bf16 = is_bfloat16_supported()`: Enables `bfloat16` precision if supported by the hardware.
#    - `logging_steps = 1`: Logs training metrics every step.
#    - `optim = "paged_adamw_8bit"`: Uses the `paged_adamw_8bit` optimizer to save memory.
#    - `weight_decay = 0.01`: Sets the weight decay for regularization.
#    - `lr_scheduler_type = "linear"`: Uses a linear learning rate scheduler.
#    - `seed = 3407`: Sets the random seed for reproducibility.
#    - `output_dir = "outputs"`: Directory where training outputs and checkpoints will be saved.
#    - `report_to = "none"`: Disables reporting to external services like WandB. This can be changed to enable integration with monitoring tools.
# 
# 5. **Precision Handling:**
#    - `is_bfloat16_supported()` checks if the current GPU supports `bfloat16`. If supported, `bf16` is enabled; otherwise, `fp16` is used. This ensures optimal precision based on hardware capabilities.

# %%
from unsloth.chat_templates import train_on_responses_only

# trainer = train_on_responses_only(
#     trainer,
#     instruction_part = "<|im_start|>user\n",
#     response_part = "<|im_start|>assistant\n",
   
# )
# trainer = train_on_responses_only(
#     trainer, 
#     instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
#     response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
# )

# %% [markdown]
# **Explanation:**
# 
# This cell modifies the training process to focus exclusively on the response segments of the dataset, effectively ignoring the input prompts. Here's the detailed explanation:
# 
# 1. **Importing the Function:**
#    - `from unsloth.chat_templates import train_on_responses_only`: Imports the `train_on_responses_only` function from Unsloth's chat templates module.
# 
# 2. **Applying the Function:**
#    - `trainer = train_on_responses_only(...)`: Modifies the `trainer` instance to train only on the response parts of the dataset.
#    
# 3. **Parameters:**
#    - `trainer`: The existing `SFTTrainer` instance configured in the previous cell.
#    - `instruction_part = "<|im_start|>user\n"`: Specifies the prefix that identifies the instruction or user input in the dataset.
#    - `response_part = "<|im_start|>assistant\n"`: Specifies the prefix that identifies the assistant's response in the dataset.

# %%
# tokenizer.decode(trainer.train_dataset[5]["input_ids"])

# %%
# space = tokenizer(" ", add_special_tokens = False).input_ids[0]
# tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])

# %% [markdown]
# **Explanation:**
# 
# This cell provides a way to inspect and verify the formatting of the training data after preprocessing. Here's the step-by-step breakdown:
# 
# 1. **Decoding Input IDs:**
#    - `tokenizer.decode(trainer.train_dataset[5]["input_ids"])`: Decodes the input IDs of the sixth example (index `5`) in the training dataset back into human-readable text. This helps in verifying that the input data has been correctly tokenized and formatted.
# 
# 2. **Decoding Labels with Special Handling:**
#    - `space = tokenizer(" ", add_special_tokens = False).input_ids[0]`: Retrieves the token ID for a single space character. This is used to replace special tokens in the labels.
#    - `tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])`: Decodes the label IDs of the same example, replacing any occurrence of `-100` (a special masking value often used in loss calculations) with the space token. This ensures that the decoded labels are readable and free from masking artifacts.
# 

# %%
#@title Show current memory stats
# gpu_stats = torch.cuda.get_device_properties(0)
# start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

# print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
# print(f"{start_gpu_memory} GB of memory reserved.")

# %%
# layers_to_transform

# %%
trainer_stats = trainer.train(resume_from_checkpoint=False)

# %% [markdown]
# **Explanation:**
# 
# This cell starts the training process and captures the training statistics upon completion.
# 
# 1. **Starting Training:**
#    - `trainer_stats = trainer.train()`: Initiates the training loop using the configured `trainer`. The `train()` method runs the training process based on the previously defined `TrainingArguments` and dataset.
# 
# 2. **Capturing Training Statistics:**
#    - The result of the `train()` method is stored in `trainer_stats`, which contains metrics and information about the training run, such as runtime, loss values, and other performance indicators.
# 

# %%
#@title Show final memory and time stats
# used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
# used_percentage = round(used_memory         /max_memory*100, 3)
# lora_percentage = round(used_memory_for_lora/max_memory*100, 3)

# print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
# print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
# print(f"Peak reserved memory = {used_memory} GB.")
# print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
# print(f"Peak reserved memory % of max memory = {used_percentage} %.")
# print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# %% [markdown]
# ## Model Inferencing

# %%
# from unsloth.chat_templates import get_chat_template

# # tokenizer = get_chat_template(
# #     tokenizer,
# #     # chat_template = "qwen-2.5",
# #     chat_template='chatml'
# #     chat_template="llama-3.1"
# # )
# FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# messages = [
#     {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
# ]
# inputs = tokenizer.apply_chat_template(
#     messages,
#     tokenize = True,
#     add_generation_prompt = True, # Must add for generation
#     return_tensors = "pt",
# ).to("cuda")

# outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
#                          temperature = 1.5, min_p = 0.1)
# tokenizer.batch_decode(outputs)

# %% [markdown]
# **Explanation:**
# 
# This cell demonstrates how to perform inference (generate responses) using the trained model with optimized settings. Here's the detailed breakdown:
# 
# 1. **Importing and Configuring the Tokenizer:**
#    - `from unsloth.chat_templates import get_chat_template`: Imports the `get_chat_template` function.
#    - `tokenizer = get_chat_template(tokenizer, chat_template = "qwen-2.5")`: Reapplies the `"qwen-2.5"` chat template to the tokenizer to ensure consistency with training.
# 
# 2. **Enabling Optimized Inference:**
#    - `FastLanguageModel.for_inference(model)`: Activates native optimizations within the `FastLanguageModel` to enable faster inference, potentially doubling the speed.
# 
# 3. **Preparing the Input Message:**
#    - `messages = [{"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},]`: Defines a prompt asking the model to continue the Fibonacci sequence.
#    - `inputs = tokenizer.apply_chat_template(...)`: Applies the chat template to the messages.
#      - `tokenize = True`: Tokenizes the input text.
#      - `add_generation_prompt = True`: Adds necessary prompts for the model to generate a response.
#      - `return_tensors = "pt"`: Returns the inputs as PyTorch tensors.
#      - `.to("cuda")`: Moves the input tensors to the GPU for faster computation.
# 
# 4. **Generating the Output:**
#    - `outputs = model.generate(...)`: Generates text based on the input.
#      - `input_ids = inputs`: Passes the tokenized input.
#      - `max_new_tokens = 64`: Limits the generation to 64 new tokens.
#      - `use_cache = True`: Utilizes caching for faster generation.
#      - `temperature = 1.5`: Increases randomness in generation, leading to more diverse outputs.
#      - `min_p = 0.1`: Sets the minimum probability threshold for nucleus sampling.
# 
# 5. **Decoding the Output:**
#    - `tokenizer.batch_decode(outputs)`: Converts the generated token IDs back into human-readable text.
# 

# %%
# FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# messages = [
#     {"role": "user", "content": "Continue the fibonnaci sequence with more 10 numbers: 1, 1, 2, 3, 5, 8,"},
# ]
# inputs = tokenizer.apply_chat_template(
#     messages,
#     tokenize = True,
#     add_generation_prompt = True, # Must add for generation
#     return_tensors = "pt",
# ).to("cuda")
# messages = [
#     {"role": "user", "content": "baca tem pernas?"},
# ]
# inputs = tokenizer.apply_chat_template(
#     messages,
#     tokenize = True,
#     add_generation_prompt = True, # Must add for generation
#     return_tensors = "pt",
# ).to("cuda")

# from transformers import TextStreamer
# text_streamer = TextStreamer(tokenizer, skip_prompt = True)
# _ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
#                    use_cache = True, temperature = 0.1, min_p = 0.1)

# %% [markdown]
# **Explanation:**
# 
# This cell showcases how to perform streaming inference, where the generated text is output incrementally as it is being produced. Here's the detailed breakdown:
# 
# 1. **Enabling Optimized Inference:**
#    - `FastLanguageModel.for_inference(model)`: Reiterates the activation of native optimizations for faster inference.
# 
# 2. **Preparing the Input Message:**
#    - Similar to the previous cell, defines a prompt to continue the Fibonacci sequence.
#    - Applies the chat template, tokenizes the input, adds generation prompts, and moves the inputs to the GPU.
# 
# 3. **Setting Up the Streamer:**
#    - `from transformers import TextStreamer`: Imports the `TextStreamer` class from Hugging Face's `transformers` library.
#    - `text_streamer = TextStreamer(tokenizer, skip_prompt = True)`: Initializes a `TextStreamer` instance.
#      - `tokenizer`: Passes the tokenizer for decoding tokens into text.
#      - `skip_prompt = True`: Configures the streamer to omit the initial prompt from the output, focusing only on the generated response.
# 
# 4. **Generating with Streaming:**
#    - `_ = model.generate(...)`: Initiates text generation with streaming.
#      - `input_ids = inputs`: Provides the tokenized input.
#      - `streamer = text_streamer`: Passes the streamer to handle incremental output.
#      - `max_new_tokens = 128`: Limits generation to 128 new tokens.
#      - `use_cache = True`, `temperature = 1.5`, `min_p = 0.1`: Similar parameters as the previous generation.
# 

# %% [markdown]
# ## Model Saving and Loading

# %%
from huggingface_hub import create_repo
import os

TOKEN = os.getenv('HUGGING_fACE_TOKEN')
try:
    repo_url = create_repo(repo_id="D33_mistral_7B_lora", private=True, token=TOKEN)
except:
    print('repo already exist')
try:
    repo_url = create_repo(repo_id="D33_mistral_7B", private=True, token=TOKEN)
except:
    print('repo already exist')
from transformers import AutoModelForCausalLM, AutoTokenizer

# name = "mistral-7B_0.3_chatml_dpo_mix-pt_and_guilherme34-pt_and_D33"

# model.save_pretrained("mistral-7B_0.3_chatml_D33_lora") # Local saving
# model.push_to_hub('BornSaint/D33_mistral_7B_lora', token=TOKEN)
model.push_to_hub_merged("BornSaint/D33_mistral_7B_lora", tokenizer, save_method="lora", token=TOKEN)
model.push_to_hub_merged("BornSaint/D33_mistral_7B", tokenizer, save_method="merged_16bit", token=TOKEN)
# tokenizer.save_pretrained("mistral-7B_0.3_chatml_D33_lora")
# tokenizer.push_to_hub('BornSaint/D33_mistral_7B_lora')

# model.save_pretrained_merged(name, tokenizer, save_method = "merged_16bit",)

# model = AutoModelForCausalLM.from_pretrained(name, device_map='cpu')
# tokenizer = AutoTokenizer.from_pretrained(name)

# model.push_to_hub('D33_mistral_7B', token=TOKEN)
# tokenizer.push_to_hub('D33_mistral_7B', token=TOKEN)


# %% [markdown]
# **Explanation:**
# 
# This cell saves the trained model and tokenizer for future use, either locally or by pushing to Hugging Face's Model Hub.
# 
# 1. **Saving Locally:**
#    - `model.save_pretrained("lora_model")`: Saves the fine-tuned model weights and configuration to a directory named `"lora_model"`.
#    - `tokenizer.save_pretrained("lora_model")`: Saves the tokenizer configuration to the same directory, ensuring that the model can be reloaded with the correct tokenizer.
# 
# 2. **Optional Online Saving (Commented Out):**
#    - `# model.push_to_hub("your_name/lora_model", token = "...")`: Uncommenting this line would push the model to Hugging Face's Model Hub under the specified repository name and authentication token.
#    - `# tokenizer.push_to_hub("your_name/lora_model", token = "...")`: Similarly, this would push the tokenizer to the same repository.
# 


# %%
# from unsloth import FastLanguageModel
# from transformers import TextStreamer
# # from fase1 import pulse_loop
# # from langchain.llms import HuggingFaceLLM
# if False:
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name="lora_model", # YOUR MODEL YOU USED FOR TRAINING
#         max_seq_length=max_seq_length,
#         dtype=dtype,
#         load_in_4bit=load_in_4bit,
#     )

# FastLanguageModel.for_inference(model)
# # llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)
# # pulse_loop(llm)
# messages = [
#     # {"role": "user", "content": "Describe a tall tower in the capital of France."},
#     {"role": "user", "content": ""},
# ]
# inputs = tokenizer.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True, # Must add for generation
#     return_tensors="pt",
# ).to("cuda")

# text_streamer = TextStreamer(tokenizer, skip_prompt = True)
# # _ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)

# %% [markdown]
# **Explanation:**
# 
# This cell demonstrates how to use the saved (fine-tuned) model to generate responses to new prompts.
# 
# 1. **Preparing the Input Message:**
#    - `messages = [{"role": "user", "content": "Describe a tall tower in the capital of France."},]`: Defines a new prompt asking the model to describe a tall tower in France's capital.
# 
# 2. **Applying the Chat Template:**
#    - `inputs = tokenizer.apply_chat_template(...)`: Formats the message using the chat template, tokenizes it, adds generation prompts, converts it to PyTorch tensors, and moves it to the GPU.
# 
# 3. **Setting Up the Streamer:**
#    - `from transformers import TextStreamer`: Imports the `TextStreamer` class.
#    - `text_streamer = TextStreamer(tokenizer, skip_prompt = True)`: Initializes a streamer to handle incremental output without displaying the initial prompt.
# 
# 4. **Generating the Output:**
#    - `_ = new_model.generate(...)`: Uses the `new_model` (assumed to be loaded in the next cell) to generate a response.
#      - `input_ids=inputs`: Provides the tokenized input.
#      - `streamer=text_streamer`: Enables streaming of the generated text.
#      - `max_new_tokens=128`, `use_cache=True`, `temperature=1.5`, `min_p=0.1`: Sets generation parameters for output quality and diversity.
# 

# %%
# from unsloth import FastLanguageModel
# from transformers import TextStreamer

# if False:    
#     new_model, new_tokenizer = FastLanguageModel.from_pretrained(
#         model_name="lora_model", # YOUR MODEL YOU USED FOR TRAINING
#         max_seq_length=max_seq_length,
#         dtype=dtype,
#         load_in_4bit=load_in_4bit,
#     )
#     FastLanguageModel.for_inference(new_model) # Enable native 2x faster inference
    
#     messages = [
#         {"role": "user", "content": "Describe a famous statue in New York City."},
#     ]
#     inputs = new_tokenizer.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True, # Must add for generation
#         return_tensors="pt",
#     ).to("cuda")
    
#     text_streamer = TextStreamer(new_tokenizer, skip_prompt = True)
#     _ = new_model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)

# %% [markdown]
# **Explanation:**
# 
# This cell demonstrates how to load the previously saved fine-tuned model and perform inference with a new prompt.
# 
# 1. **Importing FastLanguageModel:**
#    - `from unsloth import FastLanguageModel`: Imports the `FastLanguageModel` class from Unsloth.
# 
# 2. **Loading the Saved Model and Tokenizer:**
#    - `new_model, new_tokenizer = FastLanguageModel.from_pretrained(...)`: Loads the fine-tuned model and tokenizer from the local directory `"lora_model"`.
#      - `model_name = "lora_model"`: Specifies the directory where the trained model and tokenizer were saved.
#      - `#model_name = "unsloth/Qwen2.5-Coder-14B-Instruct",`: (Commented out) An alternative to load the original pre-trained model if needed.
#      - `max_seq_length = max_seq_length`, `dtype = dtype`, `load_in_4bit = load_in_4bit`: Passes the same configuration parameters used during initial model loading to ensure consistency.
# 
# 3. **Enabling Optimized Inference:**
#    - `FastLanguageModel.for_inference(new_model)`: Activates native inference optimizations for the loaded model, ensuring faster response generation.
# 
# 4. **Preparing a New Input Message:**
#    - `messages = [{"role": "user", "content": "Describe a famous statue in New York City."},]`: Defines a new prompt asking for a description of a famous statue in NYC.
#    - `inputs = new_tokenizer.apply_chat_template(...)`: Applies the chat template, tokenizes the message, adds generation prompts, converts to PyTorch tensors, and moves to the GPU.
# 
# 5. **Setting Up the Streamer and Generating Output:**
#    - `from transformers import TextStreamer`: Imports the `TextStreamer` class.
#    - `text_streamer = TextStreamer(new_tokenizer, skip_prompt = True)`: Initializes the streamer.
#    - `_ = new_model.generate(...)`: Generates the response using the loaded model.
#      - `input_ids=inputs`, `streamer=text_streamer`, `max_new_tokens=128`, `use_cache=True`, `temperature=1.5`, `min_p=0.1`: Specifies generation parameters for efficient and diverse output.
# 

# %% [markdown]
# ## Conclusion
# 
# The Jupyter notebook effectively illustrates how Unsloth's `FastLanguageModel` can revolutionize the fine-tuning and deployment of large language models. By implementing 4-bit quantization and PEFT with LoRA, the workflow achieves significant reductions in memory usage and training time, making it feasible to work with larger models even on hardware with limited resources. The integration with Hugging Face's ecosystem ensures compatibility and ease of use, allowing users to leverage a wide range of existing tools and datasets seamlessly. Furthermore, the optimized inference techniques demonstrated in the notebook enable faster and more efficient generation of responses, enhancing the practicality of deploying these models in real-time applications such as chatbots and conversational agents.
# 
# Overall, this approach not only accelerates the fine-tuning process but also ensures that the resulting models maintain high levels of accuracy and responsiveness. By focusing on selective training and leveraging advanced optimization strategies, Unsloth's `FastLanguageModel` provides a robust framework for developing and deploying state-of-the-art language models efficiently and effectively.
# 


