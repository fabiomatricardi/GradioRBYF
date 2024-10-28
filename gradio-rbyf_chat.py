""""
GRADIO INTERFACE
with llama.cpp core engine
using Llama.generate() default method, with no chat_template
This playground does not show history messages, and it is not
meant to be a chat-bot
But all generations are logged in text file
"""

import gradio as gr
from llama_cpp import Llama
import datetime

#MODEL SETTINGS also for DISPLAY
convHistory = ''
modelfile = "models/qwen2.5-0.5b-instruct-q8_0.gguf"
root='qwen2.5-0.5b-instruct'
repetitionpenalty = 1.15
contextlength=8196
logfile = f'{root}_GR_logs.txt'
print("loading model...")
stt = datetime.datetime.now()
llm = Llama(
  model_path=modelfile,  # Download the model file first
  n_ctx=contextlength,  # The max sequence length to use - note that longer sequence lengths require much more resources
  #n_threads=2,            # The number of CPU threads to use, tailor to your system and the resulting performance
)
dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")

def writehistory(text):
    with open(logfile, 'a') as f:
        f.write(text)
        f.write('\n')
    f.close()

def combine(a, b, c, d,e,f):
    global convHistory
    import datetime
    SYSTEM_PROMPT = f"""{a}


    """ 
    temperature = c
    max_new_tokens = d
    repeat_penalty = f
    top_p = e
    prompt = [
			{"role": "user", "content": b}
		]
    start = datetime.datetime.now()
    generation = ""
    delta = ""
    promptTKN = len(llm.tokenize(bytes(b,encoding='utf-8')))
    prompt_tokens = f"Prompt Tokens: {promptTKN}"
    generated_text = ""
    answer_tokens = ''
    total_tokens = ''   
    fisrtround=0
    for chunk in llm.create_chat_completion(messages=prompt, 
                max_tokens=max_new_tokens, 
                stop=['<|im_end|>'],
                temperature = temperature,
                repeat_penalty = repeat_penalty,
                top_p = top_p,   # Example stop token - not necessarily correct for this specific model! Please check before using.
                stream=True):
        try:
            if chunk["choices"][0]["delta"]["content"]:
                if fisrtround==0:
                    print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                    generation += chunk["choices"][0]["delta"]["content"]
                    ttftoken = datetime.datetime.now() - start 
                    secondsTTFT =  ttftoken.total_seconds()
                    total_tokens = f"TimeToFristToken: {secondsTTFT:.2f} sec"
                    fisrtround = 1
                else:
                    print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                    generation += chunk["choices"][0]["delta"]["content"]                              
        except:
            pass  
        answrTKN = len(llm.tokenize(bytes(generation,encoding='utf-8')))
        answer_tokens = f"Out Tkns: {answrTKN}"
        totTKN = promptTKN + answrTKN
        #total_tokens = f"Total Tkns: {totTKN}"
        delta = datetime.datetime.now() - start
        seconds = delta.total_seconds()
        speed = totTKN/seconds
        speed_tokens = f"Gen Speed: {speed:.2f} t/s"
        yield generation, delta, prompt_tokens, answer_tokens, total_tokens, speed_tokens
    timestamp = datetime.datetime.now()
    logger = f"""____________________________________________________
time: {timestamp}
Temp: {temperature} - MaxNewTokens: {max_new_tokens} 
RepPenalty: {repeat_penalty} 
____________________________________________________
PROMPT: \n{prompt}
{root}: {generation}
---
Generated in {delta}
{prompt_tokens}
{answer_tokens}
{total_tokens}
Total Tokens: {totTKN}
Generation Speed: {speed:.2f} t/s
---"""
    writehistory(logger)
    convHistory = convHistory + b + "\n" + generation + "\n"
    print(convHistory)
    return generation, delta, prompt_tokens, answer_tokens, total_tokens    

# MAIN GRADIO INTERFACE
with gr.Blocks(theme=gr.themes.Glass()) as demo:   #theme= 'Medguy/base2' #theme='remilia/Ghostly'
    #TITLE SECTION
    with gr.Row(variant='compact'):
            with gr.Column(scale=1):            
                gr.Image(value='img/qwen.png', 
                        show_label = False, 
                        show_download_button = False, container = False)              
            with gr.Column(scale=4):
                gr.HTML("<center>"
                + "<h3>Revised Benchmark with You as a Feedback!</h3>"
                + "<h1>💎 Qwen2.5-0.5B-it - 8K context window</h2></center>")  
                with gr.Row():
                        with gr.Column(min_width=80):
                            gentime = gr.Textbox(value="", placeholder="Generation Time:", min_width=50, show_label=False)                          
                        with gr.Column(min_width=80):
                            prompttokens = gr.Textbox(value="", placeholder="Prompt Tkn:", min_width=50, show_label=False)
                        with gr.Column(min_width=80):
                            outputokens = gr.Textbox(value="", placeholder="Output Tkn:", min_width=50, show_label=False)            
                        with gr.Column(min_width=80):
                            totaltokens = gr.Textbox(value="", placeholder="Total Tokens:", min_width=50, show_label=False)   
                        with gr.Column(min_width=80):
                            genspeed = gr.Textbox(value="", placeholder="Generation Speed:", min_width=50, show_label=False)  

    # PLAYGROUND INTERFACE SECTION
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            ### Tunning Parameters""")
            temp = gr.Slider(label="Temperature",minimum=0.0, maximum=1.0, step=0.01, value=0.42)
            top_p = gr.Slider(label="Top_P",minimum=0.0, maximum=1.0, step=0.01, value=0.8)
            repPen = gr.Slider(label="Repetition Penalty",minimum=0.0, maximum=4.0, step=0.01, value=1.2)
            max_len = gr.Slider(label="Maximum output lenght", minimum=10,maximum=(contextlength-500),step=2, value=900)
            gr.Markdown(
            """
            Fill the System Prompt and User Prompt
            And then click the Button below
            """)
            btn = gr.Button(value="💎🦜 Generate", variant='primary')
            gr.Markdown(
            f"""
            - **Prompt Template**: Qwen
            - **LLM Engine**: llama-cpp
            - **Log File**: {logfile}
            """) 

        with gr.Column(scale=4):
            txt = gr.Textbox(label="System Prompt", value = "", placeholder = "This models does not have any System prompt...",lines=1, interactive = False)
            txt_2 = gr.Textbox(label="User Prompt", lines=5, show_copy_button=True)
            txt_3 = gr.Textbox(value="", label="Output", lines = 10, show_copy_button=True)
            eval = gr.Textbox(label="RBYF - Your feedback", lines=2, 
                              placeholder = "smething like '4 there are some issues'...",
                              show_copy_button=True)
            btn2 = gr.Button(value="Save Comment", variant='secondary')
            def savelogs(a):
                logging = f"YOUR FEEDBACK:\n{a}\n____________________________________________________\n\n\n"
                writehistory(logging)
                print(logging)
            btn.click(combine, inputs=[txt, txt_2,temp,max_len,top_p,repPen], outputs=[txt_3,gentime,prompttokens,outputokens,totaltokens, genspeed])
            btn2.click(savelogs,inputs=[eval], outputs=[])

if __name__ == "__main__":
    demo.launch(inbrowser=True)


"""
MODEL CARD
______________________________________________________________________________________________________
Python 3.12.6 (tags/v3.12.6:a4a2d2b, Sep  6 2024, 20:11:23) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> mp = 'models/qwen2.5-0.5b-instruct-q8_0.gguf'
>>> from llama_cpp import Llama
>>> llm = Llama(model_path=mp)
llama_model_loader: loaded meta data with 26 key-value pairs and 291 tensors 
from models/qwen2.5-0.5b-instruct-q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
           general.architecture str              = qwen2
                   general.type str              = model
                   general.name str              = qwen2.5-0.5b-instruct
                general.version str              = v0.1
               general.finetune str              = qwen2.5-0.5b-instruct
             general.size_label str              = 630M
              qwen2.block_count u32              = 24
           qwen2.context_length u32              = 32768
         qwen2.embedding_length u32              = 896
      qwen2.feed_forward_length u32              = 4864
     qwen2.attention.head_count u32              = 14
  qwen2.attention.head_count_kv u32              = 2
           qwen2.rope.freq_base f32              = 1000000.000000
ttention.layer_norm_rms_epsilon f32              = 0.000001
              general.file_type u32              = 7
           tokenizer.ggml.model str              = gpt2
             tokenizer.ggml.pre str              = qwen2
          tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
      tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
          tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
    tokenizer.ggml.eos_token_id u32              = 151645
tokenizer.ggml.padding_token_id u32              = 151643
    tokenizer.ggml.bos_token_id u32              = 151643
   tokenizer.ggml.add_bos_token bool             = false
        tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
   general.quantization_version u32              = 2
llama_model_loader: - type  f32:  121 tensors
llama_model_loader: - type q8_0:  170 tensors
llm_load_vocab: special tokens cache size = 22
llm_load_vocab: token to piece cache size = 0.9310 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 896
llm_load_print_meta: n_layer          = 24
llm_load_print_meta: n_head           = 14
llm_load_print_meta: n_head_kv        = 2
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 64
llm_load_print_meta: n_embd_head_v    = 64
llm_load_print_meta: n_gqa            = 7
llm_load_print_meta: n_embd_k_gqa     = 128
llm_load_print_meta: n_embd_v_gqa     = 128
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 4864
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 32768
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: ssm_dt_b_c_rms   = 0
llm_load_print_meta: model type       = 1B
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 630.17 M
llm_load_print_meta: model size       = 638.74 MiB (8.50 BPW)
llm_load_print_meta: general.name     = qwen2.5-0.5b-instruct
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: EOG token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOG token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
ggml_vulkan: Found 1 Vulkan devices:
Vulkan0: Intel(R) UHD Graphics (Intel Corporation) | uma: 1 | fp16: 1 | warp size: 32
llm_load_tensors: ggml ctx size =    0.13 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/25 layers to GPU
llm_load_tensors:        CPU buffer size =   638.74 MiB
...........................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init: Vulkan_Host KV buffer size =     6.00 MiB
llama_new_context_with_model: KV self size  =    6.00 MiB, K (f16):    3.00 MiB, V (f16):    3.00 MiB
llama_new_context_with_model: Vulkan_Host  output buffer size =     0.58 MiB
llama_new_context_with_model: Intel(R) UHD Graphics compute buffer size =   436.44 MiB
llama_new_context_with_model: Vulkan_Host compute buffer size =     2.76 MiB
llama_new_context_with_model: graph nodes  = 846
llama_new_context_with_model: graph splits = 340
AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | 
AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | 
RISCV_VECT = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
Model metadata: {'general.name': 'qwen2.5-0.5b-instruct', 'general.architecture': 'qwen2', 
'general.type': 'model', 'general.finetune': 'qwen2.5-0.5b-instruct', 'general.version': 'v0.1', 
'qwen2.block_count': '24', 'general.size_label': '630M', 'qwen2.context_length': '32768', 
'qwen2.embedding_length': '896', 'general.quantization_version': '2', 
'tokenizer.ggml.bos_token_id': '151643', 'qwen2.feed_forward_length': '4864', 'qwen2.attention.head_count': '14', 
'qwen2.attention.head_count_kv': '2', 'tokenizer.ggml.padding_token_id': '151643', 
'qwen2.rope.freq_base': '1000000.000000', 'qwen2.attention.layer_norm_rms_epsilon': '0.000001', 
'tokenizer.ggml.eos_token_id': '151645', 'general.file_type': '7', 'tokenizer.ggml.model': 'gpt2', 
'tokenizer.ggml.pre': 'qwen2', 'tokenizer.ggml.add_bos_token': 'false', 
'tokenizer.chat_template': '{%- if tools %}\n    {{- \'<|im_start|>system\\n\' }}\n    {%- if messages[0][\'role\'] == \'system\' %}\n        {{- messages[0][\'content\'] }}\n    {%- else %}\n        {{- \'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\' }}\n    {%- endif %}\n    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n    {%- for tool in tools %}\n        {{- "\\n" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}\n    {%- if messages[0][\'role\'] == \'system\' %}\n        {{- \'<|im_start|>system\\n\' + messages[0][\'content\'] + \'<|im_end|>\\n\' }}\n    {%- else %}\n        {{- \'<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n\' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}\n        {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content + \'<|im_end|>\' + \'\\n\' }}\n    {%- elif message.role == "assistant" %}\n        {{- \'<|im_start|>\' + message.role }}\n        {%- if message.content %}\n            {{- \'\\n\' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- \'\\n<tool_call>\\n{"name": "\' }}\n            {{- tool_call.name }}\n            {{- \'", "arguments": \' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \'}\\n</tool_call>\' }}\n        {%- endfor %}\n        {{- \'<|im_end|>\\n\' }}\n    {%- elif message.role == "tool" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}\n            {{- \'<|im_start|>user\' }}\n        {%- endif %}\n        {{- \'\\n<tool_response>\\n\' }}\n        {{- message.content }}\n        {{- \'\\n</tool_response>\' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n            {{- \'<|im_end|>\\n\' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|im_start|>assistant\\n\' }}\n{%- endif %}\n'}
Available chat formats from metadata: chat_template.default
Using gguf chat template: {%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}

Using chat eos_token: <|im_end|>
Using chat bos_token: <|endoftext|>
"""