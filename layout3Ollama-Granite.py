# Chat with an intelligent assistant in your terminal  
# MODEL: ollama-granite3dense
# this wil run granite3-2B-instruct through ollamaAPI
# sources: https://github.com/fabiomatricardi/-LLM-Studies/raw/main/00.consoleAPI_stream.py
# https://github.com/fabiomatricardi/-LLM-Studies/blob/main/01.st-API-openAI_stream.py
# OLLAMA MODEL CARD: https://ollama.com/library/granite3-dense/blobs/604785e698e9
# OPenAI API for Ollama: https://github.com/ollama/ollama/blob/main/docs/openai.md
# https://github.com/ibm-granite/granite-3.0-language-models
# https://www.ibm.com/granite/docs/
# HUGGINFACE: https://huggingface.co/ibm-granite/granite-3.0-2b-instruct
#####################################################################################################

"""
> ollama show granite3-dense
  Model
    architecture        granite
    parameters          2.6B
    context length      4096
    embedding length    2048
    quantization        Q4_K_M

  License
    Apache License
    Version 2.0, January 2004
"""
import gradio as gr
import datetime
from promptLibv2Qwen import countTokens, writehistory, createCatalog
from promptLibv2Qwen import genRANstring, createStats
from gradio import ChatMessage
from openai import OpenAI

## PREPARING FINAL DATASET

pd_id = []
pd_task = []
pd_vote = []
pd_remarks = []
test_progress = 0
history = []
tasks = createCatalog()
modelname = 'granite3-dense-2b'
stops = ['<|end_of_text|>']
#load client with OpenAI API toward Ollama Endpoint
client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')
print(f"2. Model {modelname} loaded with OLLAMA...")
# fizing issue on dipsplaying avatars 
# https://www.gradio.app/guides/custom-CSS-and-JS
# https://github.com/gradio-app/gradio/issues/9702
custom_css = """
           
            .message-row img {
                margin: 0px !important;
            }

            .avatar-container img {
            padding: 0px !important;
}
        """

def generate_response(history):
    history.append(
        ChatMessage(role="user",
                    content="Hi, my name is Fabio, a Medium writer. Who are you?")
        )
    history.append(
        ChatMessage(role="assistant",
                    content="Hi, I am your local GPT. How can I help you?")
        )
    return history

history = generate_response(history)
with gr.Blocks(theme=gr.themes.Glass(), css=custom_css) as demo:
    #TITLE SECTION
    with gr.Row(variant='compact'):
            with gr.Column(scale=1):            
                gr.Image(value='img/granite3.png', 
                        show_label = False, 
                        show_download_button = False, container = False)              
            with gr.Column(scale=4):
                gr.HTML("<center>"
                + "<h1>Revised Benchmark with You as a Feedback!</h1>"
                + "<h4>💎 granite3-dense-2B - 4K context window with Ollama engine</h4></center>")  
                gr.Markdown("""*Run a prompt catalogue with 11 tasks*
                            to validate the performances of a Small Langage Model<br>
                            At the end of every generation the process will wait for the Feedback by the user<br>
                            ### Fixed tuning Parameters:
                            ```
                            temperature = 0.25
                            repeat_penalty = 1.178
                            max_new_tokens = 900

                            ```
                            """)
    with gr.Row(variant='compact'): # Progress status
            with gr.Column(scale=1):
                btn_test = gr.Button(value='Start AutoTest', variant='huggingface')
                act_task = gr.Text('', placeholder="running task..",show_label=False)
            with gr.Column(scale=4):          
                actual_progress = gr.Slider(0, len(tasks), 
                                            value=test_progress, label="Prompt Catalogue Progress", 
                                            #info="Run the most used NLP tasks with a Language Model",
                                            interactive=False)    
    with gr.Row(variant='compact'): # KpI
        # with gr.Column():
              txt_ttft = gr.Text('', placeholder="seconds..",
                                   label='Time to first token')
        # with gr.Column():
              txt_gentime = gr.Text('', placeholder="TimeDelta..",
                                   label='Generation Time')
        # with gr.Column():
              txt_speed = gr.Text('', placeholder="t/s..",
                                   label='Generation Speed')
        # with gr.Column():
              txt_TOTtkns = gr.Text('', placeholder="tokens..",
                                   label='Total num of Tokens')

    with gr.Row(variant='compact'): # ChatBot Area
                myBOT =gr.Chatbot(history,type='messages',avatar_images=("./img/user.jpg","./img/bot.jpg"))      #
    
    with gr.Row(variant='compact'): #Temporary Area            
                temp_input = gr.Text('what is Artificial Intelligence?', 
                                   label='ACTUAL TASK PROMPT',lines=2)                

    with gr.Row(variant='compact'): # Feedback from the user
            with gr.Column(scale=1):
                gr.Markdown("""#### Respect this format:
                            
                            Put a number from 0 to 5, a space, and then your comments<br>
                            ```
                            5 very good one
                            ```
                            """)
                
            with gr.Column(scale=4):          
                txt_fbck = gr.Text('', placeholder="Your evaluation feedback..",
                                   label='User Feedback',lines=2)          
                btn_fbck = gr.Button(value='submit feedback', variant='huggingface')
    
    def update_history(history,a):
        history.append(
            ChatMessage(role="user",
                        content=a)
            )
        return history    
    
    def startInference(a):
        print(a[-1]['content'])
        prompt = [
			{"role": "user", "content": a[-1]['content']}
		    ]
        promptTKNS = countTokens(a[-1]['content'])
        a.append({"role":"assistant","content":''})
        generation = ''
        fisrtround=0
        start = datetime.datetime.now()
        completion = client.chat.completions.create(
            messages=prompt,
            model='granite3-dense',
            temperature=0.25,
            frequency_penalty  = 1.178,
            stop=stops,
            max_tokens=900,
            stream=True            
        )
        for chunk in completion:
            try:
                if chunk.choices[0].delta.content:
                    if fisrtround==0:
                        a[-1]['content'] += chunk.choices[0].delta.content
                        ttftoken = datetime.datetime.now() - start 
                        secondsTTFT =  ttftoken.total_seconds()
                        ttFT = f"TimeToFristToken: {secondsTTFT:.2f} sec"
                        fisrtround = 1
                    else:
                        a[-1]['content'] += chunk.choices[0].delta.content                              
            except:
                pass  
            answrTKN = countTokens(a[-1]['content'])
            totTKN = promptTKNS + answrTKN
            total_tokens = f"Total Tkns: {totTKN}"
            delta = datetime.datetime.now() - start
            seconds = delta.total_seconds()
            speed = totTKN/seconds
            speed_tokens = f"Gen Speed: {speed:.2f} t/s"
            yield a, delta, speed_tokens, ttFT,total_tokens         

    btn_test.click(update_history, inputs=[myBOT,temp_input], 
              outputs=[myBOT]).then(startInference,[myBOT],[myBOT,txt_gentime,txt_speed,txt_ttft,txt_TOTtkns])

    def startloop():
         #rasie flag to wait
         #start the loop
         #display task
         #append chatbot prompt
         #start the generation
         pass


if __name__ == "__main__":
    demo.launch(inbrowser=True)