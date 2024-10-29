import gradio as gr
import datetime
from promptLibv2Qwen import countTokens, writehistory, createCatalog
from promptLibv2Qwen import genRANstring, createStats
from gradio import ChatMessage


## PREPARING FINAL DATASET

pd_id = []
pd_task = []
pd_vote = []
pd_remarks = []
test_progress = 0
history = []
tasks = createCatalog()
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
                gr.Image(value='img/qwen.png', 
                        show_label = False, 
                        show_download_button = False, container = False)              
            with gr.Column(scale=4):
                gr.HTML("<center>"
                + "<h1>Revised Benchmark with You as a Feedback!</h1>"
                + "<h4>💎 Qwen2.5-0.5B-it - 8K context window</h4></center>")  
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
    with gr.Row(variant='compact'): # ChatBot Area
                gr.Chatbot(history,type='messages',avatar_images=("./img/user.jpg","./img/bot.jpg"))      #
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

         

if __name__ == "__main__":
    demo.launch(inbrowser=True)