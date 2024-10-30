# GradioRBYF
Gradio Interface for automatic Revised Benchmark with You as a Feedback



# About Gradio
Issues on avatars in the chatbot
- [https://www.gradio.app/guides/custom-CSS-and-JS](https://www.gradio.app/guides/custom-CSS-and-JS)
- [https://github.com/gradio-app/gradio/issues/9702](https://github.com/gradio-app/gradio/issues/9702)
- [Gradio Theming](https://www.gradio.app/guides/theming-guide)
```
gr.themes.Base() - the "base" theme sets the primary color to blue but otherwise has minimal styling, making it particularly useful as a base for creating new, custom themes.
gr.themes.Default() - the "default" Gradio 5 theme, with a vibrant orange primary color and gray secondary color.
gr.themes.Origin() - the "origin" theme is most similar to Gradio 4 styling. Colors, especially in light mode, are more subdued than the Gradio 5 default theme.
gr.themes.Citrus() - the "citrus" theme uses a yellow primary color, highlights form elements that are in focus, and includes fun 3D effects when buttons are clicked.
gr.themes.Monochrome() - the "monochrome" theme uses a black primary and white secondary color, and uses serif-style fonts, giving the appearance of a black-and-white newspaper.
gr.themes.Soft() - the "soft" theme uses a purpose primary color and white secondary color. It also increases the border radii and around buttons and form elements and highlights labels.
gr.themes.Glass() - the "glass" theme has a blue primary color and a transclucent gray secondary color. The theme also uses vertical gradients to create a glassy effect.
gr.themes.Ocean() - the "ocean" theme has a blue-green primary color and gray secondary color. The theme also uses horizontal gradients, especially for buttons and some form elements.
```
- [Gradio Theme Gallery](https://huggingface.co/spaces/gradio/theme-gallery)
- [GUIDE How to Create a Chatbot with Gradio](https://www.gradio.app/guides/creating-a-chatbot-fast)

### Newspaper3k: Article scraping & curation
- [https://github.com/codelucas/newspaper](https://github.com/codelucas/newspaper)
- [https://github.com/fabiomatricardi/MetadataIsAllYouNeed/blob/main/KeyBERT_gr.py](https://github.com/fabiomatricardi/MetadataIsAllYouNeed/blob/main/KeyBERT_gr.py)


### General inference rules
The Mistral models allows you to chat with a model that has been fine-tuned to follow instructions and respond to natural language prompts. A prompt is the input that you provide to the Mistral model. It can come in various forms, such as asking a question, giving an instruction, or providing a few examples of the task you want the model to perform. Based on the prompt, the Mistral model generates a text output as a response.
The chat completion API accepts a list of chat messages as input and generates a response. This response is in the form of a new chat message with the role "assistant" as output.
- [https://docs.mistral.ai/capabilities/completion/](https://docs.mistral.ai/capabilities/completion/)
- 

# Models to try
### H2OVL-Mississippi-800M
The H2OVL-Mississippi-800M is a compact yet powerful vision-language model from H2O.ai, featuring 0.8 billion parameters. Despite its small size, it delivers state-of-the-art performance in text recognition, excelling in the Text Recognition segment of OCRBench and outperforming much larger models in this domain. Built upon the robust architecture of our H2O-Danube language models, the Mississippi-800M extends their capabilities by seamlessly integrating vision and language tasks.

<img src='https://huggingface.co/h2oai/h2ovl-mississippi-800m/resolve/main/assets/text_recognition.png' width=500>
- [HuggingFace repo at H2O](https://huggingface.co/h2oai/h2ovl-mississippi-800m)
- [H2O article by Asghar Ghorbani](https://h2o.ai/blog/2024/document-classification-with-h2o-vl-mississippi--a-quick-guide/)
### MaziyarPanahi's Collections
🚀GGUF Llama.cpp compatible models, can be used on CPUs and GPUs!<br>
Here the [AMAZING COLLECTION](https://huggingface.co/collections/MaziyarPanahi/gguf-65afc99c3997c4b6d2d9e1d5)
### h2oai/deberta_finetuned_pii
A finetuned model designed to recognize and classify Personally Identifiable Information (PII) within unstructured text data. This powerful model accurately identifies a wide range of PII categories, such as account names, credit card numbers, emails, phone numbers, and addresses. The model is specifically trained to detect various PII types, including but not limited to:

```
| Category               | Data                                                                                   |
|------------------------|----------------------------------------------------------------------------------------|
| Account-related information | Account name, account number, and transaction amounts                             |
| Banking details        | BIC, IBAN, and Bitcoin or Ethereum addresses                                           |
| Personal information   | Full name, first name, middle name, last name, gender, and date of birth               |
| Contact information    | Email, phone number, and street address (including building number, city, county, state, and zip code) |
| Job-related data       | Job title, job area, job descriptor, and job type                                      |
| Financial data         | Credit card number, issuer, CVV, and currency information (code, name, and symbol)     |
| Digital identifiers    | IP addresses (IPv4 and IPv6), MAC addresses, and user agents                           |
| Online presence        | URL, usernames, and passwords                                                          |
| Other sensitive data   | SSN, vehicle VIN and VRM, phone IMEI, and nearby GPS coordinates                       |

```

The PII Identifier Model ensures data privacy and compliance by effectively detecting and categorizing sensitive information within documents, emails, user-generated content, and more. Make your data processing safer and more secure with our state-of-the-art PII detection technology.
- [Hugging Face Repo](https://huggingface.co/h2oai/deberta_finetuned_pii)
- 



# Large Language Models course free
- [llm-engineering-handbook](https://github.com/aofoegbu/llm-engineers-handbook)
- [DeepLearning.AI course on Agents](https://learn.deeplearning.ai/courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/lesson/1/introduction)
- [Microsoft BitNet.cpp](https://github.com/microsoft/BitNet)
- [ArXiv paper *1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs*](https://arxiv.org/abs/2410.16144v1)
- [MotleyCrew AI](https://motleycrew.ai/)
- [MotleyCrewAI-readthedocs](https://motleycrew.readthedocs.io/en/latest/quickstart.html)
- [OpenVino quick guide CheatSheet](https://docs.openvino.ai/2024/_static/download/OpenVINO_Quick_Start_Guide.pdf)
- [OpenVino Toolkit Getting Started](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/get-started.html)

# About IBM/Granite  models
- [Ollama serving](https://ollama.com/library/granite3-moe:1b)
- [Granite3 IbM on GitHub](https://github.com/ibm-granite/granite-3.0-language-models)
-  [Fabio you are the Benchmark](https://github.com/fabiomatricardi/YouAreTheBenchmark)
-  


## Markdown resoources
- [MArkdown Videos](https://github.com/Snailedlt/Markdown-Videos)
- 




## [Universal Assisted Generation: Faster Decoding with Any Assistant Model](https://huggingface.co/blog/universal_assisted_generation)
TL;DR: Many LLMs such as gemma-2-9b and Mixtral-8x22B-Instruct-v0.1 lack a much smaller version to use for assisted generation. In this blog post, we present Universal Assisted Generation: a method developed by Intel Labs and Hugging Face which extends assisted generation to work with a small language model from any model family 🤯. As a result, it is now possible to accelerate inference from any decoder or Mixture of Experts model by 1.5x-2.0x at almost zero-cost 🔥🔥🔥!
In order to mitigate this pain point, Intel Labs, together with our friends at Hugging Face, has developed Universal Assisted Generation (UAG). UAG enables selecting any pair of target and assistant models regardless of their tokenizer. For example, gemma-2-9b can be used as the target model, with the tiny vicuna-68m as the assistant.
The main idea behind the method we propose is 2-way tokenizer translations. Once the assistant model completes a generation iteration, the assistant tokens are converted to text, which is then tokenized using the target model's tokenizer to generate target tokens. After the verification step, the target tokens are similarly converted back to the assistant tokens format, which are then appended to the assistant model's context before the next iteration begins.
Since the assistant and target tokenizers use different vocabularies it's necessary to handle the discrepancies between them. To accurately re-encode the newly generated assistant tokens, it’s essential to prepend a context window consisting of several previous tokens. This entire sequence is then re-encoded into the target token format and aligned with the most recent target tokens to pinpoint the exact location where the newly generated tokens should be appended. This process is illustrated in the video below.




https://github.com/user-attachments/assets/5d10e09f-b5c5-40a7-8414-1dd0465593b4



```
[![Watch the video](https://github.com/fabiomatricardi/GradioRBYF/blob/main/img/videoframe_11799.png)](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/universal-assisted-generation/method-animation.mov)
```

```
<video width="320" height="240" controls>
  <source src="[video.mov](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/universal-assisted-generation/method-animation.mov)" type="video/mp4">
</video>
```
References for videos in github:
- [https://stackoverflow.com/questions/4279611/how-to-embed-a-video-into-github-readme-md](https://stackoverflow.com/questions/4279611/how-to-embed-a-video-into-github-readme-md)
- [https://stackoverflow.com/questions/4279611/how-to-embed-a-video-into-github-readme-md/4279746#4279746](https://stackoverflow.com/questions/4279611/how-to-embed-a-video-into-github-readme-md/4279746#4279746)
- [https://github.com/alelievr/Mixture/blob/0.4.0/README.md](https://github.com/alelievr/Mixture/blob/0.4.0/README.md)
- [https://stackoverflow.com/questions/4279611/how-to-embed-a-video-into-github-readme-md](https://stackoverflow.com/questions/4279611/how-to-embed-a-video-into-github-readme-md)
- [https://www.geeksforgeeks.org/how-to-add-videos-on-readme-md-file-in-a-github-repository/](https://www.geeksforgeeks.org/how-to-add-videos-on-readme-md-file-in-a-github-repository/)


