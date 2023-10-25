# MixtureOfMedicalExperts
MoE Mode with Mixture of Experts to Support Health Care Scenarios in Multi Agent Systems

# Program List:
1. https://huggingface.co/spaces/awacke1/California-Medical-Centers-Streamlit  ![image](https://github.com/AaronCWacker/MixtureOfMedicalExperts/assets/30595158/1459bc70-2d8a-417b-ad29-9285d5af07f9)
2. https://huggingface.co/spaces/awacke1/Google-Maps-Web-Service-Py   ![image](https://github.com/AaronCWacker/MixtureOfMedicalExperts/assets/30595158/2f2f5582-f9c6-47d4-b7e2-56842ef59862)
3. https://huggingface.co/spaces/awacke1/Prompt-Refinery-Text-to-Image-Generation   ![image](https://github.com/AaronCWacker/MixtureOfMedicalExperts/assets/30595158/2ab762b0-7ad2-4608-8127-d92e5838d5f1)
4. https://huggingface.co/spaces/awacke1/Azure-Cosmos-DB    ![image](https://github.com/AaronCWacker/MixtureOfMedicalExperts/assets/30595158/a22df687-ccbb-4e90-b0df-ec89a06ed145)
5. https://huggingface.co/spaces/awacke1/Top-Ten-United-States    ![image](https://github.com/AaronCWacker/MixtureOfMedicalExperts/assets/30595158/7a11636c-47a4-454b-9d37-726958b461cb)
6. https://huggingface.co/spaces/awacke1/MN.Map.Hospitals.Top.Five    ![image](https://github.com/AaronCWacker/MixtureOfMedicalExperts/assets/30595158/bd620164-6c38-4e7f-bac6-00d3dea0d09b)
7. https://huggingface.co/spaces/awacke1/MusicGenStreamFacebook    ![image](https://github.com/AaronCWacker/MixtureOfMedicalExperts/assets/30595158/a61f91ac-a1aa-474f-af60-221af31b37ca)
8. https://huggingface.co/spaces/awacke1/MixtureOfMedicalExperts    ![image](https://github.com/AaronCWacker/MixtureOfMedicalExperts/assets/30595158/ac1b1edb-bdc1-439a-a024-06d43ead215e)
9. https://huggingface.co/spaces/awacke1/MistralCoder         ![image](https://github.com/AaronCWacker/MixtureOfMedicalExperts/assets/30595158/09c4a8b4-2c6d-4f2a-960c-d4bde4c2b669)









# Program - Llama


# 3. Stream Llama Response
# @st.cache_resource
def StreamLLMChatResponse(prompt):
    try:
        endpoint_url = API_URL
        hf_token = API_KEY
        client = InferenceClient(endpoint_url, token=hf_token)
        gen_kwargs = dict(
            max_new_tokens=512,
            top_k=30,
            top_p=0.9,
            temperature=0.2,
            repetition_penalty=1.02,
            stop_sequences=["\nUser:", "<|endoftext|>", "</s>"],
        )
        stream = client.text_generation(prompt, stream=True, details=True, **gen_kwargs)
        report=[]
        res_box = st.empty()
        collected_chunks=[]
        collected_messages=[]
        allresults=''
        for r in stream:
            if r.token.special:
                continue
            if r.token.text in gen_kwargs["stop_sequences"]:
                break
            collected_chunks.append(r.token.text)
            chunk_message = r.token.text
            collected_messages.append(chunk_message)
            try:
                report.append(r.token.text)
                if len(r.token.text) > 0:
                    result="".join(report).strip()
                    res_box.markdown(f'*{result}*')
                    
            except:
                st.write('Stream llm issue')
        SpeechSynthesis(result)
        return result
    except:
        st.write('Llama model is asleep. Starting up now on A10 - please give 5 minutes then retry as KEDA scales up from zero to activate running container(s).')


# P -> NP - Difficulty of program genesis by crossbreeding programs:

Combine these two programs to produce one program that iterates through each expert with a prompt and executes the program flow for program two which uses Mistral model to run a system prompt plus context and agent prompt through LLM and produces speech output.  Program 1:  import streamlit as st

def triage_checkin():
    st.write("### Triage and Check-in Expert 🚑")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Triage")

def lab_analyst():
    st.write("### Lab Analyst 🧪")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Lab Analysis")

def medicine_specialist():
    st.write("### Medicine Specialist 💊")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Medicine")

def service_expert():
    st.write("### Service Expert 💲")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Service")

def care_expert():
    st.write("### Level of Care Expert 🏥")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Level of Care")

def terminology_expert():
    st.write("### Terminology Expert 📚")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Terminology")

def cmo():
    st.write("### Chief Medical Officer 🩺")
    for i in range(1, 11):
        st.text_input(f"Question {i} for CMO")

def medical_director():
    st.write("### Medical Director Team 🏢")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Medical Director")

def main():
    st.title("Mixture of Medical Experts Model")
    st.write("Harness the power of AI with this specialized healthcare framework! 🎉")

    role = st.selectbox("Select AI Role:", [
        "Triage and Check-in Expert",
        "Lab Analyst",
        "Medicine Specialist",
        "Service Expert",
        "Level of Care Expert",
        "Terminology Expert",
        "Chief Medical Officer",
        "Medical Director Team"
    ])

    if role == "Triage and Check-in Expert":
        triage_checkin()
    elif role == "Lab Analyst":
        lab_analyst()
    elif role == "Medicine Specialist":
        medicine_specialist()
    elif role == "Service Expert":
        service_expert()
    elif role == "Level of Care Expert":
        care_expert()
    elif role == "Terminology Expert":
        terminology_expert()
    elif role == "Chief Medical Officer":
        cmo()
    elif role == "Medical Director Team":
        medical_director()

if __name__ == "__main__":
    main()
  Program 2:  from huggingface_hub import InferenceClient
import gradio as gr

client = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.1"
)


def format_prompt(message, history):
  prompt = "<s>"
  for user_prompt, bot_response in history:
    prompt += f"[INST] {user_prompt} [/INST]"
    prompt += f" {bot_response}</s> "
  prompt += f"[INST] {message} [/INST]"
  return prompt

def generate(
    prompt, history, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt, history)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    return output


additional_inputs=[
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=256,
        minimum=0,
        maximum=1048,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.90,
        minimum=0.0,
        maximum=1,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
    gr.Slider(
        label="Repetition penalty",
        value=1.2,
        minimum=1.0,
        maximum=2.0,
        step=0.05,
        interactive=True,
        info="Penalize repeated tokens",
    )
]

css = """
  #mkd {
    height: 200px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""

with gr.Blocks(css=css) as demo:
  
    gr.ChatInterface(
        generate,
        additional_inputs=additional_inputs,       
        examples = [
            ["🐍 Write a Python Streamlit program that shows a thumbs up and thumbs down button for scoring an evaluation. When the user clicks, maintain a saved text file that tracks and shows the number of clicks with a refresh and sorts responses by the number of clicks."],
            ["📊 Create a Pandas DataFrame and display it using Streamlit. Use emojis to indicate the status of each row (e.g., ✅ for good, ❌ for bad)."],
            ["🗂 Using Gradio, create a simple interface where users can upload a CSV file and filter the data based on selected columns."],
            ["😃 Implement emoji reactions in a Streamlit app. When a user clicks on an emoji, record the click count in a Pandas DataFrame and display the DataFrame."],
            ["🔗 Create a program that fetches a dataset from Huggingface Hub and shows basic statistics about it using Pandas in a Streamlit app."],
            ["🤖 Use Gradio to create a user interface for a text summarizer model from Huggingface Hub."],
            ["📈 Create a Streamlit app to visualize time series data. Use Pandas to manipulate the data and plot it using Streamlit’s native plotting options."],
            ["🎙 Implement a voice-activated feature in a Gradio interface. Use a pre-trained model from Huggingface Hub for speech recognition."],
            ["🔍 Create a search function in a Streamlit app that filters through a Pandas DataFrame and displays the results."],
            ["🤗 Write a Python script that uploads a model to Huggingface Hub and then uses it in a Streamlit app."],
            ["👏 Create a Gradio interface for a clapping hands emoji (👏) counter. When a user inputs a text, the interface should return the number of clapping hands emojis in the text."],
            ["📜 Use Pandas to read an Excel sheet in a Streamlit app. Allow the user to select which sheet they want to view."],
            ["🔒 Implement a login screen in a Streamlit app using Python. Secure the login by hashing the password."],
            ["🤩 Create a Gradio interface that uses a model from Huggingface Hub to generate creative text based on a user’s input. Add sliders for controlling temperature and other hyperparameters."]
        ]
    )
    gr.HTML("""<h2>🤖 Mistral Chat - Gradio 🤖</h2>
        In this demo, you can chat with <a href='https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1'>Mistral-7B-Instruct</a> model. 💬
        Learn more about the model <a href='https://huggingface.co/docs/transformers/main/model_doc/mistral'>here</a>. 📚
        <h2>🛠 Model Features 🛠</h2>
        <ul>
          <li>🪟 Sliding Window Attention with 128K tokens span</li>
          <li>🚀 GQA for faster inference</li>
          <li>📝 Byte-fallback BPE tokenizer</li>
        </ul>
        <h3>📜 License 📜  Released under Apache 2.0 License</h3>
        <h3>📦 Usage 📦</h3>
        <ul>
          <li>📚 Available on Huggingface Hub</li>
          <li>🐍 Python code snippets for easy setup</li>
          <li>📈 Expected speedups with Flash Attention 2</li>
        </ul>
    """)

    markdown="""
    | Feature | Description | Byline |
    |---------|-------------|--------|
    | 🪟 Sliding Window Attention with 128K tokens span | Enables the model to have a larger context for each token. | Increases model's understanding of context, resulting in more coherent and contextually relevant outputs. |
    | 🚀 GQA for faster inference | Graph Query Attention allows faster computation during inference. | Speeds up the model inference time without sacrificing too much on accuracy. |
    | 📝 Byte-fallback BPE tokenizer | Uses Byte Pair Encoding but can fall back to byte-level encoding. | Allows the tokenizer to handle a wider variety of input text while keeping token size manageable. |
    | 📜 License | Released under Apache 2.0 License | Gives you a permissive free software license, allowing you freedom to use, modify, and distribute the code. |
    | 📦 Usage | | |
    | 📚 Available on Huggingface Hub | The model can be easily downloaded and set up from Huggingface. | Makes it easier to integrate the model into various projects. |
    | 🐍 Python code snippets for easy setup | Provides Python code snippets for quick and easy model setup. | Facilitates rapid development and deployment, especially useful for prototyping. |
    | 📈 Expected speedups with Flash Attention 2 | Upcoming update expected to bring speed improvements. | Keep an eye out for this update to benefit from performance gains. |
# 🛠 Model Features and More 🛠
## Features
- 🪟 Sliding Window Attention with 128K tokens span  
  - **Byline**: Increases model's understanding of context, resulting in more coherent and contextually relevant outputs.
- 🚀 GQA for faster inference  
  - **Byline**: Speeds up the model inference time without sacrificing too much on accuracy.
- 📝 Byte-fallback BPE tokenizer  
  - **Byline**: Allows the tokenizer to handle a wider variety of input text while keeping token size manageable.
- 📜 License: Released under Apache 2.0 License  
  - **Byline**: Gives you a permissive free software license, allowing you freedom to use, modify, and distribute the code.
## Usage 📦
- 📚 Available on Huggingface Hub  
  - **Byline**: Makes it easier to integrate the model into various projects.
- 🐍 Python code snippets for easy setup  
  - **Byline**: Facilitates rapid development and deployment, especially useful for prototyping.
- 📈 Expected speedups with Flash Attention 2  
  - **Byline**: Keep an eye out for this update to benefit from performance gains.
    """
    gr.Markdown(markdown)  
    
            
    def SpeechSynthesis(result):
        documentHTML5='''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Read It Aloud</title>
            <script type="text/javascript">
                function readAloud() {
                    const text = document.getElementById("textArea").value;
                    const speech = new SpeechSynthesisUtterance(text);
                    window.speechSynthesis.speak(speech);
                }
            </script>
        </head>
        <body>
            <h1>🔊 Read It Aloud</h1>
            <textarea id="textArea" rows="10" cols="80">
        '''
        documentHTML5 = documentHTML5 + result
        documentHTML5 = documentHTML5 + '''
            </textarea>
            <br>
            <button onclick="readAloud()">🔊 Read Aloud</button>
        </body>
        </html>
        '''
        gr.HTML(documentHTML5)
        # components.html(documentHTML5, width=1280, height=1024)
        #return result
    SpeechSynthesis(markdown)
    
    
demo.queue().launch(debug=True)

# Program 2:

import streamlit as st
from huggingface_hub import InferenceClient

# Mistral model client
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.1")

# Function to format the prompt for Mistral
def format_prompt(message, history):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt

# Function to generate text using Mistral
def generate(prompt, history):
    generate_kwargs = {
        'temperature': 0.9,
        'max_new_tokens': 256,
        'top_p': 0.95,
        'repetition_penalty': 1.0,
        'do_sample': True,
        'seed': 42
    }

    formatted_prompt = format_prompt(prompt, history)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text

    return output

# Medical expert roles and their functions
EXPERT_ROLES = {
    'Triage and Check-in Expert': 'triage_checkin',
    'Lab Analyst': 'lab_analyst',
    'Medicine Specialist': 'medicine_specialist',
    'Service Expert': 'service_expert',
    'Level of Care Expert': 'care_expert',
    'Terminology Expert': 'terminology_expert',
    'Chief Medical Officer': 'cmo',
    'Medical Director Team': 'medical_director',
}

def triage_checkin():
    st.write("### Triage and Check-in Expert 🚑")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Triage")

# ... (all the other role functions similar to triage_checkin)

def medical_director():
    st.write("### Medical Director Team 🏢")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Medical Director")

# Main application
def main():
    st.title("Mixture of Medical Experts Model")
    st.write("Harness the power of AI with this specialized healthcare framework! 🎉")

    role = st.selectbox("Select AI Role:", list(EXPERT_ROLES.keys()))

    st.write("#### Generate a Prompt")
    prompt = st.text_input("Enter your prompt:")
    if st.button("Generate Response"):
        history = []  # Assuming an empty history for this example
        response = generate(prompt, history)
        st.write(f"Generated Response: {response}")
        st.audio(response, format="audio/wav")

    # Run the selected role's function
    globals()[EXPERT_ROLES[role]]()

if __name__ == "__main__":
    main()




# Prompt

Create a mixture of experts model based on below CHARMSED model where you have eight roles that each contain a multiple 10 line question form which presents a system prompt for a role.  Set the roles so that they follow a Doctor specialty expert role model where there are eight general experts in doctor specialties and nursing specialties that generate output for the following tasks:  1. Triage and Check in for patients, 2. a lab analyst that studies condition lists and produces a lab, test, summary.  3. a medicine specialist that understands conditions over time against controlled and uncontrolled status with drugs at the time.  4. A service expert that is empathetic to patient cost and invasiveness of treatments, knows treatment and can list a top ten service codes for a list of diagnosis conditions.  5. A level of care expert that understands the mixture of doctor specialties a given patient needs and can name the specialty taxonomy code and descriptions.  6. A terminology expert which mastery over USMLE for medical licensing which is also a psychiatrist and neurologist that can help augment intelligence of the doctor working with this multi system agent.  7. A chief medical officer. and 8. a medical director team with oversight of the payer, the patient, and the collection of providers and facilities like medical centers within driving distance of any patient by state.  Here is the CHARMSED model:  Harness the power of AI with the CHARMSED framework.

This suite of roles brings together a comprehensive set of AI capabilities, tailored for diverse tasks:
Coder 💻: Craft pythonic solutions with precision.
Humanities Expert 📚: Dive deep into arts, literature, and history.
Analyst 🤔: Derive insights through logical reasoning.
Roleplay Expert 🎭: Mimic behaviors or adopt personas for engaging interactions.
Mathematician ➗: Crunch numbers and solve mathematical enigmas.
STEM Expert 🔬: Navigate through the realms of Science, Technology, Engineering, and Mathematics.
Extraction Expert 🔍: Extract concise information with a laser-focus.
Drafter 📝: Generate textual content and narratives with flair.
Empower your tasks with the perfect AI role and unleash the magic of CHARMSED!

Select AI Role:

# Program:

import streamlit as st

def triage_checkin():
    st.write("### Triage and Check-in Expert 🚑")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Triage")

def lab_analyst():
    st.write("### Lab Analyst 🧪")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Lab Analysis")

def medicine_specialist():
    st.write("### Medicine Specialist 💊")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Medicine")

def service_expert():
    st.write("### Service Expert 💲")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Service")

def care_expert():
    st.write("### Level of Care Expert 🏥")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Level of Care")

def terminology_expert():
    st.write("### Terminology Expert 📚")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Terminology")

def cmo():
    st.write("### Chief Medical Officer 🩺")
    for i in range(1, 11):
        st.text_input(f"Question {i} for CMO")

def medical_director():
    st.write("### Medical Director Team 🏢")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Medical Director")

def main():
    st.title("Mixture of Medical Experts Model")
    st.write("Harness the power of AI with this specialized healthcare framework! 🎉")

    role = st.selectbox("Select AI Role:", [
        "Triage and Check-in Expert",
        "Lab Analyst",
        "Medicine Specialist",
        "Service Expert",
        "Level of Care Expert",
        "Terminology Expert",
        "Chief Medical Officer",
        "Medical Director Team"
    ])

    if role == "Triage and Check-in Expert":
        triage_checkin()
    elif role == "Lab Analyst":
        lab_analyst()
    elif role == "Medicine Specialist":
        medicine_specialist()
    elif role == "Service Expert":
        service_expert()
    elif role == "Level of Care Expert":
        care_expert()
    elif role == "Terminology Expert":
        terminology_expert()
    elif role == "Chief Medical Officer":
        cmo()
    elif role == "Medical Director Team":
        medical_director()

if __name__ == "__main__":
    main()

